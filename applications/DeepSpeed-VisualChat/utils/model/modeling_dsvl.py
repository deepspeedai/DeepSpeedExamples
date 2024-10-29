import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor 
from .third_party_model.hf_model.modeling_llama import LlamaForCausalLM
from .third_party_model.hf_model.configuration_llama import LlamaConfig
from .third_party_model.qwen_clip.qwen_clip import VisionTransformer
from torch import nn
from torch.nn import  CrossEntropyLoss
import copy
import os
import sys
from ..data import build_dataset, DataCollatorPadToMaxLen, add_special_token
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import data.DST as DST # default special tokens
from torch.utils.data import DataLoader
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import numpy as np
from .vis_proj import VisProjection_vit, VisProjection_perceiver

def get_name(huggingface_path):
    if 'opt' in huggingface_path.lower():
        return 'opt'
    elif 'gpt2' in huggingface_path.lower():
        return 'gpt2'
    elif 'llama-2' in huggingface_path.lower():
        return 'llama-2'
    else:
        raise ValueError('We currently only support llama, opt and gpt2')

def create_dsvl_model_and_transforms(
        text_tokenizer=None,
        ds_config=None,
        args=None):
    assert args.vision_model_name_or_path is not None
    assert args.lm_model_name_or_path is not None
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        dschf = HfDeepSpeedConfig(ds_config)
    lang_config = AutoConfig.from_pretrained(args.lm_model_name_or_path)


    if 'qwen' in args.vision_model_name_or_path.lower():
        # use a fake config for consistent
        vis_config = AutoConfig.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        vis_config = vis_config.vision_config
        vis_encoder = VisionTransformer(
            image_size=448,
            patch_size=vis_config.patch_size,
            width=vis_config.hidden_size,
            layers=vis_config.num_hidden_layers,
            heads=vis_config.num_attention_heads,
            mlp_size=vis_config.intermediate_size,
            output_dim=4096,
        ) 
        vis_encoder.load_state_dict(torch.load(os.path.join(args.vision_model_name_or_path, 'pytorch_model.bin'), map_location='cpu'), strict=True)
        vis_config.hidden_size = 4096 # we need to change the hidden size to 4096
    elif 'clip' in args.vision_model_name_or_path.lower():
        vis_encoder = CLIPVisionModel.from_pretrained(args.vision_model_name_or_path) 
        vis_config = vis_encoder.config
    else:
        raise ValueError("We currently only support qwen's modifed clip and other clip models")
    
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model_name_or_path)
    
    tokenizer = add_special_token(text_tokenizer)  
    tokenizer.pad_token = tokenizer.eos_token
    if 'llama' in args.lm_model_name_or_path.lower():
        lang_config = LlamaConfig.from_pretrained(args.lm_model_name_or_path)
        lang_config.enable_mmca_attention = args.enable_mmca_attention
        lang_config.max_position_embeddings = args.max_seq_len
    
    if 'llama' in args.lm_model_name_or_path.lower():
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            lang_decoder = LlamaForCausalLM.from_pretrained(args.lm_model_name_or_path, config=lang_config)
        else:
            try:
                device = torch.device("cuda", args.local_rank)
            except:
                device = "auto"
            lang_decoder = LlamaForCausalLM.from_pretrained(args.lm_model_name_or_path, config=lang_config, device_map=device)
        decoder_name = 'llama'
    else:
        raise NotImplemented("We for now only support LLaMA family and do not support other models yet")
    
    lang_config.vocab_size = len(tokenizer)
    lang_decoder.resize_token_embeddings(len(tokenizer))
    model = DeepSpeedViLModel(vis_encoder, lang_decoder, \
                                tokenizer, \
                                vis_config=vis_config, \
                                decoder_name=decoder_name, \
                                lang_config=lang_config, \
                                max_seq_length=args.max_seq_len,
                                args=args)
    
    return model, image_processor, tokenizer


class DeepSpeedViLModel(nn.Module):
    def __init__(self, vis_encoder,
                    lang_decoder,
                    tokenizer,
                    vis_config=None, 
                    decoder_name='gpt2',
                    lang_config=None,
                    max_seq_length=512,
                    args=None):
        super().__init__()
        self.vis_encoder = vis_encoder
         
        self.lang_decoder = lang_decoder 
        self.tokenizer = tokenizer 
        self.args = args
        self._enable_special_token()

        self.lang_config = lang_config
        self._get_model_stat(decoder_name)
        lang_embed, pos_embedding = self._languag_embedding()
        self.pos_embedding = pos_embedding
        self.max_seq_length = max_seq_length
        if lang_embed is None:
            print ('randomly initialized a language embedding')
            self.lang_embed = nn.Embedding(self.lang_config.vocab_size,\
                                            self.hidden_size,\
                                            self.pad_token_id) # randomly initialized language embedder
        else:
            self.lang_embed = lang_embed

        self.pos_embedding = pos_embedding
        self.projection = self.build_projection(vis_config, self.lang_config.hidden_size)   
        self._init_weight()
        

        # get padding token embedding
        self.padding_embedding = None 
        self.vis_encoder_update = None

    def _enable_special_token(self):
        self.DEFAULT_IMAGE_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(DST.DEFAULT_IMAGE_TOKEN)
        self.DEFAULT_IMAGE_PATCH_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(DST.DEFAULT_IMAGE_PATCH_TOKEN)
        self.DEFAULT_IM_START_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(DST.DEFAULT_IM_START_TOKEN)
        self.DEFAULT_IM_END_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(DST.DEFAULT_IM_END_TOKEN)

        
    def _get_model_stat(self, model_name):   
        config_dic = {
            'llama-2': ['max_position_embeddings','num_hidden_layers'],
            'llama': ['max_position_embeddings','num_hidden_layers'],
            'gpt2': ['n_positions','n_layer'],
            'opt': ['max_position_embeddings','num_hidden_layers']
        }
        pos_name, layer_name = config_dic[model_name][0], config_dic[model_name][1]
        self.n_positions = getattr(self.lang_config, pos_name)
        self.num_layer = getattr(self.lang_config, layer_name)
        self.hidden_size  = getattr(self.lang_config, 'hidden_size')
        self.vocab_size = getattr(self.lang_config, 'vocab_size')
        
    def _languag_embedding(self):
        pos_embedding = None
        token_embedding = None
        for name, module in self.lang_decoder.named_modules():
            if isinstance(module, nn.Embedding):
                try:
                    # z3 shape
                    rows = module.weight.ds_shape[0]
                except:
                    rows = module.weight.size()[0]
                     
                if rows == self.vocab_size:
                    token_embedding = copy.deepcopy(module)
                if rows == self.n_positions:
                    pos_embedding = copy.deepcopy(module)
        return token_embedding, pos_embedding
     
        
    def _init_weight(self):
        self.vis_encoder.requires_grad_(False)  
        self.lang_decoder.requires_grad_(False)  
        self.lang_embed.requires_grad_(True)   
        self.projection.requires_grad_(True) 
        if  self.pos_embedding  is not None:     
            self.pos_embedding.requires_grad_(True) 
        

    def build_projection(self, vis_config, lang_dim):
        if self.args.vis_proj == 'vit':
            output =  VisProjection_vit(vis_config, lang_dim=lang_dim)
            return output 
        elif self.args.vis_proj == 'baseline':
            return nn.Sequential( 
                            nn.Linear(vis_config.hidden_size, lang_dim), # an example implementation
                            nn.LayerNorm(lang_dim, eps=1e-12))
        elif self.args.vis_proj == 'perceiver':
            return VisProjection_perceiver(vis_config, lang_dim=lang_dim)

    def concat(self, img_proj, lang, attention_mask, input_labels, image_num, do_generation=False):
        output_lang = []
        output_attention_mask = []
        output_input_labels = []

        def split_tensor_by_a_list(tensor, split_list):
            output = []
            initial_pos = 0
            accumulated_sum = [sum(split_list[:i]) for i in range(1, len(split_list)+1)]
            for pos in accumulated_sum:
                output.append(tensor[initial_pos:pos])
                initial_pos = pos
            del tensor
            return output
        
        img_proj = split_tensor_by_a_list(img_proj, image_num)
        
        for index in range(len(img_proj)): # each seq has multi iamges, so we need to use it as index
            initial_pos = 0
            cur_img = img_proj[index]
            cur_lang = lang[index]
            cur_attention_mask = attention_mask[index]
            cur_input_labels = input_labels[index]
            img_pos_list = cur_lang.eq(self.DEFAULT_IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
            assert len(img_pos_list) == image_num[index], "the number of images in the lang and image_num does not match"
            if len(img_pos_list) == 0:
                continue # there is no image probably it is a pure text insturctio
            
            cur_lang = self.lang_embed(cur_lang) # get the real embedding
            for img_i, img_pos in zip(cur_img, torch.flip(img_pos_list, dims=(0,))): # do it reversely so that we can easily insert the image
                lang_pre_img_embed = cur_lang[initial_pos:img_pos]
                attention_mask_pre_img = cur_attention_mask[initial_pos:img_pos]
                input_labels_pre_img = cur_input_labels[initial_pos:img_pos]

                lang_post_img_embed = cur_lang[img_pos+1:]
                attention_mask_post_img = cur_attention_mask[img_pos+1:]
                input_labels_post_img = cur_input_labels[img_pos+1:]
                # now we need to concat the image embedding
                lang_full = torch.cat((lang_pre_img_embed, img_i, lang_post_img_embed), dim=0)
                # label the position of all images as 2 instead of 1
    
                attention_mask_full = torch.cat( (attention_mask_pre_img, 2 * torch.ones_like(img_i[:, 0]), attention_mask_post_img), dim=0)

                input_labels_full = torch.cat((input_labels_pre_img.long(), DST.DEFAULT_LABEL_PADDING_NUM * torch.ones_like(img_i[:, 0], dtype=torch.long), input_labels_post_img),   dim=0)

                cur_lang = lang_full
                cur_attention_mask = attention_mask_full
                cur_input_labels = input_labels_full
            # append to the output 
            output_lang.append(lang_full.unsqueeze(0))
            output_attention_mask.append(attention_mask_full.unsqueeze(0))
            output_input_labels.append(input_labels_full.unsqueeze(0))

        if self.padding_embedding is None:
            with torch.no_grad():
                self.padding_embedding = self.lang_embed(torch.tensor(self.tokenizer.pad_token_id).to(lang.device).unsqueeze(0)).unsqueeze(0).detach()

        def pad_tensor_list(tensor_list, pad_token_id, pad_vec=False):
            max_len = max([tensor.size(1) for tensor in tensor_list])
            if not do_generation:
                max_len = int(np.ceil(max_len / 8) * 8) # make it divisible by 8
            padded_tensor_list = []
            for tensor in tensor_list:
                if max_len > tensor.size(1):
                    if pad_vec: # output_lang padding
                        # pad with self.padding_embedding 
                        padded_tensor = torch.cat([tensor] + [self.padding_embedding] * (max_len - tensor.size(1)), dim=1)
                        
                    else:
                        padded_tensor = F.pad(tensor, (0, max_len - tensor.size(1)), value=pad_token_id)
                else:
                    padded_tensor = tensor
                padded_tensor_list.append(padded_tensor)
            return padded_tensor_list
        output_lang = pad_tensor_list(output_lang, self.tokenizer.pad_token_id, pad_vec=True)
        output_attention_mask = pad_tensor_list(output_attention_mask, 0)
        output_input_labels = pad_tensor_list(output_input_labels, DST.DEFAULT_LABEL_PADDING_NUM)

        return torch.cat(output_lang, dim=0), torch.cat(output_attention_mask, dim=0), torch.cat(output_input_labels, dim=0)

    def forward(self, img, lang, 
            attention_mask=None,
            input_labels=None,
            image_num=1,
            past_key_values=None,
            use_cache=False,
            output_attentions=False, 
            output_hidden_states=False,
            return_dict=True):
        
        assert attention_mask is not None, "attention mask is required"
        assert input_labels is not None, "input labels is required"

        if self.vis_encoder_update is None:
            self.vis_encoder_update = False # default is False
            for p in self.vis_encoder.parameters():
                if p.requires_grad:
                    self.vis_encoder_update = True
        # this part for now does not require gradient
        if self.vis_encoder_update:
            # update vis encoder
            img_feature = self.vis_encoder(img) 
            if not isinstance(img_feature, torch.Tensor):
                img_feature = img_feature.last_hidden_state
        else:
            # do not update vis encoder
            with torch.no_grad():
                img_feature = self.vis_encoder(img)
                if not isinstance(img_feature, torch.Tensor):
                    img_feature = img_feature.last_hidden_state
        img_proj = self.projection(img_feature)
       
        hidden_states, attention_mask, input_labels = self.concat(img_proj, lang, attention_mask, input_labels, image_num)
        labels = input_labels   
            
        if self.pos_embedding is not None:
            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)
            position_ids = torch.arange(past_length, hidden_states.size()[1] + past_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, hidden_states.size()[1])
            position_embeds = self.pos_embedding(position_ids)
            hidden_states = hidden_states + position_embeds
            
        logits = self.lang_decoder(input_ids=None, 
                                    inputs_embeds=hidden_states,
                                    attention_mask=attention_mask,
                                    labels=None,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions, 
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict).logits
        
        
        logits_shift = logits[..., :-1, :].contiguous().view(-1, self.vocab_size) # remove the last token
        labels_shift = labels[..., 1:].contiguous().to(logits_shift.device).view(-1) # remove the first token
        # select index that is not -100
        labels_index = labels_shift != -100
        if torch.sum(labels_index) ==0:
            logits_shift = logits_shift[-2:,:].contiguous()
            labels_shift = labels_shift[-2:].contiguous()            
        else:
            logits_shift = logits_shift[labels_index,:].contiguous()
            labels_shift = labels_shift[labels_index].contiguous()

        loss_fct = CrossEntropyLoss() 
        loss = loss_fct(logits_shift, labels_shift) 
        
        return [loss,] 
    
    @torch.no_grad()
    def generate(self, img, lang, 
            attention_mask=None,
            input_labels=None,
            generation_length=128,
            generation_kwargs={}, # add some meaningful default values
            ):
        assert lang.size()[0] == 1, "only support batch size == 1 for now"
        attention_mask = torch.ones_like(lang) 
        input_labels = torch.ones_like(lang) 
        # this part for now does not require gradient
        img_feature = self.vis_encoder(img) 
        if not isinstance(img_feature, torch.Tensor):
            img_feature = img_feature.last_hidden_state
        img_proj = self.projection(img_feature)
        hidden_states, attention_mask, input_labels = self.concat(img_proj, lang, attention_mask, input_labels, image_num=[img.size(0)], do_generation=True)
        
        output = self.lang_decoder.generate(input_ids=None,
                                inputs_embeds=hidden_states,
                                attention_mask=attention_mask, # we need the mask to diff img and text
                                pad_token_id=self.tokenizer.pad_token_id,
                                max_new_tokens=generation_length, # this is the number of tokens you want to generate
                                **generation_kwargs)
        return (output, self.tokenizer.batch_decode(output, skip_special_tokens=True)[0])


    def gradient_checkpointing_enable(self):
        self.vis_encoder.gradient_checkpointing_enable()
        self.lang_decoder.gradient_checkpointing_enable()