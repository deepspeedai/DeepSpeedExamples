import os
from dataclasses import dataclass
from typing import Iterable, Optional
from typing_extensions import Self

import numpy as np
import torch
from loguru import logger
from pydantic import model_validator
from transformers import AutoTokenizer

from .config import BaseConfigModel

# Avoids a warning from transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class Prompt:
    text: str
    num_tokens: int
    max_new_tokens: int
    streaming: bool = False
    return_full_text: bool = False
    request_kwargs: dict = None


class PromptConfig(BaseConfigModel):
    model: str
    """ Names of the model used to benchmark. Used to load the model/tokenizer from HuggingFace.co. """

    prompt_generator_seed: Optional[int] = None
    """ Seed value for prompt generator. """

    max_prompt_length: int = 4000
    """ Maximum prompt length for any request. """

    prompt_length: int = 2600
    """ Mean prompt length for requests. """

    prompt_length_var: float = 0.3
    """ Variance of prompt length. """

    max_new_tokens: int = 60
    """ Mean number of new tokens to generate in each request. """

    max_new_tokens_var: float = 0.3
    """ Variance of new tokens to generate. """

    streaming: bool = False
    """ Whether to enable streaming mode for the client. """

    @model_validator(mode="after")
    def set_max_prompt_length(self) -> Self:
        if self.prompt_length > self.max_prompt_length:
            logger.warning(
                f"Prompt length {self.prompt_length} is greater than max prompt length {self.max_prompt_length}. Setting max prompt length to {self.prompt_length}."
            )
        self.max_prompt_length = max(self.max_prompt_length, self.prompt_length)
        return self


class PromptGenerator:
    def __init__(self, model: str, prompt_text_source: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if os.path.isfile(prompt_text_source):
            with open(prompt_text_source, "r") as f:
                prompt_text_source = f.read()
        self.input_text = prompt_text_source
        self.tokenized_input = self.tokenizer.encode(
            self.input_text, return_tensors="pt", padding=False
        )[0]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def __call__(self, config: PromptConfig, num_prompts: int) -> Iterable[Prompt]:
        tokenized_input = self.tokenized_input
        if len(tokenized_input) < config.max_prompt_length:
            tokenized_input = torch.cat(
                [
                    tokenized_input
                    for _ in range(config.max_prompt_length // len(tokenized_input) + 1)
                ]
            ).flatten()

        if config.prompt_generator_seed is not None:
            np.random.seed(config.prompt_generator_seed)

        for _ in range(num_prompts):
            # Take the absolute value here because sometimes the normal
            # distribution will return a negative value. This is technically
            # wrong, but works out OK for most scenarios.
            prompt_length = min(
                abs(
                    int(
                        np.random.normal(
                            config.prompt_length,
                            config.prompt_length * config.prompt_length_var,
                        )
                    )
                ),
                config.max_prompt_length,
            )
            max_new_tokens = abs(
                int(
                    np.random.normal(
                        config.max_new_tokens,
                        config.max_new_tokens * config.max_new_tokens_var,
                    )
                )
            )
            yield Prompt(
                text=self.tokenizer.decode(tokenized_input[:prompt_length]),
                num_tokens=prompt_length,
                max_new_tokens=max_new_tokens,
            )
