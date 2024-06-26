import time
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import Field

from .base import BaseClient
from ..config import BaseConfigModel
from ..prompt import Prompt


class FastGenClientConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    deployment_name: str = "fastgen-benchmark-deployment"
    tp_size: int = 1
    num_replicas: int = 1
    max_ragged_batch_size: int = 768
    quantization_mode: Optional[str] = None


class FastGenClient(BaseClient):
    def __init__(self, config: FastGenClientConfig):
        super().__init__(config)
        try:
            import mii
        except ImportError as e:
            logger.error(
                "Please install the `deepspeed-mii` package to use this client."
            )
            raise e

        self.mii_client = mii.client(config.deployment_name)
        self.streaming = config.streaming

    def start_service(self) -> None:
        import mii
        from deepspeed.inference import RaggedInferenceEngineConfig, DeepSpeedTPConfig
        from deepspeed.inference.v2.ragged import DSStateManagerConfig

        tp_config = DeepSpeedTPConfig(tp_size=self.config.tp_size)
        mgr_config = DSStateManagerConfig(
            max_ragged_batch_size=self.config.max_ragged_batch_size,
            max_ragged_sequence_count=self.config.max_ragged_batch_size,
        )
        inference_config = RaggedInferenceEngineConfig(
            tensor_parallel=tp_config, state_manager=mgr_config
        )
        mii.serve(
            self.config.model,
            deployment_name=self.config.deployment_name,
            tensor_parallel=self.config.tp_size,
            inference_engine_config=inference_config,
            replica_num=self.config.num_replicas,
            quantization_mode=self.config.quantization_mode,
        )

    def stop_service(self) -> None:
        import mii

        mii.client(self.config.deployment_name).terminate_server()

    def _streaming_callback(self, raw_response) -> None:
        self.streaming_response_tokens.append(raw_response[0].generated_text)
        time_now = time.time()
        self.streaming_token_gen_time.append(time_now - time_last_token)
        time_last_token = time_now

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        request_kwargs = {
            "prompts": prompt.text,
            "max_new_tokens": prompt.max_new_tokens,
        }
        if self.streaming:
            self.streaming_response_tokens = []
            self.streaming_token_gen_time = []
            self.streaming_time_last_token = None
            request_kwargs["streaming_fn"] = self._streaming_callback
        return request_kwargs

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        if self.streaming:
            self.streaming_time_last_token = time.time()
        response = self.mii_client(**request_kwargs)
        if self.streaming:
            response = self.streaming_response_tokens

        return response

    def process_response(self, raw_response: Any) -> str:
        if not self.streaming:
            return raw_response[0].generated_text
