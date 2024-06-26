import time
import random
from typing import Any, Dict

from transformers import AutoTokenizer

from .base import BaseClient
from ..config import BaseConfigModel
from ..prompt import Prompt


class DummyClientConfig(BaseConfigModel):
    model: str
    dummy_client_latency_time: float = 0.1


class DummyClient(BaseClient):
    def __init__(self, config: DummyClientConfig) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.latency_time = config.dummy_client_latency_time

    def start_service(self) -> None:
        pass

    def stop_service(self) -> None:
        pass

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        return {"input_text": prompt.text, "max_new_tokens": prompt.max_new_tokens}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        time.sleep(
            abs(random.uniform(self.latency_time - 0.1, self.latency_time + 0.2))
        )
        output_text = self.tokenizer.decode(
            random.choices(
                self.tokenizer.encode(request_kwargs["input_text"]),
                k=request_kwargs["max_new_tokens"],
            )
        )
        return output_text

    def process_response(self, raw_response: Any) -> str:
        return raw_response
