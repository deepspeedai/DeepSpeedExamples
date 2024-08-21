import os
import json
import requests
import subprocess
import time
from typing import Any, Dict

from loguru import logger
from pydantic import Field

from .base import BaseClient
from ..config import BaseConfigModel
from ..prompt import Prompt


# client to test any openai API
class openaiClientConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    url: str = "http://127.0.0.1:26500/v1/completions"


class openaiClient(BaseClient):
    def __init__(self, config: openaiClientConfig):
        super().__init__(config)

    def start_service(self) -> None:
        pass

    def stop_service(self) -> None:
        pass

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        api_url = self.config.url
        headers = {
            "User-Agent": "Benchmark Client",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        pload = {
            "prompt": prompt.text,
            "model": self.config.model,
            "n": 1,
            "use_beam_search": False,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": prompt.max_new_tokens,
            "ignore_eos": False,
        }
        return {"url": api_url, "headers": headers, "json": pload, "timeout": 180}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        response = requests.post(**request_kwargs)
        output = json.loads(response.content)
        return output

    def process_response(self, raw_response: Any) -> str:
        return raw_response["choices"][0]["text"]
