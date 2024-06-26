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


class vLLMClientConfig(BaseConfigModel):
    model: str = Field(..., description="HuggingFace.co model name")
    tp_size: int = 1
    port: int = 26500


class vLLMClient(BaseClient):
    def __init__(self, config: vLLMClientConfig):
        super().__init__(config)
        try:
            import vllm
        except ImportError as e:
            logger.error("Please install the `vllm` package to use this client.")
            raise e

    def start_service(self) -> None:
        vllm_cmd = (
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config.port),
            "--tensor-parallel-size",
            str(self.config.tp_size),
            "--model",
            self.config.model,
        )
        p = subprocess.Popen(
            vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_fds=True
        )
        start_time = time.time()
        timeout_after = 60 * 5  # 5 minutes
        while True:
            line = p.stderr.readline().decode("utf-8")
            if "Application startup complete" in line:
                break
            if "error" in line.lower():
                p.terminate()
                # self.stop_service(config)
                raise RuntimeError(f"Error starting VLLM server: {line}")
            if time.time() - start_time > timeout_after:
                p.terminate()
                # self.stop_service(config)
                raise TimeoutError("Timed out waiting for VLLM server to start")
            time.sleep(0.01)

    def stop_service(self) -> None:
        vllm_cmd = ("pkill", "-f", "vllm.entrypoints.api_server")
        p = subprocess.Popen(vllm_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        api_url = "http://localhost:26500/generate"
        headers = {"User-Agent": "Benchmark Client"}
        pload = {
            "prompt": prompt.text,
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
        return raw_response["text"]
