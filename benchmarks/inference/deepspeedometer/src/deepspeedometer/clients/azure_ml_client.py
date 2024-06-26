import json
import requests
from typing import Any, Dict

from loguru import logger

from .base import BaseClient
from ..config import BaseConfigModel
from ..prompt import Prompt


class AzureMLClientConfig(BaseConfigModel):
    api_url: str = ""
    """ URL for the AzureML REST API. """

    api_key: str = ""
    """ REST API key for the AzureML deployment. """

    deployment_name: str = ""
    """ Name of the AzureML deployment. """


class AzureMLClient(BaseClient):
    def __init__(self, config: AzureMLClientConfig) -> None:
        super().__init__(config)
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.deployment_name = config.deployment_name

    def start_service(self) -> None:
        # Verify that the server exists, this could be extended to actually
        # start an AML deployment. However currently we assume one exists.
        test_prompt = Prompt("hello world", num_tokens=5, max_new_tokens=16)
        _ = self.process_response(self.send_request(self.prepare_request(test_prompt)))

    def stop_service(self) -> None:
        pass

    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        # TODO: add support for OpenAI chat completion template
        if prompt.streaming:
            raise ValueError("AzureMLClient does not support streaming prompts.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
            "azureml-model-deployment": self.deployment_name,
        }
        pload = {
            "input_data": {
                "input_string": [
                    prompt.text,
                ],
                "parameters": {
                    "max_tokens": prompt.max_new_tokens,
                    "return_full_text": prompt.return_full_text,
                },
            }
        }
        return {"url": self.api_url, "headers": headers, "json": pload, "timeout": 180}

    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        while True:
            try:  # Sometimes the AML endpoint will return an error, so we send the request again
                response = requests.post(**request_kwargs)
                output = json.loads(response.content)
                assert (
                    response.status_code == 200
                ), f"Status code: {response.status_code}"
                assert output[0]["0"], f"Empty response"
                break
            except Exception as e:
                logger.debug(f"Connection failed with {e}. Retrying AML request")

        return output

    def process_response(self, raw_response: Any) -> str:
        response_text = raw_response[0]["0"]
        return response_text
