from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import BaseConfigModel
from ..prompt import Prompt


class BaseClient(ABC):
    def __init__(self, config: BaseConfigModel) -> None:
        self.config = config

    @abstractmethod
    def start_service(self) -> None:
        pass

    @abstractmethod
    def stop_service(self) -> None:
        pass

    @abstractmethod
    def prepare_request(self, prompt: Prompt) -> Dict[str, Any]:
        pass

    @abstractmethod
    def send_request(self, request_kwargs: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def process_response(self, raw_response: Any) -> str:
        pass
