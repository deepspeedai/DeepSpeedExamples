from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Response:
    prompt_text: str = ""
    prompt_tokens: int = 0
    generated_output: str = ""
    generated_tokens: int = 0
    request_time: float = 0
    raw_response: Any = None
    client_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
