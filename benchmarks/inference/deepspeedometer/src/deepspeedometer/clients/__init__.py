from .base import BaseClient

from .azure_ml_client import AzureMLClientConfig, AzureMLClient
from .dummy_client import DummyClientConfig, DummyClient
from .fastgen_client import FastGenClientConfig, FastGenClient
from .vllm_client import vLLMClientConfig, vLLMClient

client_config_classes = {
    "dummy": DummyClientConfig,
    "azure_ml": AzureMLClientConfig,
    "fastgen": FastGenClientConfig,
    "vllm": vLLMClientConfig,
}
client_classes = {
    "dummy": DummyClient,
    "azure_ml": AzureMLClient,
    "fastgen": FastGenClient,
    "vllm": vLLMClient,
}
