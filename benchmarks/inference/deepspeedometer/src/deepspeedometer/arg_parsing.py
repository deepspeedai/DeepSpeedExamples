import argparse
from typing import List, Tuple

from .benchmark_runner import BenchmarkConfig
from .clients import client_config_classes
from .config import BaseConfigModel


def parse_args_to_configs(args: List[str]) -> Tuple[BenchmarkConfig, BaseConfigModel]:
    def add_model(parser: argparse.ArgumentParser, model: BaseConfigModel):
        """Adds fields from pydantic model to the parser."""
        for name, field in model.model_fields.items():
            field_type = field.annotation

            # Get information about number of arguments expected
            nargs = None
            if getattr(field.annotation, "_name", "") == "List":
                nargs = "+"
                field_type = field.annotation.__args__[0]

            # Add field to parser
            parser.add_argument(
                f"--{name}",
                dest=name,
                nargs=nargs,
                type=field_type,
                required=getattr(field, "required", False),
                default=getattr(field, "default", None),
                help=getattr(field, "description", ""),
            )

    # Parse benchmark config fields
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_model(parser, BenchmarkConfig)
    benchmark_args, remaining_args = parser.parse_known_args(args)
    benchmark_config = BenchmarkConfig(**vars(benchmark_args))
    unused_args = set(remaining_args)

    # Parse client config fields
    client_config_class = client_config_classes[benchmark_config.api]
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_model(parser, client_config_class)
    client_args, remaining_args = parser.parse_known_args(args)
    client_config = client_config_class(**vars(client_args))

    # Check for unused arguments
    unused_args = unused_args.intersection(remaining_args)
    if unused_args:
        raise ValueError(f"Unused arguments: {unused_args}")

    return benchmark_config, client_config
