from pydantic import BaseModel, ConfigDict


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=False,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )
