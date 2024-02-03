from dataclasses import fields, is_dataclass


def get_init_field_names(cls):
    if is_dataclass(cls):
        return [f.name for f in fields(cls) if f.init]
