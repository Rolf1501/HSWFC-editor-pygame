from dataclasses import fields, is_dataclass


def get_init_field_names(cls):
    if is_dataclass(cls):
        return [f.name for f in fields(cls) if f.init]


def dict_to_dataclass_instance(cls, dct: dict):
    fields = get_init_field_names(cls)
    return cls(**{f: dct[f] for f in fields})
