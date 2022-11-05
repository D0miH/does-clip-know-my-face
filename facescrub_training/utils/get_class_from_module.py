from types import ModuleType


def get_class_from_module(module: ModuleType, class_name: str):
    """
    Gets a class from a module.
    """
    if hasattr(module, class_name):
        module_class = getattr(module, class_name)
    else:
        classes_in_module = dict([(name, cls) for name, cls in module.__dict__.items() if isinstance(cls, type)])
        raise Exception(
            f'Did not find class with name {class_name}. Possible values are {[x for x in classes_in_module.keys()]}'
        )

    return module_class
