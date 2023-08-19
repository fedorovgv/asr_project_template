from typing import Any

from omegaconf import DictConfig, OmegaConf

from asr.utils import import_class_by_path


class Serialization:
    """
    Base Class for all modules that allows to instantiate a class from a DictConfig object.
    """
    @staticmethod
    def from_config(cfg: DictConfig, **kwargs) -> Any:
        """
        Each module initizializated by the part of the config file in .yaml format. 
        The structure must follow pattern:
            module:
                target_cls: abc.ABC
                kwarg1: value1
                kwarg2: value2
                ...
        Then an initialized copy of abc.ABC(kwarg1=value1, ...) will be returned.
        """
        target_cls = cfg.target_cls
        imported_cls = import_class_by_path(target_cls)

        config_kwargs = cfg.copy()

        del config_kwargs.target_cls

        resolved_config_kwargs: dict = OmegaConf.to_container(
            config_kwargs, resolve=True,
        )
        resolved_config_kwargs.update(kwargs)

        return imported_cls(**resolved_config_kwargs)
