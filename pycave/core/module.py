import dataclasses
import json
from abc import ABC
from pathlib import Path
from typing import Generic, get_args, get_origin, Type, TypeVar
import torch
from torch import jit, nn

C = TypeVar("C")
M = TypeVar("M", bound="ConfigModule")  # type: ignore


class ConfigModule(nn.Module, Generic[C], ABC):
    """
    Base module for modules that can are configured with a configuration. The configuration type
    must be a Python dataclass.

    This base class allows these models to be easily saved and loaded.
    """

    @classmethod
    def load(cls: Type[M], path: Path) -> M:
        """
        Loads the module's configurations and parameters from files in the specified directory at
        first. Then, it initializes the model with the stored configurations and loads the
        parameters. This method is typically used after calling :meth:`save` on the model.

        Args:
            path: The directory which contains the ``config.json`` and ``parameters.pt`` files to
                load.

        Returns:
            The loaded model.
        """
        assert path.is_dir(), "Modules can only be loaded from a directory."

        config_cls = cls._get_config_class()
        with (path / "config.json").open("r") as f:
            config = config_cls(**json.load(f))

        model = cls(config)
        with (path / "parameters.pt").open("rb") as f:
            state_dict = torch.load(f)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def _get_config_class(cls: Type[M]) -> Type[C]:
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) == ConfigModule:
                args = get_args(base)
                if not args:
                    raise ValueError(
                        f"`{cls.__name__} does not provide a generic parameter for `ConfigModule`"
                    )
                return get_args(base)[0]
        raise ValueError(f"`{cls.__name__}` does not inherit from `ConfigModule`")

    def __init__(self, config: C):
        super().__init__()
        self.config = config

    @jit.unused
    def save(self, path: Path, compile_model: bool = False) -> None:
        """
        Saves the module's configuration and parameters to files in the specified directory. It
        creates two files, namely ``config.json`` and ``parameters.pt`` which contain the
        configuration and parameters, respectively.

        Args:
            path: The directory to which to save the configuration and parameter files.
            compile_model: Whether the model should be compiled via TorchScript. An additional file
                called `model.ptc` will then be stored. Note that you can simply load the compiled
                model via :meth:`torch.jit.load` at a later point.
        """
        assert path.is_dir(), "Modules can only be saved to a directory."

        with (path / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.config), f)
        with (path / "parameters.pt").open("wb+") as f:
            torch.save(self.state_dict(), f)

        if compile_model:
            compiled_model = jit.script(self)
            with (path / "model.ptc").open("wb+") as f:
                jit.save(compiled_model, f)
