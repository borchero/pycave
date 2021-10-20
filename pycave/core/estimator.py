from __future__ import annotations
import inspect
import logging
import pickle
from abc import ABC
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    get_args,
    get_origin,
    Optional,
    Sized,
    Type,
    TypeVar,
    Union,
)
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import DataParallelPlugin, DDP2Plugin, DDPSpawnPlugin
from torch.utils.data import DataLoader, Dataset
from pycave.data import (
    DistributedTensorBatchSampler,
    TensorBatchSampler,
    TensorDataLoader,
    UnrepeatedDistributedTensorBatchSampler,
)
from .exception import NotFittedError
from .module import ConfigModule

M = TypeVar("M", bound=ConfigModule)  # type: ignore
E = TypeVar("E", bound="Estimator")  # type: ignore

logger = logging.getLogger(__name__)


class Estimator(Generic[M], ABC):
    """
    Base estimator class from which all PyCave estimators should inherit.
    """

    # We have these as private and public property to properly generate documentation.
    _trainer: pl.Trainer
    _model: M

    def __init__(
        self,
        *,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        verbose: bool = False,
        default_params: Optional[Dict[str, Any]] = None,
        user_params: Optional[Dict[str, Any]] = None,
        overwrite_params: Optional[Dict[str, Any]] = None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose
        self.trainer_params_user = user_params
        self.trainer_params = {
            **dict(
                enable_checkpointing=False,
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=logger.getEffectiveLevel() <= logging.INFO,
                enable_model_summary=logger.getEffectiveLevel() <= logging.DEBUG,
            ),
            **(default_params or {}),
            **(user_params or {}),
            **(overwrite_params or {}),
        }

    @property
    def model_(self) -> M:
        """
        The fitted PyTorch module containing all estimated parameters.
        """
        return self._model

    @property
    def trainer_(self) -> pl.Trainer:
        """
        The PyTorch Lightning trainer used for running the model.
        """
        return self._trainer

    @property
    def _is_fitted(self) -> bool:
        try:
            getattr(self, "model_")
            return True
        except NotFittedError:
            return False

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    def load_model(self, model: M) -> None:
        """
        Loads the provided model that has been fitted previously by this estimator or manually
        without the use of the estimator.

        Args:
            model: The model to load. In case, this estimator is already fitted, this model
                overrides the existing fitted model.
        """
        self._init_trainer(overwrite=False)
        self._model = model

    def save(self, path: Path) -> None:
        """
        Saves the estimator to the provided directory. It saves a file named ``estimator.pickle``
        for the configuration of the estimator and additional files for the fitted model (if
        applicable). For more information on the files saved for the fitted model or for more
        customization, look at :meth:`get_params` and :meth:`pycave.core.ConfigModule.save`.

        Args:
            path: The directory to which all files should be saved.

        Note:
            This method may be called regardless of whether the estimator has already been fitted.

        Attention:
            Use this method with care. It uses :mod:`pickle` to store the configuration options of
            the estimator and is thus not necessarily backwards-compatible. Instead, consider
            using :meth:`pycave.core.ConfigModule.save` on the fitted model accessible via
            :attr:`model_`.
        """
        assert path.is_dir(), "Estimators can only be saved to a directory."

        with (path / "estimator.pickle").open("wb+") as f:
            pickle.dump(self.get_params(), f)

        if self._is_fitted:
            self.model_.save(path)

    @classmethod
    def load(cls: Type[E], path: Path) -> E:
        """
        Loads the estimator and (if available) the fitted model. See :meth:`save` for more
        information about the required filenames for loading.

        Args:
            path: The directory from which to load the estimator.

        Returns:
            The loaded estimator, either fitted or not, depending on the availability of the
            ``config.json`` file.
        """
        estimator = cls()
        with (path / "estimator.pickle").open("rb") as f:
            estimator.set_params(pickle.load(f))  # type: ignore

        if (path / "config.json").exists():
            model_cls = cls._get_model_class()
            model = model_cls.load(path)
            estimator.load_model(model)

        return estimator

    # ---------------------------------------------------------------------------------------------
    # SKLEARN INTERFACE

    def get_params(self, deep: bool = True) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """
        Returns the estimator's parameters as passed to the initializer.

        Args:
            deep: Ignored. For sklearn compatibility.

        Returns:
            The mapping from init parameters to values.
        """
        signature = inspect.signature(self.__class__.__init__)
        parameters = [p.name for p in signature.parameters.values() if p.name != "self"]
        return {p: getattr(self, p) for p in parameters}

    def set_params(self: E, values: Dict[str, Any]) -> E:
        """
        Sets the provided values on the estimator. The estimator is returned as well, but the
        estimator on which this function is called is also modified.

        Args:
            values: The values to set.

        Returns:
            The estimator where the values have been set.
        """
        for key, value in values.items():
            setattr(self, key, value)
        return self

    # ---------------------------------------------------------------------------------------------
    # DATA HANDLING

    def _data_collate_fn(
        self,
        for_tensor: bool,  # pylint: disable=unused-argument
    ) -> Optional[Callable[[Any], Any]]:
        """Overridable for custom collation function."""
        return None

    def _init_data_loader(
        self,
        data: Union[npt.NDArray[np.float32], torch.Tensor, Dataset[torch.Tensor]],
        *,
        for_training: bool,
    ) -> DataLoader[torch.Tensor]:
        # pylint: disable=assignment-from-none

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        batch_size = self.batch_size or len(data)  # type: ignore

        if isinstance(data, torch.Tensor):
            sampler_kwargs = self.trainer_.distributed_sampler_kwargs
            if sampler_kwargs is None:
                sampler = TensorBatchSampler(data, batch_size)
            elif for_training:
                sampler = DistributedTensorBatchSampler(data, batch_size, **sampler_kwargs)
            else:
                sampler = UnrepeatedDistributedTensorBatchSampler(
                    data, batch_size, **sampler_kwargs
                )
            # Although this is not actually a PyTorch data loader, we make the type checker think
            # that it is one so that downstream users of this utility function do not have to
            # handle this type explicitly.
            collate_fn = self._data_collate_fn(for_tensor=True)
            if collate_fn is not None:
                return TensorDataLoader(  # type: ignore
                    data,
                    sampler=sampler,
                    collate_fn=collate_fn,
                )
            return TensorDataLoader(data, sampler=sampler)  # type: ignore

        collate_fn = self._data_collate_fn(for_tensor=False)
        if collate_fn is not None:
            return DataLoader(
                data, num_workers=self.num_workers, batch_size=batch_size, collate_fn=collate_fn
            )
        return DataLoader(data, num_workers=self.num_workers, batch_size=batch_size)

    # ---------------------------------------------------------------------------------------------
    # HELPER METHODS

    def _init_trainer(
        self,
        *,
        overwrite: bool = True,
        updated_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "_trainer") or overwrite:
            self._trainer = pl.Trainer(**{**self.trainer_params, **(updated_params or {})})
            assert not isinstance(
                self.trainer_.training_type_plugin,
                (DDP2Plugin, DataParallelPlugin, DDPSpawnPlugin),
            ), (
                "Trainer is using an unsupported training type plugin. "
                "`ddp2`, `dp` and `ddp_spawn` are currently not supported."
            )

    def _uses_batch_training(self, data: Sized) -> bool:
        return (
            (self.batch_size is not None and len(data) / self.batch_size > 1)
            or self.trainer_params.get("gpus", 0) > 1
            or self.trainer_params.get("num_nodes", 1) > 1
            or self.trainer_params.get("num_processes", 1) > 1
            or self.trainer_params.get("tpu_cores", 0) > 1
            or self.trainer_params.get("ipus", 0) > 1
        )

    # ---------------------------------------------------------------------------------------------
    # SPECIAL METHODS

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        if key.endswith("_") and not key.endswith("__"):
            raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet")
        raise AttributeError(
            f"Attribute `{key}` does not exist on type `{self.__class__.__name__}`."
        )

    # ---------------------------------------------------------------------------------------------
    # GENERICS

    @classmethod
    def _get_model_class(cls: Type[E]) -> Type[M]:
        return cls._get_generic_type(0)

    @classmethod
    def _get_generic_type(cls: Type[E], index: int) -> Any:
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) == Estimator:
                args = get_args(base)
                if not args:
                    raise ValueError(
                        f"`{cls.__name__} does not provide at least {index+1} generic parameters"
                        " for `Estimator`"
                    )
                return get_args(base)[index]
        raise ValueError(f"`{cls.__name__}` does not inherit from `Estimator`")
