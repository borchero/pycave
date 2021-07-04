from __future__ import annotations
import inspect
import pickle
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Generic, get_args, get_origin, Type, TypeVar
from .exception import NotFittedError
from .module import ConfigModule

M = TypeVar("M", bound=ConfigModule)  # type: ignore
E = TypeVar("E", bound="Estimator")  # type: ignore


class Estimator(Generic[M], ABC):
    """
    Base estimator class from which all PyCave estimators should inherit.
    """

    @classmethod
    def load(cls: Type[E], path: Path) -> E:
        """
        Loads the estimator and (if available) the fitted model. See :meth:`save` for more
        information about the required filenames for loading.

        Args:
            path: The directory from which to load the estimator.

        Returns:
            The loaded estimator, either fitted or not, depending on the availability of the
            :code:`config.json` file.
        """
        estimator = cls()
        with (path / "estimator.pickle").open("rb") as f:
            estimator.set_params(pickle.load(f))  # type: ignore

        if (path / "config.json").exists():
            model_cls = cls._get_model_class()
            model = model_cls.load(path)
            estimator.load_model(model)

        return estimator

    @classmethod
    def _get_model_class(cls: Type[E]) -> Type[M]:
        for base in cls.__orig_bases__:  # type: ignore
            if get_origin(base) == Estimator:
                args = get_args(base)
                if not args:
                    raise ValueError(
                        f"`{cls.__name__} does not provide a generic parameter for `Estimator`"
                    )
                return get_args(base)[0]
        raise ValueError(f"`{cls.__name__}` does not inherit from `Estimator`")

    def __init__(self):
        self.model_: M

    @property
    def is_fitted(self) -> bool:
        """
        Checks whether the estimator is already fitted.

        Returns:
            A boolean whether the estimator has been fitted.
        """
        try:
            getattr(self, "model_")
            return True
        except NotFittedError:
            return False

    def load_model(self, model: M) -> None:
        """
        Loads the provided model that has been fitted previously by this estimator or manually
        without the use of the estimator.

        Args:
            model: The model to load. In case, this estimator is already fitted, this model
                overrides the existing fitted model.
        """
        self.model_ = model

    def save(self, path: Path) -> None:
        """
        Saves the estimator to the provided directory. It saves a file named
        :code:`estimator.pickle` for the configuration of the estimator and additional files for
        the fitted model (if applicable). For more information on the files saved for the fitted
        model or for more customization, look at :meth:`get_params` and
        :meth:`pycave.core.ConfigModule.save`.

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

        if self.is_fitted:
            self.model_.save(path)

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

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        if key.endswith("_") and not key.endswith("__"):
            raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet")
        raise AttributeError
