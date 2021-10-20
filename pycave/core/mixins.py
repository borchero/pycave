from abc import ABC, abstractmethod
from typing import Generic, TypeVar

D = TypeVar("D")
TM = TypeVar("TM", bound="TransformerMixin")  # type: ignore
PM = TypeVar("PM", bound="PredictorMixin")  # type: ignore
T = TypeVar("T")


class TransformerMixin(Generic[D, T], ABC):
    """
    Mixin that provides a ``fit_transform`` method that chains fitting the estimator and
    transforming the data it was fitted on.
    """

    @abstractmethod
    def fit(self: TM, data: D) -> TM:  # pylint: disable=missing-docstring
        pass

    @abstractmethod
    def transform(self, data: D) -> T:  # pylint: disable=missing-docstring
        pass

    def fit_transform(self, data: D) -> T:
        """
        Fits the estimator using the provided data and subsequently transforms the data using the
        fitted estimator. It simply chains calls to :meth:`fit` and :meth:`transform`.

        Args:
            data: The data to use for fitting and to transform. Consult the :meth:`fit` method for
                information on the type.

        Returns:
            The transformed data. Consult the :meth:`transform` documentation for more information
            on the return type.
        """
        return self.fit(data).transform(data)


class PredictorMixin(Generic[D, T], ABC):
    """
    Mixin that provides a ``fit_predict`` method that chains fitting the estimator and
    making predictions for the data it was fitted on.
    """

    @abstractmethod
    def fit(self: PM, data: D) -> PM:  # pylint: disable=missing-docstring
        pass

    @abstractmethod
    def predict(self, data: D) -> T:  # pylint: disable=missing-docstring
        pass

    def fit_predict(self, data: D) -> T:
        """
        Fits the estimator using the provided data and subsequently predicts the labels for the
        data using the fitted estimator. It simply chains calls to :meth:`fit` and
        :meth:`predict`.

        Args:
            data: The data to use for fitting and to predict labels for. Consult the :meth:`fit`
                method for information on the type.

        Returns:
            The predicted labels. Consult the :meth:`predict` documentation for more information
            on the return type.
        """
        return self.fit(data).predict(data)
