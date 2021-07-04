class NotFittedError(Exception):
    """
    Exception which is raised whenever properties of an estimator are accessed before the estimator
    has been fitted.
    """
