"""Custom exceptions for echemistpy package."""


class EchemistpyError(Exception):
    """Base exception class for echemistpy."""


class DataLoadError(EchemistpyError):
    """Exception raised when data loading fails."""


class AnalysisError(EchemistpyError):
    """Exception raised when analysis fails."""


class PreprocessingError(EchemistpyError):
    """Exception raised when preprocessing fails."""


class ValidationError(EchemistpyError):
    """Exception raised when data validation fails."""
