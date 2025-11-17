"""Transmission Electron Microscopy (TEM) characterization module."""

from pathlib import Path
from typing import Any

from echemistpy.core.base import BaseCharacterization, BaseData


class TEM(BaseCharacterization):
    """Transmission Electron Microscopy characterization class.

    This class handles TEM, TEM-EDS, and TEM-EELS data.

    Examples
    --------
    >>> tem = TEM()
    >>> data = tem.load_data('tem_image.dm3')
    >>> results = tem.analyze()
    """

    def __init__(self):
        """Initialize TEM."""
        super().__init__("TEM")

    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        """Load TEM data from file.

        Parameters
        ----------
        filepath : Path or str
            Path to data file
        **kwargs : Any
            Additional arguments for loading

        Returns
        -------
        BaseData
            Loaded data
        """
        # TEM data often uses HyperSpy for .dm3, .dm4 files
        import hyperspy.api as hs

        data = hs.load(str(filepath))
        self.data = BaseData(data, metadata={"source": str(filepath)})
        return self.data

    def preprocess(self, **kwargs: Any) -> BaseData:
        """Preprocess TEM data.

        Parameters
        ----------
        **kwargs : Any
            Preprocessing parameters

        Returns
        -------
        BaseData
            Preprocessed data
        """
        if self.data is None:
            msg = "No data loaded. Call load_data() first."
            raise ValueError(msg)

        return self.data

    def analyze(self, **kwargs: Any) -> dict[str, Any]:
        """Analyze TEM data.

        Parameters
        ----------
        **kwargs : Any
            Analysis parameters

        Returns
        -------
        dict
            Analysis results
        """
        if self.data is None:
            msg = "No data loaded. Call load_data() first."
            raise ValueError(msg)

        return {
            "technique": self.technique_name,
            "particle_size": [],
            "elemental_maps": {},
        }
