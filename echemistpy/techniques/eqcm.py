"""Electrochemical Quartz Crystal Microbalance (EQCM) characterization module."""

from pathlib import Path
from typing import Any

from echemistpy.core.base import BaseCharacterization, BaseData
from echemistpy.io.loaders import load_csv, load_excel


class EQCM(BaseCharacterization):
    """Electrochemical Quartz Crystal Microbalance characterization class.

    This class handles EQCM data for mass change measurements during
    electrochemical processes.

    Examples
    --------
    >>> eqcm = EQCM()
    >>> data = eqcm.load_data('eqcm_data.csv')
    >>> results = eqcm.analyze()
    """

    def __init__(self):
        """Initialize EQCM."""
        super().__init__("EQCM")

    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        """Load EQCM data from file.

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
        filepath = Path(filepath)
        if filepath.suffix == ".csv":
            data = load_csv(filepath, **kwargs)
        else:
            data = load_excel(filepath, **kwargs)

        self.data = BaseData(data, metadata={"source": str(filepath)})
        return self.data

    def preprocess(self, **kwargs: Any) -> BaseData:
        """Preprocess EQCM data.

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
        """Analyze EQCM data.

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
            "frequency_change": None,
            "mass_change": None,
        }
