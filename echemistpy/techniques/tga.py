"""Thermogravimetric Analysis (TGA) characterization module."""

from pathlib import Path
from typing import Any

from echemistpy.core.base import BaseCharacterization, BaseData
from echemistpy.io.loaders import load_csv, load_excel


class TGA(BaseCharacterization):
    """Thermogravimetric Analysis characterization class.

    This class handles TGA data for thermal decomposition analysis.

    Examples
    --------
    >>> tga = TGA()
    >>> data = tga.load_data('tga_data.csv')
    >>> results = tga.analyze()
    """

    def __init__(self):
        """Initialize TGA."""
        super().__init__("TGA")

    def load_data(self, filepath: Path | str, **kwargs: Any) -> BaseData:
        """Load TGA data from file.

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
        """Preprocess TGA data.

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
        """Analyze TGA data.

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
            "decomposition_steps": [],
            "residual_mass": None,
        }
