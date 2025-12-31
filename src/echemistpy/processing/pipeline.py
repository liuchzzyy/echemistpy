"""High-level orchestration for data analysis workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from echemistpy.io import load
from echemistpy.io.structures import ResultsData, ResultsDataInfo
from .analyzers.registry import create_default_registry, TechniqueRegistry


class AnalysisPipeline:
    """Orchestrates loading, analysis, and result management.

    This class provides a unified interface for processing experimental data
    files from raw format to analyzed results.
    """

    def __init__(self, registry: Optional[TechniqueRegistry] = None) -> None:
        """Initialize the pipeline with an analyzer registry.

        Args:
            registry: Optional custom registry. If None, uses the default registry.
        """
        self.registry = registry or create_default_registry()

    def run(
        self,
        path: str | Path,
        technique: Optional[str] = None,
        instrument: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[ResultsData, ResultsDataInfo]:
        """Run the full pipeline for a single file.

        Args:
            path: Path to the data file
            technique: Technique identifier (if None, will try to detect from file)
            instrument: Instrument identifier (passed to loader)
            **kwargs: Additional arguments passed to both loader and analyzer

        Returns:
            Tuple of (ResultsData, ResultsDataInfo)

        Raises:
            ValueError: If technique cannot be determined or analyzer is missing
        """
        # 1. Load data
        raw_data, raw_info = load(path, technique=technique, instrument=instrument, **kwargs)

        # 2. Determine technique
        if technique is None:
            # Try to get technique from raw_info
            techniques = raw_info.technique
            if techniques and techniques[0] != "Unknown":
                technique = techniques[0]
            else:
                raise ValueError(f"Could not detect technique for {path}. Please specify 'technique' explicitly.")

        # 3. Get analyzer
        try:
            analyzer = self.registry.get_analyzer(technique, raw_info.instrument)
        except KeyError as exc:
            raise ValueError(f"No analyzer registered for technique '{technique}' and instrument '{raw_info.instrument}'.") from exc

        # 4. Analyze
        return analyzer.analyze(raw_data, raw_info, **kwargs)


def run_analysis(
    path: str | Path,
    technique: Optional[str] = None,
    instrument: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[ResultsData, ResultsDataInfo]:
    """Convenience function to run analysis on a file using default settings.

    Args:
        path: Path to the data file
        technique: Technique identifier
        instrument: Instrument identifier
        **kwargs: Additional arguments

    Returns:
        Tuple of (ResultsData, ResultsDataInfo)
    """
    pipeline = AnalysisPipeline()
    return pipeline.run(path, technique=technique, instrument=instrument, **kwargs)
