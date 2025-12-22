"""High-level orchestration and pipeline management."""

from __future__ import annotations

from typing import List, Sequence

import xarray as xr

from echemistpy.processing.analyzers import TechniqueRegistry
from echemistpy.io.structures import AnalysisResult, Measurement


class AnalysisPipeline:
    """Coordinate loading, analysis and aggregation."""

    def __init__(self, registry: TechniqueRegistry) -> None:
        """Initialize pipeline with analyzer registry.

        Args:
            registry: TechniqueRegistry containing available analyzers
        """
        self.registry = registry

    def run(self, measurements: Sequence[Measurement], *, technique: str | None = None) -> List[AnalysisResult]:
        """Run analysis pipeline on measurements.

        Args:
            measurements: Sequence of Measurement objects to analyze
            technique: Optional override for technique identifier

        Returns:
            List of AnalysisResult objects
        """
        results: List[AnalysisResult] = []
        for measurement in measurements:
            # Get technique name from parameter or measurement metadata
            technique_name = (technique or measurement.metadata.others.get("technique", "unknown")).lower()
            analyzer = self.registry.get(technique_name)
            results.append(analyzer.analyze(measurement))
        return results

    def summary_table(self, results: Sequence[AnalysisResult]) -> xr.Dataset:
        """Condense a list of AnalysisResult objects into a dataset.

        Args:
            results: Sequence of AnalysisResult objects

        Returns:
            xarray.Dataset containing summarized results

        Examples:
            >>> from echemistpy.analyzers import TechniqueRegistry
            >>> from echemistpy.io.structures import AnalysisResult
            >>> registry = TechniqueRegistry()
            >>> pipeline = AnalysisPipeline(registry)
            >>> results = [
            ...     AnalysisResult(data=xr.Dataset()),
            ...     AnalysisResult(data=xr.Dataset()),
            ... ]
            >>> table = pipeline.summary_table(results)
        """
        if not results:
            return xr.Dataset()

        # Use empty dataset - to be implemented based on AnalysisResult structure
        return xr.Dataset()
