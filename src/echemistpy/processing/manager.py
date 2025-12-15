"""High-level orchestration helpers."""

from __future__ import annotations

from typing import List, Sequence

import xarray as xr

from echemistpy.analysis import TechniqueRegistry
from echemistpy.io import AnalysisResult, Measurement


class AnalysisPipeline:
    """Coordinate loading, analysis and aggregation."""

    def __init__(self, registry: TechniqueRegistry) -> None:
        self.registry = registry

    def run(self, measurements: Sequence[Measurement], *, technique: str | None = None) -> List[AnalysisResult]:
        results: List[AnalysisResult] = []
        for measurement in measurements:
            technique_name = (technique or measurement.metadata.technique).lower()
            analyzer = self.registry.get(technique_name)
            results.append(analyzer.analyze(measurement))
        return results

    def summary_table(self, results: Sequence[AnalysisResult]) -> xr.Dataset:
        """Condense a list of :class:`AnalysisResult` objects into a dataset.

        Examples
        --------
        >>> from echemistpy.analysis import TechniqueRegistry
        >>> from echemistpy.io import AnalysisResult
        >>> registry = TechniqueRegistry()
        >>> pipeline = AnalysisPipeline(registry)
        >>> table = pipeline.summary_table([
        ...     AnalysisResult("xrd", "Sample-1", {"peak": 42.1}),
        ...     AnalysisResult("xps", "Sample-2", {"peak": 11.5}),
        ... ])
        >>> sorted(table.data_vars)
        ['peak', 'technique']
        >>> table.technique.values.tolist()
        ['xrd', 'xps']
        """
        if not results:
            return xr.Dataset()

        entries = [result.sample_name for result in results]
        coord_name = "entry"
        variables: dict[str, tuple[tuple[str], list[object]]] = {}

        def collect(field: str, values: List[object]) -> None:
            variables[field] = ((coord_name,), values)

        collect("technique", [result.technique for result in results])
        summary_keys = sorted({key for result in results for key in result.summary})
        for key in summary_keys:
            collect(key, [result.summary.get(key) for result in results])

        return xr.Dataset(data_vars=variables, coords={coord_name: entries})
