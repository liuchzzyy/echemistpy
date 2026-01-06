"""科学数据的统一文件加载接口。

本模块提供简化的数据文件加载接口。自动检测文件格式并委托给
插件目录中相应的读取器进行处理。

主要功能：
- 自动格式检测：根据文件扩展名自动选择合适的读取器
- 多仪器支持：同一扩展名可支持多个仪器的读取器
- 元数据覆盖：支持手动覆盖样本名称、操作员等元数据
- 目录加载：支持加载整个目录的数据文件

使用示例：
    >>> from echemistpy.io import load
    >>> # 自动检测格式
    >>> raw_data, raw_info = load("data.mpt", sample_name="MySample")
    >>> # 指定仪器（对于有多个读取器的格式）
    >>> raw_data, raw_info = load("data.xlsx", instrument="lanhe")
    >>> # 加载目录
    >>> raw_data, raw_info = load("./data_dir", instrument="biologic")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from echemistpy.io.plugin_manager import get_plugin_manager
from echemistpy.io.standardizer import (
    standardize_names,
)
from echemistpy.io.structures import (
    RawData,
    RawDataInfo,
)

# Import default plugins
try:
    from echemistpy.io.plugins.Echem_BiologicMPTReader import BiologicMPTReader
except ImportError:
    BiologicMPTReader = None  # type: ignore

try:
    from echemistpy.io.plugins.Echem_LanheXLSXReader import LanheXLSXReader
except ImportError:
    LanheXLSXReader = None  # type: ignore

try:
    from echemistpy.io.plugins.XRD_MSPD import MSPDReader
except ImportError:
    MSPDReader = None  # type: ignore

try:
    from echemistpy.io.plugins.XAS_CLAESS import CLAESSReader
except ImportError:
    CLAESSReader = None  # type: ignore

try:
    from echemistpy.io.plugins.TXM_MISTRAL import MISTRALReader
except ImportError:
    MISTRALReader = None  # type: ignore

if TYPE_CHECKING:
    pass


def load(
    path: str | Path,
    fmt: Optional[str] = None,
    technique: Optional[str | list[str]] = None,
    instrument: Optional[str] = None,
    standardize: bool = True,
    **kwargs: Any,
) -> tuple[RawData, RawDataInfo]:
    """加载数据文件并返回标准化的 RawData 和 RawDataInfo。

    Args:
        path: 数据文件路径
        fmt: 可选的格式覆盖（如 '.mpt'）
        technique: 可选的技术类型提示（字符串或字符串列表）
        instrument: 可选的仪器名称覆盖
        standardize: 是否自动标准化数据（默认为 True）
        **kwargs: 传递给特定读取器的额外参数：
            - sample_name: 可选的样本名称覆盖
            - start_time: 可选的开始时间覆盖
            - operator: 可选的操作员名称覆盖
            - active_material_mass: 可选的活性物质质量覆盖
            - wave_number: 可选的波数覆盖

    Returns:
        (RawData, RawDataInfo) 元组

    Raises:
        ValueError: 如果目录加载未指定 instrument 或 fmt
        ValueError: 如果找不到指定扩展名的加载器
        ValueError: 如果多个加载器可用但未指定 instrument
        ValueError: 如果指定的 instrument 没有匹配的加载器
        FileNotFoundError: 如果文件不存在
        RuntimeError: 如果读取器类未实现 load() 方法
    """
    # Extract metadata overrides from kwargs
    overrides: dict[str, Any] = {
        "sample_name": kwargs.get("sample_name"),
        "start_time": kwargs.get("start_time"),
        "operator": kwargs.get("operator"),
        "active_material_mass": kwargs.get("active_material_mass"),
        "wave_number": kwargs.get("wave_number"),
        "instrument": instrument,
    }
    if technique:
        overrides["technique"] = [technique] if isinstance(technique, str) else technique

    path = Path(path) if isinstance(path, str) else path
    ext = fmt if fmt else path.suffix.lower()

    pm = get_plugin_manager()

    # For directory loading, instrument or fmt must be specified
    if path.is_dir() and not instrument and not fmt:
        raise ValueError(f"Loading a directory requires specifying 'instrument' or 'fmt'. Path: {path}")

    # If it's a directory and no extension is provided, we need to find a loader by instrument
    if path.is_dir() and not ext and instrument:
        # 精确匹配 instrument 名称，避免子串匹配错误
        instrument_normalized = instrument.lower().strip()
        for supported_ext in pm.list_supported_extensions():
            available_instruments = [inst.lower().strip() for inst in pm.get_loader_instruments(supported_ext)]
            if instrument_normalized in available_instruments:
                ext = supported_ext
                break

    # Check available instruments for this extension
    available_instruments = pm.get_loader_instruments(ext)

    if not available_instruments:
        if path.is_dir():
            raise ValueError(f"Could not determine loader for directory: {path}. Please specify 'instrument' or 'fmt'.")
        raise ValueError(f"No loader registered for extension: {ext}")

    # If multiple loaders exist but no instrument is specified, prompt the user
    if instrument is None and len(available_instruments) > 1:
        raise ValueError(f"Multiple loaders available for extension '{ext}'. Please specify 'instrument' to choose one.\nAvailable instruments: {available_instruments}")

    reader_class = pm.get_loader(ext, instrument=instrument)

    if reader_class is None:
        # This happens if instrument was provided but didn't match any registered loader
        raise ValueError(f"No loader found for extension '{ext}' with instrument '{instrument}'.\nAvailable instruments for this format: {available_instruments}")

    # Instantiate reader and load raw data
    # Pass standard metadata to reader if provided
    for k, v in overrides.items():
        if v is not None:
            kwargs[k] = v

    reader = reader_class(filepath=path, **kwargs)
    if not hasattr(reader, "load"):
        raise RuntimeError(f"Reader class {reader_class.__name__} does not implement 'load' method yet.")

    # Pass kwargs to load() to support reader-specific parameters like 'edges'
    raw_data, raw_info = reader.load(**kwargs)

    # Filter out None values and apply manual overrides
    raw_info.update({k: v for k, v in overrides.items() if v is not None})

    if not standardize:
        return raw_data, raw_info

    # Auto-standardize
    standardized_data, standardized_info = standardize_names(raw_data, raw_info, technique_hint=technique)

    return standardized_data, standardized_info


def _register_loader(extensions: list[str], loader_class: Any) -> None:
    """Register a new loader class for specific extensions.

    Args:
        extensions: List of file extensions (e.g., ['.mpt', '.xlsx'])
        loader_class: The class to handle these files
    """
    pm = get_plugin_manager()
    pm.register_loader(extensions, loader_class)


# ============================================================================
# Utility Functions
# ============================================================================


def list_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions."""
    pm = get_plugin_manager()
    loaders = pm.get_supported_loaders()

    formats = {}
    for ext, plugin_names in loaders.items():
        # plugin_names is a list of strings
        is_biologic = any("biologic" in name.lower() for name in plugin_names)
        is_lanhe = any("lanhe" in name.lower() for name in plugin_names)
        is_mspd = any("mspd" in name.lower() for name in plugin_names)
        is_claess = any("claess" in name.lower() for name in plugin_names)
        is_mistral = any("mistral" in name.lower() for name in plugin_names)

        if is_biologic:
            formats[ext] = "BioLogic EC-Lab files (.mpt)"
        elif is_lanhe:
            formats[ext] = "LANHE battery test files (.xlsx)"
        elif is_mspd:
            formats[ext] = "MSPD XRD files (.xye)"
        elif is_claess:
            formats[ext] = "ALBA CLAESS XAS files (.dat)"
        elif is_mistral:
            formats[ext] = "MISTRAL TXM files (.hdf5)"
        else:
            formats[ext] = f"Loaded by {', '.join(plugin_names)}"

    return formats


# ============================================================================
# Plugin System Initialization
# ============================================================================


def _initialize_default_plugins() -> None:
    """Initialize and register default loader and saver plugins."""
    pm = get_plugin_manager()
    if pm.initialized:
        return

    # Register echem-specific plugins
    if BiologicMPTReader is not None:
        pm.register_loader([".mpt"], BiologicMPTReader)

    if LanheXLSXReader is not None:
        pm.register_loader([".xlsx"], LanheXLSXReader)

    if MSPDReader is not None:
        pm.register_loader([".xye"], MSPDReader)

    if CLAESSReader is not None:
        pm.register_loader([".dat"], CLAESSReader)

    if MISTRALReader is not None:
        pm.register_loader([".hdf5"], MISTRALReader)

    pm.initialized = True


# Initialize plugins on module import
_initialize_default_plugins()


__all__ = [
    "list_supported_formats",
    "load",
]
