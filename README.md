# echemistpy

电化学与材料表征的统一数据处理框架。

## 快速开始

使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```powershell
uv sync --all-groups
```

## 开发规范

请参考 [AGENTS.md](AGENTS.md) 了解代码风格、Lint 和测试要求。

## 核心架构

```
Raw Files -> IOPluginManager -> RawData
                                  |
                           TechniqueAnalyzer
                                  |
                             AnalysisData
```

- **I/O**: 自动识别格式并标准化。
- **Processing**: 模块化分析器支持。
