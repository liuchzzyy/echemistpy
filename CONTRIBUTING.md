# Contributing to echemistpy

感谢关注 echemistpy！本仓库为专有项目，贡献前请与维护者确认访问权限和需求。

## 开发流程
1. fork 仓库或在受控环境中创建分支。
2. 使用 `pyproject.toml` 中声明的依赖创建虚拟环境。
3. 在 `src/echemistpy/` 下实现功能，并在 `tests/` 中补充/更新 Pytest 覆盖。
4. 运行 `pytest` 与其他必要的质量检查（如 `ruff`）。
5. 更新 `CHANGELOG.md`、必要时同步文档与示例。
6. 通过 Pull Request 提交，描述修改内容并引用相关 issue。

## 风格约定
- 代码遵循 PEP 8，类型提示使用 Python 3.10+ 语法。
- 文档采用 Markdown，必要时在 `docs/` 中添加更长的说明或图片。
- 提交信息使用祈使句，例如 "Add TEM characterization loader"。
