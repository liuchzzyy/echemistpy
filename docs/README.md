# echemistpy 文档

本文档使用 Sphinx 构建。

## 构建文档

### 安装依赖

```bash
pip install -e ".[docs]"
```

### 构建文档

在 Linux/macOS 上：

```bash
cd docs
make html
```

在 Windows 上：

```bash
cd docs
sphinx-build -b html . _build/html
```

### 查看文档

构建完成后，在浏览器中打开 `_build/html/index.html`。

## 文档结构

```
docs/
├── api/                    # API 文档
│   ├── io.rst             # IO 模块文档
│   └── processing.rst     # 处理模块文档
├── user_guide/            # 用户指南
│   └── index.rst
├── developer_guide/       # 开发者指南
│   └── index.rst
├── conf.py                # Sphinx 配置
├── index.rst              # 文档首页
└── Makefile               # 构建脚本
```

## 更新文档

修改源代码后，重新构建文档：

```bash
make clean
make html
```
