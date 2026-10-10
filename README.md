# 多模态包材图像检索

使用 Qdrant 存储向量，结合 CLIP 图像向量、DashScope 文本向量和 Qwen-VL 生成的结构化描述，检索相似包材图片。

## 包含内容

- `build_index.py`：扫描本地图片并建立索引。
- `search.py`：使用查询图片检索相似图片。
- `analyze.py`、`embedding.py`、`image_embedding.py`：描述生成与向量化。
- `*_full.py` 和其他脚本：扩展检索、权重实验及其他模型的探索版本；可能需要额外的服务或配置。

本公开版只包含业务代码。不包含原仓库历史、业务图片、描述缓存、Qdrant 数据、模型权重或内嵌的第三方 SAM3 源码。分割模块如需使用，请根据相应上游项目的许可自行安装。

## 环境准备

需要 Python 3.10+、运行中的 Qdrant 服务，以及你自己的 DashScope API Key。CLIP 模型首次使用时需要下载；相关服务可能产生费用。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

将 `.env.example` 复制为 `.env`，填写 `DASHSCOPE_API_KEY`。不要提交 `.env`。

本机使用 Docker 启动 Qdrant 的示例：

```bash
docker run --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## 基本用法

1. 把你有权使用的图片放入 `data/images/`。
2. 运行 `python build_index.py --recreate` 创建索引；之后可用 `python build_index.py --skip-indexed` 增量入库。
3. 运行 `python search.py 路径/查询图.jpg 5`，返回前 5 个结果。

配置项及权重见 `config.py`。扩展版本分别使用 `config_full.py`、`config_full_qwen.py` 和 `config_full_gemini.py`。

## 数据与密钥

公开仓库不提供图片、缓存或 API Key。请自行确认输入数据和模型使用条件；运行产生的缓存、模型文件和本地向量库已列入 `.gitignore`。

## 许可证

自有业务代码以 [MIT License](LICENSE) 发布。外部模型、服务和依赖遵循各自的许可与使用条款。

