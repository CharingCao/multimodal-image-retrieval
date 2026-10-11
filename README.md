# 多模态包材图像检索

使用 Qdrant 存储向量，结合 CLIP 图像向量、DashScope 文本向量和 Qwen-VL 生成的结构化描述，检索相似包材图片。

## 项目结构

```text
├── analyze.py, embedding.py, image_embedding.py  # 基础描述和向量化
├── build_index.py, search.py, config.py           # 基础建库、检索与配置
├── extended/                                      # 扩展检索和模型实验
├── tools/                                         # 辅助脚本
└── data/                                          # 本地输入目录（不提交图片）
```

扩展脚本包括多模型检索、深度图检索、重排序和权重实验；它们可能需要额外服务或配置。

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

配置项及权重见 `config.py`。扩展版本的配置在 `extended/` 中。请从仓库根目录用模块方式运行扩展脚本，例如 `python -m extended.search_full`；辅助脚本同理，例如 `python -m tools.sync_cache_from_qdrant`。
深度图实验脚本需要自行设置 `DEPTH_DOWNLOAD_URL` 和 `DEPTH_BUCKET`，连接你有权使用的深度图服务。

## 数据与密钥

公开仓库不提供图片、缓存或 API Key。请自行确认输入数据和模型使用条件；运行产生的缓存、模型文件和本地向量库已列入 `.gitignore`。

## 许可证

自有业务代码以 [MIT License](LICENSE) 发布。外部模型、服务和依赖遵循各自的许可与使用条款。
