# 多模态包材图像检索

这是一个面向包材图片的多模态检索项目。**`optimized/` 是主要成果**：在初始 CLIP 检索系统上，改进了提示词与分类分析，增加瓶盖专项分析，并扩展了建库和检索流程。系统使用 Qdrant 存储向量，结合 CLIP 图像向量、DashScope 文本向量和 Qwen-VL 生成的结构化描述。

## 项目结构

```text
├── optimized/  # 优化后的分析、建库与检索；包含视觉模型对比实验
│   └── weight_tuning/  # 自动调权重测试脚本
├── baseline/   # 初始版本，供对照
├── shared/     # 两套流程共用的配置及向量化代码
├── tools/      # 缓存同步等辅助脚本
└── data/       # 本地输入目录（不提交图片）
```

### 优化版的关键文件

| 文件 | 用途 |
| --- | --- |
| `optimized/analyze_structure.py` | 优化后的结构与类别分析提示词 |
| `optimized/analyze_cap.py` | 瓶盖专项分析 |
| `optimized/analyze_full.py` | 合并结构分析和瓶盖分析结果 |
| `optimized/build_index_full.py` | 使用优化后的描述建立完整索引 |
| `optimized/search_full.py` | **CLIP 主检索脚本**：CLIP 视觉向量与分层文本向量加权检索 |
| `optimized/search_full_clip_depth.py` | 在 CLIP 检索方案上加入深度图接口的版本 |

`optimized/search_full_qwen.py` 和 `optimized/search_full_gemini.py` 是**替换视觉向量模型**的尝试，用来与 CLIP 方案比较；相关的 `config_full_qwen.py`、`config_full_gemini.py` 及图像向量模块也在 `optimized/`。所有名称含 `depth` 的检索脚本都在对应视觉方案上加入了**深度图接口**。`optimized/weight_tuning/search_weight.py` 与 `auto_weight_test_full.py` 用于测试自动调权重，不属于默认检索流程。这些实验脚本可能需要额外服务或配置。

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
2. 从仓库根目录运行 `python -m optimized.build_index_full --recreate` 创建优化版索引；之后可用 `python -m optimized.build_index_full --skip-indexed` 增量入库。
3. 运行 `python -m optimized.search_full 路径/查询图.jpg 5`，返回前 5 个结果。

主流程的模型、集合与检索权重见 `optimized/config_full.py`。初始版本留在 `baseline/`，可用 `python -m baseline.build_index` 和 `python -m baseline.search` 对照。辅助脚本从仓库根目录以模块方式运行，例如 `python -m tools.sync_cache_from_qdrant`。
深度图实验脚本需要自行设置 `DEPTH_DOWNLOAD_URL` 和 `DEPTH_BUCKET`，连接你有权使用的深度图服务。

## 数据与密钥

公开仓库不提供图片、缓存或 API Key。请自行确认输入数据和模型使用条件；运行产生的缓存、模型文件和本地向量库已列入 `.gitignore`。

## 许可证

自有业务代码以 [MIT License](LICENSE) 发布。外部模型、服务和依赖遵循各自的许可与使用条款。
