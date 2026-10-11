# 多模态包材图像检索

这是一个面向包材图像的垂类检索系统。系统先分析产品类别和瓶体结构等特征，再结合视觉向量与分层文本向量，在 Qdrant 中检索相似图像。`optimized/` 是项目的主要成果：相较初始版本，优化了分析提示词、分类维度、瓶盖专项分析、建库流程及检索权重。

当前提示词和分类字段围绕包材设计。修改 `optimized/analyze_structure.py`、`optimized/analyze_cap.py` 中的提示词，以及 `optimized/config_full.py` 中的分类字段和权重后，可将这套流程适配到更广泛的图像检索领域；实际效果需要针对新领域重新建库和评估。

## 代码结构

```text
baseline/                  初始系统，供对照
optimized/                 优化后的 CLIP 主系统
  qwen/                    更换为 Qwen 视觉向量的实验
  gemini/                  更换为 Gemini 视觉向量的实验
  weight_tuning/           自动调权重测试
tools/                     辅助脚本
```

### 优化后的 CLIP 主系统

以下文件属于同一套系统，建议从这条主线了解项目：

| 文件 | 作用 |
| --- | --- |
| `optimized/analyze_structure.py` | 优化提示词和分类后的结构分析 |
| `optimized/analyze_cap.py` | 瓶盖专项分析 |
| `optimized/analyze_full.py` | 汇总结构与瓶盖分析 |
| `optimized/config_full.py` | 模型、集合、分类字段与检索权重配置 |
| `optimized/build_index_full.py` | 生成描述和向量，建立优化版索引 |
| `optimized/search_full.py` | 使用 CLIP 视觉向量与分层文本向量检索 |

`optimized/clip_embedding.py` 和 `optimized/text_embedding.py` 为这套主系统提供向量化功能。`baseline/` 中的配置及图像、文本向量模块只服务于初始系统，不应直接作为优化版的配置使用。

### 更换视觉模型的实验

`optimized/qwen/` 内的 `config_full_qwen.py`、`image_embedding_Qwen.py`、`search_full_qwen.py` 是一组；`optimized/gemini/` 内的 `config_full_gemini.py`、`image_embedding_gemini.py`、`search_full_gemini.py` 是另一组。它们尝试更换主系统的视觉向量模型，保留优化后的结构分析思路。各模型需要相应的服务、密钥和向量集合；这些脚本用于实验对比，不是默认 CLIP 流程。

`optimized/weight_tuning/search_weight.py` 与 `auto_weight_test_full.py` 用于测试自动调权重，也不属于默认检索流程。

## 环境与数据

需要 Python 3.10+、运行中的 Qdrant 服务、自己的 DashScope API Key。CLIP 模型首次使用时需要下载；外部模型服务可能产生费用。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

复制 `.env.example` 为 `.env` 并填写 `DASHSCOPE_API_KEY`。本机启动 Qdrant 的一个示例：

```bash
docker run --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

公开仓库不包含 `data/`，也不包含业务图片、描述缓存、Qdrant 数据、密钥、模型权重或原始仓库历史。运行前请在仓库根目录自行创建 `data/images/`，放入你有权使用的图片；查询图片可放在任意本地路径，例如自行创建的 `data/image_input/`。整个 `data/` 已被 Git 忽略。

## 运行主系统

在仓库根目录运行：

```bash
python -m optimized.build_index_full --recreate
python -m optimized.search_full data/image_input/query.jpg 5
```

首次建库使用 `--recreate`；后续增量入库可使用 `python -m optimized.build_index_full --skip-indexed`。初始版本可用 `python -m baseline.build_index` 和 `python -m baseline.search` 对照。模型、集合名称和权重见 `optimized/config_full.py`。

## 后续优化方向

引入深度图接口，以增加对形状和空间结构的描述，是下一步可以探索的优化方案。目前仓库没有提供完成的深度图检索实现，也没有将其计入上述主系统的效果。

## 许可证

自有代码以 [MIT License](LICENSE) 发布。外部模型、服务和依赖遵循各自的许可与使用条款。
