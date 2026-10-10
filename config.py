import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "packaging_materials")

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
IMAGE_INPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "image_input")

EMBEDDING_DIM = 1024  # text-embedding-v4 输出维度
CLIP_DIM = 768        # clip-ViT-L-14 输出维度

TOP_K = 5

# 四层权重配置（加入 CLIP 图片向量后）
TIER_WEIGHTS = {
    "clip":  0.45,  # CLIP 图片直接视觉相似度（主力）
    "tier1": 0.15,  # 产品类别
    "tier2": 0.25,  # 整体瓶型 + 封口方式 + 泵头 + 瓶盖
    "tier3": 0.15,  # 瓶颈 + 瓶肩 + 瓶身 + 瓶底
}

TIER1_FIELDS = ["产品类别"]
TIER2_FIELDS = ["整体瓶型", "封口方式", "泵头", "瓶盖"]
TIER3_FIELDS = ["瓶颈", "瓶肩", "瓶身", "瓶底"]
