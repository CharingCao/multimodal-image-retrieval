import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Qwen-VL结构分析与Tier文本向量
# 结构分析模型保持不变
# ============================================================

DASHSCOPE_API_KEY = os.getenv( "DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv( "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1" )
QWEN_VL_MODEL = os.getenv( "QWEN_VL_MODEL", "qwen-vl-plus" )

EMBEDDING_MODEL = os.getenv( "EMBEDDING_MODEL", "text-embedding-v4" )

# ============================================================
# Qdrant
# ============================================================

QDRANT_HOST = os.getenv("QDRANT_HOST_GEMINI", os.getenv("QDRANT_HOST_QWEN", "localhost"))

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT_GEMINI",
        os.getenv(
            "QDRANT_PORT_QWEN",
            "6333",
        ),
    )
)

COLLECTION_NAME = os.getenv( "QDRANT_COLLECTION_GEMINI", "packaging_full_v3_gemini" )

# ============================================================
# 图片目录
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
IMAGES_DIR = os.path.join( BASE_DIR, "data", "images" )
IMAGE_INPUT_DIR = os.path.join( BASE_DIR, "data", "image_input" )

# ============================================================
# Gemini视觉向量
# ============================================================

GEMINI_VECTOR_NAME = ( "gemini-embedding-2-original" )
GEMINI_VECTOR_DIM = 3072

# ============================================================
# Tier文本向量
# ============================================================

EMBEDDING_DIM = 1024
TOP_K = 5

# ============================================================
# 四层融合权重
# ============================================================

TIER_WEIGHTS = { "gemini": 0.55, "tier1": 0.05, "tier2": 0.15, "tier3": 0.25,}
TIER1_FIELDS = [ "产品类别" ]
TIER2_FIELDS = [ "整体瓶型", "封口方式", "泵头", "瓶盖",]
TIER3_FIELDS = [ "瓶颈", "瓶肩", "瓶身", "瓶底",]
