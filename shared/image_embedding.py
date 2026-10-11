"""
使用 CLIP (clip-ViT-L-14) 生成图片向量。

直接对比像素级视觉特征，不经过文本中间层，
用于弥补 LLM 文字描述不一致导致的检索偏差。
"""

from PIL import Image
from sentence_transformers import SentenceTransformer

_model = None

CLIP_MODEL_NAME = "clip-ViT-L-14"
CLIP_DIM = 768


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(CLIP_MODEL_NAME)
    return _model


def get_image_embedding(image_path: str) -> list[float]:
    """将一张图片编码为 CLIP 向量 (768维)。"""
    model = _get_model()
    img = Image.open(image_path).convert("RGB")
    vec = model.encode(img)
    return vec.tolist()


def get_image_embeddings_batch(image_paths: list[str]) -> list[list[float]]:
    """批量编码图片为 CLIP 向量。"""
    model = _get_model()
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    vecs = model.encode(imgs, batch_size=32, show_progress_bar=True)
    return [v.tolist() for v in vecs]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/image_0002.jpg"
    vec = get_image_embedding(path)
    print(f"CLIP 向量维度: {len(vec)}")
    print(f"前5维: {vec[:5]}")
