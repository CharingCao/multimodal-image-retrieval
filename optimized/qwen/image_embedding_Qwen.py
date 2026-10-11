import base64
import mimetypes
import os
import sys
import requests
import numpy as np
from optimized.qwen.config_full_qwen import DASHSCOPE_API_KEY

MODEL_NAME = "qwen3-vl-embedding"

API_URL = (
    "https://dashscope.aliyuncs.com/api/v1/"
    "services/embeddings/multimodal-embedding/"
    "multimodal-embedding"
)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(
            f.read()
        ).decode("utf-8")

def get_mime_type(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(
        image_path
    )
    return mime or "image/jpeg"

def get_image_embedding(
    image_path: str
) -> list[float]:

    b64 = encode_image_to_base64( image_path )
    mime = get_mime_type( image_path )
    image_url = ( f"data:{mime};base64,{b64}" )
    headers = {
        "Authorization": (
            f"Bearer {DASHSCOPE_API_KEY}"
        ),
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "input": {
            "contents": [
                {
                    "image": image_url
                }
            ]
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    data = response.json()
    embedding = (
        data["output"]
        ["embeddings"][0]
        ["embedding"]
    )

    vec = np.asarray( embedding, dtype=np.float32 )
    norm = np.linalg.norm(vec)

    if norm > 0:
        vec = vec / norm

    return vec.tolist()

if __name__ == "__main__":
    image_path = sys.argv[1]
    vec = get_image_embedding( image_path )

    print( f"Qwen API 向量维度: {len(vec)}" )
    print( f"前5维: {vec[:5]}" )
