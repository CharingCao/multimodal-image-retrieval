import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv( "GEMINI_EMBEDDING_MODEL", "google/gemini-embedding-2" )
BASE_URL = os.getenv( "GEMINI_EMBEDDING_BASE_URL", "https://ai-gateway.vercel.sh/v1" )
API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY")
API_URL = f"{BASE_URL.rstrip('/')}/embeddings"
MAX_RETRIES = 3

def encode_image_to_base64( image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError( f"图片不存在: {image_path}" )

    with path.open("rb") as f:
        return base64.b64encode( f.read() ).decode("utf-8")

def get_mime_type( image_path: str ) -> str:
    mime, _ = mimetypes.guess_type( image_path )

    if mime not in { "image/png", "image/jpeg", }:
        raise ValueError( "Gemini Embedding 2 当前仅支持 " f"PNG/JPEG，实际类型: {mime}" )

    return mime


def get_image_embedding( image_path: str) -> list[float]:

    if not API_KEY:
        raise RuntimeError(
            "未找到 GEMINI_EMBEDDING_API_KEY，"
            "请检查 .env"
        )

    b64 = encode_image_to_base64( image_path )
    mime = get_mime_type( image_path )

    image_url = ( f"data:{mime};base64,{b64}" )

    headers = { "Authorization": ( f"Bearer {API_KEY}" ), "Content-Type": "application/json", }

    payload = {
        "model": MODEL_NAME,
        "input": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        ],
        "dimensions": 3072,
    }

    last_error = None

    for attempt in range( 1, MAX_RETRIES + 1 ):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise requests.HTTPError(
                    "Gemini临时请求失败，"
                    f"状态码={response.status_code}，"
                    f"响应={response.text[:500]}",
                    response=response,
                )

            response.raise_for_status()
            data = response.json()
            embedding = ( data["data"][0]["embedding"] )

            vec = np.asarray(
                embedding,
                dtype=np.float32,
            )

            if vec.size != 3072:
                raise ValueError(
                    "Gemini向量维度异常："
                    f"期望3072，实际{vec.size}"
                )

            norm = np.linalg.norm(vec)

            if norm > 0:
                vec = vec / norm

            return vec.tolist()

        except (
            requests.RequestException,
            KeyError,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt >= MAX_RETRIES:
                break

            wait_seconds = 2 ** attempt

            print(
                "Gemini图片向量请求失败，"
                f"{wait_seconds}秒后进行"
                f"第{attempt + 1}次尝试：{exc}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "Gemini图片向量生成失败，"
        f"已重试{MAX_RETRIES}次："
        f"{last_error}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "用法: python "
            "image_embedding_gemini.py "
            '"data/image_input/01.png"'
        )
        raise SystemExit(1)

    image_path = sys.argv[1]
    vec = get_image_embedding( image_path )

    print(
        f"Gemini API 向量维度: "
        f"{len(vec)}"
    )
    print( f"前5维: {vec[:5]}" )