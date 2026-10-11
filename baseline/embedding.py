"""
使用 DashScope text-embedding-v4 生成文本向量。
"""

from openai import OpenAI

from baseline.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM


client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def get_embedding(text: str) -> list[float]:
    """将文本转换为向量，返回 float 列表。"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIM,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """批量文本转向量（DashScope 单次最多 25 条）。"""
    all_embeddings = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
        )
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


if __name__ == "__main__":
    test_text = "圆柱形乳液瓶，按压泵头，弧形瓶肩，修长瓶身"
    vec = get_embedding(test_text)
    print(f"向量维度: {len(vec)}")
    print(f"前5维: {vec[:5]}")
