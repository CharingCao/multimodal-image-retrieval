"""
以图搜包材：用户上传图片 → CLIP图片向量 + Qwen-VL文本分析 → 四层加权检索 → 返回最相似包材。

架构: 文本硬过滤(产品类别+泵头+瓶颈) → CLIP视觉精排 + 文本辅助排序
"""

import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from analyze import analyze_packaging_image
from build_index import build_tier_text, extract_struct_flags, parse_description
from config import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
    TIER_WEIGHTS,
    TOP_K,
)
from embedding import get_embedding
from image_embedding import get_image_embedding


def search_similar_packaging(image_path: str, top_k: int = TOP_K) -> tuple[list[dict], str]:
    """
    输入一张包材图片路径，返回最相似的 top_k 个包材信息。
    四层加权检索：
      clip  (CLIP图片向量)     × 0.45  ← 主力，直接像素级视觉对比
      tier1 (产品类别)         × 0.15
      tier2 (瓶型+封口+泵头+盖) × 0.25
      tier3 (瓶颈+肩+身+底)    × 0.15
    """
    print(f"\n{'='*60}")
    print("[1/5] 正在生成 CLIP 图片向量...")
    print(f"{'='*60}")
    clip_vec = get_image_embedding(image_path)
    print(f"  CLIP 向量维度: {len(clip_vec)}")

    print(f"\n{'='*60}")
    print("[2/5] 正在用 Qwen-VL 分析瓶型结构...")
    print(f"{'='*60}")
    query_description = analyze_packaging_image(image_path)
    print(query_description)

    print(f"\n{'='*60}")
    print("[3/5] 正在解析并生成三层文本向量...")
    print(f"{'='*60}")
    fields = parse_description(query_description)
    t1_text = build_tier_text(fields, TIER1_FIELDS)
    t2_text = build_tier_text(fields, TIER2_FIELDS)
    t3_text = build_tier_text(fields, TIER3_FIELDS)

    print(f"  Tier1 (权重{TIER_WEIGHTS['tier1']:.0%}): {t1_text}")
    print(f"  Tier2 (权重{TIER_WEIGHTS['tier2']:.0%}): {t2_text}")
    print(f"  Tier3 (权重{TIER_WEIGHTS['tier3']:.0%}): {t3_text}")

    vec1 = get_embedding(t1_text)
    vec2 = get_embedding(t2_text)
    vec3 = get_embedding(t3_text)

    flags = extract_struct_flags(fields)
    struct_filter = Filter(
        must=[
            FieldCondition(key="has_pump", match=MatchValue(value=flags["has_pump"])),
            FieldCondition(key="has_neck", match=MatchValue(value=flags["has_neck"])),
        ]
    )

    pump_label = "有泵头" if flags["has_pump"] else "无泵头"
    neck_label = "有瓶颈" if flags["has_neck"] else "无瓶颈"

    print(f"\n{'='*60}")
    print(f"[4/5] 正在四层检索（硬过滤: {pump_label} + {neck_label}）...")
    print(f"{'='*60}")

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    candidate_pool = top_k * 5

    results_clip = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=clip_vec,
        using="clip",
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )
    results_t1 = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec1,
        using="tier1",
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )
    results_t2 = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec2,
        using="tier2",
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )
    results_t3 = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vec3,
        using="tier3",
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )

    print(f"  CLIP  召回 {len(results_clip.points)} 条")
    print(f"  Tier1 召回 {len(results_t1.points)} 条")
    print(f"  Tier2 召回 {len(results_t2.points)} 条")
    print(f"  Tier3 召回 {len(results_t3.points)} 条")

    print(f"\n{'='*60}")
    print("[5/5] 加权融合排序...")
    print(f"{'='*60}")

    scores = defaultdict(float)
    payloads = {}
    tier_scores = defaultdict(lambda: {"clip": 0.0, "tier1": 0.0, "tier2": 0.0, "tier3": 0.0})

    wc = TIER_WEIGHTS["clip"]
    w1, w2, w3 = TIER_WEIGHTS["tier1"], TIER_WEIGHTS["tier2"], TIER_WEIGHTS["tier3"]

    for p in results_clip.points:
        scores[p.id] += p.score * wc
        tier_scores[p.id]["clip"] = p.score
        payloads[p.id] = p.payload

    for p in results_t1.points:
        scores[p.id] += p.score * w1
        tier_scores[p.id]["tier1"] = p.score
        if p.id not in payloads:
            payloads[p.id] = p.payload

    for p in results_t2.points:
        scores[p.id] += p.score * w2
        tier_scores[p.id]["tier2"] = p.score
        if p.id not in payloads:
            payloads[p.id] = p.payload

    for p in results_t3.points:
        scores[p.id] += p.score * w3
        tier_scores[p.id]["tier3"] = p.score
        if p.id not in payloads:
            payloads[p.id] = p.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    matches = []
    for pid, final_score in ranked:
        pl = payloads[pid]
        ts = tier_scores[pid]
        matches.append({
            "score": final_score,
            "score_detail": ts,
            "filename": pl.get("filename", ""),
            "image_path": pl.get("image_path", ""),
            "description": pl.get("description", ""),
        })

    return matches, query_description


def print_results(query_image: str, query_desc: str, matches: list[dict]):
    print(f"\n{'='*60}")
    print("搜索结果")
    print(f"{'='*60}")
    print(f"查询图片: {query_image}\n")

    for i, m in enumerate(matches, 1):
        sd = m["score_detail"]
        print(f"--- 第 {i} 名 (加权总分: {m['score']:.4f}) ---")
        print(f"  分层得分: CLIP={sd['clip']:.4f} | 类别={sd['tier1']:.4f} | 外形={sd['tier2']:.4f} | 细节={sd['tier3']:.4f}")
        print(f"  文件: {m['filename']}")
        print(f"  描述:\n{m['description']}")
        print()


def main():
    if len(sys.argv) < 2:
        print("用法: python search.py <图片路径> [返回数量]")
        print("示例: python search.py data/image_input/20260330-165702.jpg 5")
        sys.exit(1)

    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K

    if not Path(image_path).exists():
        print(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)

    matches, query_desc = search_similar_packaging(image_path, top_k)
    print_results(image_path, query_desc, matches)


if __name__ == "__main__":
    main()
