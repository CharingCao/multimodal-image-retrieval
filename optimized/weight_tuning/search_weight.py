"""
以图搜包材：用户上传图片 → CLIP图片向量 + Qwen-VL文本分析 → 四层加权检索 → 返回最相似包材。

架构: 文本硬过滤(产品类别+泵头+瓶颈) → CLIP视觉精排 + 文本辅助排序
"""

TIER_CACHE = {}
import sys
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from optimized.analyze_full import analyze_full
from optimized.build_index_full import build_tier_text, extract_struct_flags, parse_description
from optimized.config_full import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
    TIER_WEIGHTS,
    TOP_K,
)
from shared.embedding import get_embedding
from shared.image_embedding import get_image_embedding

def calculate_tier1_score(
        query_fields,
        target_description
):
    """
    Tier1规则评分：产品类别 + 封口方式.返回:0~1
    """

    target_fields = parse_description( target_description )
    score = 0.0

    # ======================
    # 产品类别 85%
    # ======================

    q_cat = query_fields.get(
        "产品类别",
        ""
    )

    t_cat = target_fields.get(
        "产品类别",
        ""
    )

    if q_cat == t_cat:
        score += 0.85

    # ======================
    # 封口方式 15%
    # ======================

    q_close = query_fields.get(
        "封口方式",
        ""
    )

    t_close = target_fields.get(
        "封口方式",
        ""
    )

    if q_close == t_close:
        score += 0.15
    return score

def search_similar_packaging(image_path: str,weight = None, top_k: int = TOP_K) -> tuple[list[dict], str]:
    """
    输入一张包材图片路径，返回最相似的 top_k 个包材信息。
    四层加权检索：
      clip  (CLIP图片向量)     × 0.55  ← 主力，直接像素级视觉对比
      tier1 (产品类别)         × 0.05
      tier2 (瓶型+封口+泵头+盖) × 0.35
      tier3 (瓶颈+肩+身+底)    × 0.05
    """
    print(f"\n{'='*60}")
    print("[1/5] 正在生成 CLIP 图片向量...")
    print(f"{'='*60}")
    clip_vec = get_image_embedding(image_path)
    print(f"  CLIP 向量维度: {len(clip_vec)}")

    print(f"\n{'='*60}")
    print("[2/5] 正在用 Qwen-VL 分析瓶型结构...")
    print(f"{'='*60}")
    query_description = analyze_full(image_path)
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
    struct_filter = None

    pump_label = "有泵头" if flags["has_pump"] else "无泵头"
    neck_label = "有瓶颈" if flags["has_neck"] else "无瓶颈"

    print(f"\n{'='*60}")
    print(f"[4/5] 正在四层检索（软约束: {pump_label} + {neck_label}）...")
    print(f"{'='*60}")

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    candidate_pool = max(top_k*20,100)

    results_clip = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=clip_vec,
        using="clip",
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )

    print(f"  CLIP候选池 {len(results_clip.points)} 条")

    print(f"\n{'='*60}")
    print("[5/5] 加权融合排序...")
    print(f"{'='*60}")

    scores = defaultdict(float)
    payloads = {}
    tier_scores = defaultdict(lambda: {"clip": 0.0, "tier1": 0.0, "tier2": 0.0, "tier3": 0.0})

    if weight is None:
        weight = TIER_WEIGHTS

    wc = TIER_WEIGHTS["clip"]
    w1, w2, w3 = TIER_WEIGHTS["tier1"], TIER_WEIGHTS["tier2"], TIER_WEIGHTS["tier3"]

    for p in results_clip.points:
        scores[p.id] += p.score * wc
        tier_scores[p.id]["clip"] = p.score
        payloads[p.id] = p.payload
    # =========================
    # 对CLIP候选重新计算Tier2/Tier3
    # =========================

    for pid in payloads:

        desc = payloads[pid].get(
            "description",
            ""
        )

        target_fields = parse_description(desc)

        target_t2_text = build_tier_text(
            target_fields,
            TIER2_FIELDS
        )

        target_t3_text = build_tier_text(
            target_fields,
            TIER3_FIELDS
        )

        cache_key = pid

        if cache_key in TIER_CACHE:

            target_vec2,target_vec3 = TIER_CACHE[cache_key]

        else:

            target_vec2 = get_embedding(
                target_t2_text
            )

            target_vec3 = get_embedding(
                target_t3_text
            )

            TIER_CACHE[cache_key]=(
                target_vec2,
                target_vec3
            )

        # cosine similarity
        import numpy as np

        tier2_score = np.dot(
            vec2,
            target_vec2
        ) / (
            np.linalg.norm(vec2)
            *
            np.linalg.norm(target_vec2)
        )


        tier3_score = np.dot(
            vec3,
            target_vec3
        ) / (
            np.linalg.norm(vec3)
            *
            np.linalg.norm(target_vec3)
        )

        tier_scores[pid]["tier2"] = tier2_score
        tier_scores[pid]["tier3"] = tier3_score

        scores[pid] += tier2_score * w2
        scores[pid] += tier3_score * w3

    for pid in payloads:
        desc = payloads[pid].get(
            "description",
            ""
        )

        tier_scores[pid]["tier1"] = calculate_tier1_score(
            fields,
            desc
        )
        scores[pid] += (
            tier_scores[pid]["tier1"] * w1 )

    for pid in scores:
        target = payloads[pid]

        if target.get("has_pump") != flags["has_pump"]:
            scores[pid] -= 0.15
        if target.get("has_neck") != flags["has_neck"]:
            scores[pid] -= 0.05

    ranked = sorted(
        scores.items(),
        key=lambda x:x[1],
        reverse=True
    )[:top_k]

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
        print("用法: python -m optimized.weight_tuning.search_weight <图片路径> [返回数量]")
        print("示例: python -m optimized.weight_tuning.search_weight data/image_input/query.jpg 5")
        sys.exit(1)
        
    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K

    if not Path(image_path).exists():
        print(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)

    matches, query_desc = search_similar_packaging(image_path, None, top_k)
    print_results(image_path, query_desc, matches)

if __name__ == "__main__":
    main()
