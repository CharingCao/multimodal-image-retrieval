"""
以图搜包材：用户上传图片 → Gemini图片向量 → Qwen-VL结构分析 → Tier1/2/3融合排序 → 返回最相似包材。

只将Qwen视觉召回替换为Gemini Embedding。
Qwen-VL结构分析、Tier1/2/3、软约束和评分规则保持不变。
用于比较Gemini视觉模型与Qwen视觉模型的实际效果。
"""

import sys
import re
import numpy as np
from collections import defaultdict
from pathlib import Path
from qdrant_client import QdrantClient
from optimized.analyze_full import analyze_full
from optimized.build_index_full import ( build_tier_text, extract_struct_flags, parse_description)
from optimized.gemini.config_full_gemini import (
    COLLECTION_NAME,
    GEMINI_VECTOR_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
    TIER_WEIGHTS,
    TOP_K,
)

from optimized.text_embedding import get_embedding

# ============================================================
# Gemini 图片 embedding
# ============================================================
from optimized.gemini.image_embedding_gemini import ( get_image_embedding as get_gemini_embedding )

# ============================================================
# Tier1规则评分
# ============================================================

def calculate_tier1_score(
        query_fields,
        target_description
):

    target_fields = parse_description( target_description )
    score = 0.0

    # ======================
    # 产品类别 85%
    # ======================

    q_cat = query_fields.get( "产品类别", "" )
    t_cat = target_fields.get( "产品类别", "" )
    if q_cat == t_cat:
        score += 0.85

    # ======================
    # 封口方式 15%
    # ======================

    q_close = query_fields.get( "封口方式", "" )
    t_close = target_fields.get( "封口方式", "" )
    if q_close == t_close:
        score += 0.15

    return score

# ============================================================
# 拆分结构属性
# ============================================================

def split_structure_values(
        value: str
) -> set[str]:

    if not value:
        return set()
    values = re.split( r"[，,；;、|]", value, )

    return {
        re.sub( r"\s+", "", item )
        for item in values
        if item.strip()
    }

# ============================================================
# 单字段结构相似度
# ============================================================

def calculate_value_similarity(
        query_value: str,
        target_value: str,
) -> float:

    query_items = split_structure_values( query_value )
    target_items = split_structure_values( target_value )

    if ( not query_items or not target_items ):
        return 0.0
    if query_items == target_items:
        return 1.0

    common_count = len( query_items & target_items )

    return ( 2.0 * common_count / ( len(query_items) + len(target_items) )
    )

# ============================================================
# Tier2规则分
# ============================================================

def calculate_tier2_rule_score(
        query_fields: dict[str, str],
        target_fields: dict[str, str],
) -> float:

    field_weights = {
        "整体瓶型": 0.35,
        "封口方式": 0.15,
        "泵头": 0.20,
        "瓶盖": 0.30,
    }
    score = 0.0

    for (
        field_name,
        field_weight
    ) in field_weights.items():

        field_score = (
            calculate_value_similarity(
                query_fields.get( field_name, "" ),
                target_fields.get( field_name, "" ),
            )
        )
        score += ( field_score * field_weight )
    return score

# ============================================================
# Tier3规则分
# ============================================================

def calculate_tier3_rule_score(
        query_fields: dict[str, str],
        target_fields: dict[str, str],
) -> float:

    field_weights = {
        "瓶颈": 0.20,
        "瓶肩": 0.20,
        "瓶身": 0.40,
        "瓶底": 0.20,
    }
    score = 0.0

    for ( field_name, field_weight ) in field_weights.items():

        field_score = (
            calculate_value_similarity(
                query_fields.get( field_name, "" ),
                target_fields.get( field_name, "" ),
            )
        )
        score += ( field_score * field_weight )
    return score


# ============================================================
# 搜索
# ============================================================

def search_similar_packaging(
        image_path: str,
        top_k: int = TOP_K,
) -> tuple[list[dict], str]:

    # ========================================================
    # 1/5 Gemini图片向量
    # ========================================================

    print( f"\n{'=' * 60}")
    print("[1/5] 正在生成Gemini图片向量...")
    print(f"{'=' * 60}")
    gemini_vec = get_gemini_embedding( image_path )
    print(
        f"  Gemini 向量维度: "
        f"{len(gemini_vec)}"
    )

    # ========================================================
    # 2/5 Qwen-VL结构分析
    # ========================================================

    print(f"\n{'=' * 60}")
    print("[2/5] 正在获取瓶型结构...")
    print(f"{'=' * 60}")
    query_description = analyze_full(image_path)
    print(query_description)

    # ========================================================
    # 3/5 Tier文本向量
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[3/5] 正在解析并生成三层文本向量..." )
    print( f"{'=' * 60}" )

    fields = parse_description( query_description )
    t1_text = build_tier_text( fields, TIER1_FIELDS )
    t2_text = build_tier_text( fields, TIER2_FIELDS )
    t3_text = build_tier_text( fields, TIER3_FIELDS )

    print(
        f"  Tier1 "
        f"(权重{TIER_WEIGHTS['tier1']:.0%}): "
        f"{t1_text}"
    )
    print(
        f"  Tier2 "
        f"(权重{TIER_WEIGHTS['tier2']:.0%}): "
        f"{t2_text}"
    )
    print(
        f"  Tier3 "
        f"(权重{TIER_WEIGHTS['tier3']:.0%}): "
        f"{t3_text}"
    )

    vec2 = get_embedding( t2_text )
    vec3 = get_embedding( t3_text )

    # ========================================================
    # 结构标记
    # ========================================================

    flags = extract_struct_flags( fields )
    struct_filter = None
    pump_label = (
        "有泵头"
        if flags["has_pump"]
        else "无泵头"
    )
    neck_label = (
        "有瓶颈"
        if flags["has_neck"]
        else "无瓶颈"
    )

    # ========================================================
    # 4/5 Gemini召回
    # ========================================================

    print( f"\n{'=' * 60}" )
    print(
        f"[4/5] 正在四层检索"
        f"（软约束: "
        f"{pump_label} + {neck_label}）..."
    )

    print( f"{'=' * 60}" )

    qdrant = QdrantClient( host=QDRANT_HOST, port=QDRANT_PORT, )
    candidate_pool = max( top_k, 100 )
    results_gemini = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=gemini_vec,
        using=GEMINI_VECTOR_NAME,
        query_filter=struct_filter,
        limit=candidate_pool,
        with_payload=True,
    )

    print(
        f"  Gemini候选池 "
        f"{len(results_gemini.points)} 条"
    )

    # ========================================================
    # 5/5 加权融合
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[5/5] 加权融合排序..." )
    print( f"{'=' * 60}" )
    scores = defaultdict( float )
    payloads = {}
    tier_scores = defaultdict(
        lambda: {
            "gemini": 0.0,
            "tier1": 0.0,
            "tier2": 0.0,
            "tier3": 0.0,
        }
    )

    visual_weight = ( TIER_WEIGHTS["gemini"] )
    w1 = TIER_WEIGHTS[ "tier1" ]
    w2 = TIER_WEIGHTS[ "tier2" ]
    w3 = TIER_WEIGHTS[ "tier3" ]

    # ========================================================
    # Gemini召回分
    # ========================================================

    for p in results_gemini.points:
        scores[p.id] += ( p.score * visual_weight )
        tier_scores[ p.id ]["gemini"] = p.score
        payloads[ p.id ] = p.payload

    # ========================================================
    # 对Gemini候选重新计算Tier2 / Tier3
    # ========================================================

    for pid in payloads:

        desc = payloads[
            pid
        ].get(
            "description",
            ""
        )
        target_fields = (
            parse_description(
                desc
            )
        )
        target_t2_text = (
            build_tier_text(
                target_fields,
                TIER2_FIELDS
            )
        )
        target_t3_text = (
            build_tier_text(
                target_fields,
                TIER3_FIELDS
            )
        )
        target_vec2 = (
            get_embedding(
                target_t2_text
            )
        )
        target_vec3 = (
            get_embedding(
                target_t3_text
            )
        )

        # ====================================================
        # cosine similarity
        # ====================================================

        tier2_vector_score = float( np.dot( vec2, target_vec2 ) / ( np.linalg.norm( vec2 ) * np.linalg.norm( target_vec2 ) + 1e-12 ) )
        tier3_vector_score = float( np.dot( vec3, target_vec3 ) / ( np.linalg.norm( vec3 ) * np.linalg.norm( target_vec3 ) + 1e-12 ) )

        # ====================================================
        # 规则分
        # ====================================================

        tier2_rule_score = ( calculate_tier2_rule_score( fields, target_fields, ) )
        tier3_rule_score = ( calculate_tier3_rule_score( fields, target_fields, ) )

        tier2_score = ( tier2_vector_score * 0.70 + tier2_rule_score * 0.30 )
        tier3_score = ( tier3_vector_score * 0.70 + tier3_rule_score * 0.30 )

        tier_scores[
            pid
        ]["tier2"] = (
            tier2_score
        )
        tier_scores[
            pid
        ]["tier3"] = (
            tier3_score
        )
        scores[ pid ] += ( tier2_score * w2 )
        scores[ pid ] += ( tier3_score * w3 )

    # ========================================================
    # Tier1
    # ========================================================

    for pid in payloads:

        desc = payloads[
            pid
        ].get(
            "description",
            ""
        )

        tier_scores[
            pid
        ]["tier1"] = (
            calculate_tier1_score(
                fields,
                desc
            )
        )

        scores[
            pid
        ] += (
            tier_scores[
                pid
            ]["tier1"]
            *
            w1
        )

    for pid in scores:
        target = payloads[ pid ]

        if ( target.get("has_pump") != flags[ "has_pump" ] ):
            scores[ pid ] -= 0.15

        if ( target.get( "has_neck") != flags[ "has_neck" ] ):
            scores[ pid ] -= 0.05

    # ========================================================
    # 最终排序
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # ========================================================
    # 返回结果
    # ========================================================

    matches = []
    for ( pid, final_score ) in ranked:
        pl = payloads[ pid ]
        ts = tier_scores[ pid ]

        matches.append(
            {
                "score": final_score,
                "score_detail": ts,
                "filename": pl.get(
                    "filename",
                    ""
                ),
                "image_path": pl.get(
                    "image_path",
                    ""
                ),
                "description": pl.get(
                    "description",
                    ""
                ),
            }
        )
    return ( matches, query_description )

# ============================================================
# 输出搜索结果
# ============================================================

def print_results(
        query_image: str,
        query_desc: str,
        matches: list[dict]
):

    print( f"\n{'=' * 60}")
    print( "搜索结果" )
    print( f"{'=' * 60}" )
    print(
        f"查询图片: "
        f"{query_image}\n"
    )

    for i, m in enumerate( matches, 1 ):
        sd = m[ "score_detail" ]
        print(
            f"--- 第 {i} 名 "
            f"(加权总分: "
            f"{m['score']:.4f}) ---"
        )
        print(
            f"  分层得分: "
            f"Gemini={sd['gemini']:.4f} "
            f"| 类别={sd['tier1']:.4f} "
            f"| 外形={sd['tier2']:.4f} "
            f"| 细节={sd['tier3']:.4f}"
        )
        print(
            f"  文件: "
            f"{m['filename']}"
        )
        print(
            f"  描述:\n"
            f"{m['description']}"
        )
        print()

# ============================================================
# 命令行入口
# ============================================================

def main():

    if len( sys.argv ) < 2:

        print(
            "用法: "
            "python -m optimized.search_full_gemini "
            "<图片路径> [返回数量]"
        )
        print(
            "示例: "
            "python -m optimized.search_full_gemini "
            "data/image_input/01.png 10"
        )
        sys.exit( 1 )

    image_path = ( sys.argv[1] )

    top_k = (
        int(
            sys.argv[2]
        )
        if len(
            sys.argv
        ) > 2
        else TOP_K
    )

    if not Path( image_path ).exists():
        print(
            f"[ERROR] "
            f"图片不存在: "
            f"{image_path}"
        )
        sys.exit( 1 )

    matches, query_desc = ( search_similar_packaging( image_path, top_k ) )
    print_results( image_path, query_desc, matches )

if __name__ == "__main__":
    main()
