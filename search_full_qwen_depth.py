"""
Qwen + Depth Bonus 实验版
核心原则：
1. Qwen 保持原 search_full_qwen.py 的主模型地位
2. 不再对 Qwen score 做 Min-Max 归一化
3. Depth 不直接与 Qwen cosine 混合
4. Depth 只根据自己的召回排名提供 bonus
5. depth_alpha = 0 时，只使用 Qwen Top100 候选，尽可能严格复现 standalone Qwen baseline
"""

import sys
import re
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests

from qdrant_client import QdrantClient
from analyze_full import analyze_full
from build_index_full import ( build_tier_text, extract_struct_flags, parse_description )

# ============================================================
# Qdrant
# ============================================================

from config_full import ( COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT )

# ============================================================
# Qwen配置
# ============================================================

from config_full_qwen import (
    QWEN_VECTOR_NAME,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
    TIER_WEIGHTS,
    TOP_K,
)
from embedding import get_embedding
from image_embedding_Qwen import ( get_image_embedding as get_qwen_embedding, )

# Depth查询图片仍然使用CLIP编码
from image_embedding import ( get_image_embedding as get_clip_embedding )

# ============================================================
# Depth配置
# ============================================================

DEPTH_VECTOR_NAME = "clip-depth"

DEPTH_DOWNLOAD_URL = os.getenv("DEPTH_DOWNLOAD_URL", "")

DEPTH_BUCKET = os.getenv("DEPTH_BUCKET", "depthimages")
DEPTH_CACHE_DIR = Path( "data/depth_input")
DEFAULT_DEPTH_ALPHA = 0.05
# ============================================================
# cosine similarity
# ============================================================

def cosine_similarity( a, b, ) -> float:
    a = np.asarray( a, dtype=np.float32, )
    b = np.asarray( b, dtype=np.float32, )
    denominator = ( np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

    return float( np.dot(a, b) / denominator )

# ============================================================
# 获取Depth图片
# ============================================================

def get_depth_image( image_path: str, ) -> Path:
    DEPTH_CACHE_DIR.mkdir( parents=True, exist_ok=True, )
    source_path = Path( image_path )

    depth_key = source_path.stem
    depth_path = ( DEPTH_CACHE_DIR / f"{depth_key}_depth.png" )

    # 已有缓存
    if depth_path.exists():
        print(
            f"  使用已有深度图: "
            f"{depth_path}"
        )
        return depth_path

    print(
        f"  正在下载深度图: "
        f"key={depth_key}"
    )

    if not DEPTH_DOWNLOAD_URL:
        raise RuntimeError("请先设置 DEPTH_DOWNLOAD_URL，再运行深度图检索脚本")

    response = requests.get(
        DEPTH_DOWNLOAD_URL,
        params={ "key": depth_key, "bucket": DEPTH_BUCKET, },
        timeout=60,
    )

    response.raise_for_status()
    content_type = ( response.headers.get( "Content-Type", "", ) )

    if ( "image" not in content_type.lower() ):
        raise RuntimeError(
            "Depth接口返回内容不是图片："
            f"Content-Type={content_type}"
        )

    depth_path.write_bytes( response.content )

    print(
        f"  深度图已保存: "
        f"{depth_path}"
    )
    return depth_path

# ============================================================
# 根据Depth排名计算bonus
# ============================================================

def calculate_depth_rank_bonus( rank: int, candidate_pool: int ) -> float:

    if rank < 1:
        return 0.0

    if rank > candidate_pool:
        return 0.0

    return ( candidate_pool - rank + 1 ) / candidate_pool

# ============================================================
# Tier1规则评分
# ============================================================

def calculate_tier1_score( query_fields, target_description,):
    target_fields = (
        parse_description(
            target_description
        )
    )
    score = 0.0

    # 产品类别 85%
    q_cat = query_fields.get( "产品类别", "", )
    t_cat = target_fields.get( "产品类别", "", )

    if q_cat == t_cat:
        score += 0.85

    # 封口方式 15%
    q_close = query_fields.get(
        "封口方式",
        "",
    )

    t_close = target_fields.get(
        "封口方式",
        "",
    )

    if q_close == t_close:
        score += 0.15

    return score


# ============================================================
# 拆分结构属性
# ============================================================

def split_structure_values(value: str,) -> set[str]:
    if not value:
        return set()

    values = re.split( r"[，,；;、|]", value )

    return { re.sub( r"\s+", "", item, )
        for item in values
        if item.strip()
    }

# ============================================================
# 单字段结构相似度
# ============================================================

def calculate_value_similarity( query_value: str, target_value: str ) -> float:

    query_items = ( split_structure_values( query_value ) )
    target_items = ( split_structure_values( target_value ) )

    if ( not query_items or not target_items ):
        return 0.0
    if query_items == target_items:
        return 1.0
    common_count = len( query_items & target_items )

    return ( 2.0 * common_count / ( len(query_items) + len(target_items) ) )

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

    for ( field_name, field_weight ) in field_weights.items():

        field_score = (
            calculate_value_similarity(
                query_fields.get(
                    field_name,
                    "",
                ),
                target_fields.get(
                    field_name,
                    "",
                ),
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

    for (
        field_name,
        field_weight,
    ) in field_weights.items():

        field_score = (
            calculate_value_similarity(
                query_fields.get(
                    field_name,
                    "",
                ),
                target_fields.get(
                    field_name,
                    "",
                ),
            )
        )

        score += ( field_score * field_weight
        )

    return score

# ============================================================
# 搜索
# ============================================================

def search_similar_packaging(
    image_path: str,
    top_k: int = TOP_K,
    depth_alpha: float = DEFAULT_DEPTH_ALPHA,
) -> tuple[list[dict], str]:

    if depth_alpha < 0:
        raise ValueError(
            "depth_alpha 不能小于0"
        )

    # ========================================================
    # 1/7 Qwen图片向量
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[1/7] 正在生成 Qwen 图片向量..." )
    print( f"{'=' * 60}" )

    qwen_vec = ( get_qwen_embedding( image_path ) )
    print(
        f"  Qwen 向量维度: "
        f"{len(qwen_vec)}"
    )

    # ========================================================
    # 2/7 Depth
    # ========================================================

    depth_vec = None

    print( f"\n{'=' * 60}" )

    print( "[2/7] Depth辅助准备..." )

    print( f"{'=' * 60}" )

    print(
        f"  Depth alpha: "
        f"{depth_alpha:.3f}"
    )

    # Depth属于辅助功能，下载或编码失败时，自动退化为纯Qwen检索，不能中断整个搜索。
    if depth_alpha > 0:
        try:
            depth_path = ( get_depth_image( image_path ) )
            depth_vec = ( get_clip_embedding( str(depth_path) ) )

            print(
                f"  Depth CLIP 向量维度: "
                f"{len(depth_vec)}"
            )

        except Exception as exc:
            print( "  未能获取或编码查询图的Depth图片" )
            print(
                f"  原因: "
                f"{type(exc).__name__}: {exc}"
            )
            print(
                "  本次自动关闭Depth辅助，"
                "继续执行纯Qwen检索"
            )
            # 同时设置这两个值,后面的Depth召回和bonus会自动跳过
            depth_vec = None
            depth_alpha = 0.0

    else:
        print( "alpha=0，跳过Depth下载、编码和召回" )

    # ========================================================
    # 3/7 Qwen-VL结构分析
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[3/7] 正在获取瓶型结构..." )
    print( f"{'=' * 60}" )

    query_description = ( analyze_full( image_path ) )

    print( query_description )

    # ========================================================
    # 4/7 Tier文本向量
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[4/7] 正在解析并生成三层文本向量..." )
    print( f"{'=' * 60}" )

    fields = ( parse_description( query_description ) )
    t1_text = ( build_tier_text( fields, TIER1_FIELDS, ) )
    t2_text = ( build_tier_text( fields, TIER2_FIELDS, ) )
    t3_text = ( build_tier_text( fields, TIER3_FIELDS, ) )

    print(
        f"  Visual总权重: "
        f"{TIER_WEIGHTS['clip']:.0%}"
    )
    print(
        f"  Qwen主视觉 + "
        f"Depth bonus alpha="
        f"{depth_alpha:.3f}"
    )
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

    flags = ( extract_struct_flags( fields ) )
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
    # 5/7 Qwen召回
    # ========================================================

    print( f"\n{'=' * 60}" )
    print(
        f"[5/7] Qwen主召回"
        f"（软约束: "
        f"{pump_label} + {neck_label}）..."
    )
    print( f"{'=' * 60}" )

    qdrant = QdrantClient( host=QDRANT_HOST, port=QDRANT_PORT, )
    candidate_pool = max( top_k, 100, )
    results_qwen = (
        qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=qwen_vec,
            using=QWEN_VECTOR_NAME,
            query_filter=struct_filter,
            limit=candidate_pool,
            with_payload=True,
        )
    )

    print(
        f"  Qwen候选池: "
        f"{len(results_qwen.points)}"
    )

    # ========================================================
    # alpha=0：候选集合严格等于原Qwen Top100
    #
    # alpha>0：才允许Depth候选加入
    # ========================================================

    payloads = {}
    qwen_scores = {}

    for p in results_qwen.points:
        payloads[p.id] = ( p.payload )
        qwen_scores[p.id] = ( p.score )

    depth_rank_bonus = defaultdict( float )

    # ========================================================
    # 6/7 Depth辅助召回
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[6/7] Depth辅助召回..." )
    print( f"{'=' * 60}" )

    if ( depth_alpha > 0 and depth_vec is not None ):
        results_depth = (
            qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=depth_vec,
                using=DEPTH_VECTOR_NAME,
                query_filter=struct_filter,
                limit=candidate_pool,
                with_payload=True,
            )
        )

        print(
            f"  Depth候选池: "
            f"{len(results_depth.points)}"
        )

        # ----------------------------------------------------
        # Depth rank -> bonus
        # ----------------------------------------------------

        for rank, p in enumerate( results_depth.points, start=1, ):
            bonus = ( calculate_depth_rank_bonus( rank, candidate_pool, ) )
            depth_rank_bonus[ p.id ] = bonus

            # Depth候选允许加入候选并集
            if p.id not in payloads:
                payloads[ p.id ] = p.payload

        print(
            f"  Qwen + Depth候选并集: "
            f"{len(payloads)}"
        )

        # ----------------------------------------------------
        # Depth独有候选没有qwen p.score
        #
        # 需要读取它们的Qwen向量，
        # 重新计算与query的Qwen cosine
        # ----------------------------------------------------

        depth_only_ids = [
            pid
            for pid in payloads
            if pid not in qwen_scores
        ]

        if depth_only_ids:

            depth_only_points = (
                qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=depth_only_ids,
                    with_payload=True,
                    with_vectors=[
                        QWEN_VECTOR_NAME
                    ],
                )
            )

            for point in depth_only_points:
                vectors = ( point.vector )

                if not isinstance( vectors, dict, ):
                    continue

                target_qwen_vec = ( vectors.get( QWEN_VECTOR_NAME ) )

                if target_qwen_vec is None:
                    continue

                qwen_scores[ point.id ] = ( cosine_similarity( qwen_vec, target_qwen_vec, ) )
                payloads[ point.id ] = ( point.payload )

        # 去掉没有成功获得Qwen分数的候选
        payloads = {
            pid: payload
            for pid, payload
            in payloads.items()
            if pid in qwen_scores
        }

        print(
            f"  最终有效候选池: "
            f"{len(payloads)}"
        )

    else:
        print( "  alpha=0，完全跳过Depth召回" )
        print(
            f"  最终有效候选池: "
            f"{len(payloads)}"
        )

    # ========================================================
    # 7/7 加权排序
    # ========================================================

    print( f"\n{'=' * 60}" )
    print( "[7/7] Qwen主分 + Depth Bonus + Tier排序..." )
    print( f"{'=' * 60}" )

    scores = defaultdict( float )
    tier_scores = defaultdict(
        lambda: {
            "qwen": 0.0,
            "depth_bonus": 0.0,
            "visual": 0.0,
            "tier1": 0.0,
            "tier2": 0.0,
            "tier3": 0.0,
        }
    )

    visual_weight = ( TIER_WEIGHTS[ "clip" ] )
    w1 = ( TIER_WEIGHTS[ "tier1" ] )
    w2 = ( TIER_WEIGHTS[ "tier2" ] )
    w3 = ( TIER_WEIGHTS[ "tier3" ] )

    # ========================================================
    # Qwen原始score + Depth bonus
    # ========================================================

    for pid in payloads:
        qwen_score = ( qwen_scores[ pid ] )
        bonus = ( depth_rank_bonus.get( pid, 0.0, ) )
        visual_score = ( qwen_score + depth_alpha * bonus )
        tier_scores[ pid ]["qwen"] = ( qwen_score )
        tier_scores[ pid ]["depth_bonus"] = ( bonus )
        tier_scores[ pid ]["visual"] = ( visual_score )
        scores[ pid ] += ( visual_score * visual_weight )

    # ========================================================
    # Tier2 / Tier3
    # ========================================================

    for pid in payloads:

        desc = ( payloads[ pid ].get( "description", "", ) )
        target_fields = ( parse_description( desc ) )

        target_t2_text = (
            build_tier_text(
                target_fields,
                TIER2_FIELDS,
            )
        )
        target_t3_text = (
            build_tier_text(
                target_fields,
                TIER3_FIELDS,
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

        tier2_vector_score = float(
            np.dot(
                vec2,
                target_vec2,
            )
            /
            (
                np.linalg.norm(
                    vec2
                )
                *
                np.linalg.norm(
                    target_vec2
                )
                +
                1e-12
            )
        )

        tier3_vector_score = float(
            np.dot(
                vec3,
                target_vec3,
            )
            /
            (
                np.linalg.norm(
                    vec3
                )
                *
                np.linalg.norm(
                    target_vec3
                )
                +
                1e-12
            )
        )

        tier2_rule_score = (
            calculate_tier2_rule_score(
                fields,
                target_fields,
            )
        )
        tier3_rule_score = (
            calculate_tier3_rule_score(
                fields,
                target_fields,
            )
        )
        tier2_score = (
            tier2_vector_score
            *
            0.70
            +
            tier2_rule_score
            *
            0.30
        )
        tier3_score = (
            tier3_vector_score
            *
            0.70
            +
            tier3_rule_score
            *
            0.30
        )

        tier_scores[ pid ]["tier2"] = ( tier2_score )
        tier_scores[ pid]["tier3"] = ( tier3_score )

        scores[ pid ] += ( tier2_score * w2 )
        scores[ pid ] += ( tier3_score * w3 )

    # ========================================================
    # Tier1
    # ========================================================

    for pid in payloads:

        desc = ( payloads[ pid ].get( "description", "", ) )

        tier1_score = (
            calculate_tier1_score(
                fields,
                desc,
            )
        )

        tier_scores[ pid ]["tier1"] = ( tier1_score )
        scores[ pid ] += ( tier1_score * w1 )

    # ========================================================
    # Soft penalty
    # ========================================================

    for pid in scores:

        target = ( payloads[ pid ] )

        if ( target.get( "has_pump" ) != flags[ "has_pump" ] ):
            scores[ pid ] -= 0.15

        if ( target.get( "has_neck" ) != flags[ "has_neck" ] ):
            scores[ pid ] -= 0.05

    # ========================================================
    # 最终排序
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # ========================================================
    # 返回结果
    # ========================================================

    matches = []

    for ( pid, final_score ) in ranked:
        pl = ( payloads[ pid ] )
        ts = ( tier_scores[ pid ] )

        matches.append(
            {
                "score": final_score,
                "score_detail": ts,
                "filename": pl.get( "filename", "", ),
                "image_path": pl.get( "image_path", "", ),
                "description": pl.get( "description", "", ),
            }
        )
    return ( matches, query_description, )

# ============================================================
# 输出结果
# ============================================================

def print_results(
    query_image: str,
    query_desc: str,
    matches: list[dict],
):

    print( f"\n{'=' * 60}" )
    print( "搜索结果" )
    print( f"{'=' * 60}" )
    print(
        f"查询图片: "
        f"{query_image}\n"
    )

    for ( i, m, ) in enumerate( matches, 1 ):

        sd = ( m[ "score_detail" ] )

        print(
            f"--- 第 {i} 名 "
            f"(加权总分: "
            f"{m['score']:.4f}) ---"
        )
        print(
            f"  视觉分: "
            f"Qwen={sd['qwen']:.4f} "
            f"| Depth Bonus="
            f"{sd['depth_bonus']:.4f} "
            f"| Visual="
            f"{sd['visual']:.4f}"
        )
        print(
            f"  分层得分: "
            f"类别={sd['tier1']:.4f} "
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

def main():
    if len( sys.argv) < 2:

        print(
            "用法: "
            "python search_full_qwen_depth_bonus.py "
            "<图片路径> "
            "[返回数量] "
            "[depth_alpha]"
        )

        print( "\n示例：" )
        print( "验证Qwen baseline：" )
        print(
            "python search_full_qwen_depth_bonus.py "
            "\"data/image_input/03.png\" "
            "10 0"
        )
        print( "\nDepth辅助 alpha=0.05：" )
        print(
            "python search_full_qwen_depth_bonus.py "
            "\"data/image_input/03.png\" "
            "10 0.05"
        )

        sys.exit( 1 )

    image_path = ( sys.argv[ 1 ] )

    top_k = ( int( sys.argv[ 2 ] )
        if len( sys.argv ) > 2
        else TOP_K
    )

    depth_alpha = ( float( sys.argv[ 3 ] )
        if len( sys.argv ) > 3
        else DEFAULT_DEPTH_ALPHA
    )

    if not Path( image_path ).exists():

        print(
            f"[ERROR] "
            f"图片不存在: "
            f"{image_path}"
        )
        sys.exit( 1 )

    matches, query_desc = (
        search_similar_packaging(
            image_path,
            top_k,
            depth_alpha=depth_alpha,
        )
    )
    print_results( image_path, query_desc, matches, )

if __name__ == "__main__":
    main()
