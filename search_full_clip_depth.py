"""
CLIP + Depth 双视觉检索实验版

流程：
1. 原图 -> CLIP embedding
2. 根据原图文件名下载对应深度图
3. 深度图 -> 同一个 CLIP embedding
4. Qdrant:
      clip       Top100
      clip-depth Top100
5. 两路候选取并集
6. 对并集候选统一计算：
      clip_score
      depth_score
7. visual_score =
      CLIP_RATIO * clip_score
      +
      DEPTH_RATIO * depth_score
8. 保持原来的 Tier1 / Tier2 / Tier3 + soft penalty
9. 最终排序

注意：
- 不修改原 search_full.py
- Depth key = 原图文件名去掉扩展名
"""

import sys
import re
import os
import requests
import numpy as np

from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

from analyze_full import analyze_full

from build_index_full import (
    build_tier_text,
    extract_struct_flags,
    parse_description,
)

from config_full import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    CLIP_VECTOR_NAME,
    DEPTH_VECTOR_NAME,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
    TIER_WEIGHTS,
    TOP_K,
)

from embedding import get_embedding

from image_embedding import ( get_image_embedding as get_clip_embedding)


# ============================================================
# Depth 配置
# ============================================================

DEPTH_DOWNLOAD_URL = os.getenv("DEPTH_DOWNLOAD_URL", "")

DEPTH_BUCKET = os.getenv("DEPTH_BUCKET", "depthimages")

CLIP_VECTOR_NAME = "clip"
DEPTH_VECTOR_NAME = "clip-depth"


# ============================================================
# CLIP / Depth 视觉内部比例
#
# 第一轮先：
# 原图 CLIP = 70%
# Depth     = 30%
#
# 注意：
# 这不是整个系统的 Tier 权重。
# 两者融合后仍作为一个 visual_score，
# 再乘 config_full.py 里的 TIER_WEIGHTS["clip"]。
# ============================================================

CLIP_RATIO = 0.70
DEPTH_RATIO = 0.30


# ============================================================
# Tier1规则评分
# ============================================================

def calculate_tier1_score(
    query_fields,
    target_description
):
    """
    Tier1规则评分：

    产品类别 85%
    封口方式 15%
    """

    target_fields = parse_description(
        target_description
    )

    score = 0.0

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


# ============================================================
# 拆分结构属性
# ============================================================

def split_structure_values(
    value: str
) -> set[str]:

    if not value:
        return set()

    values = re.split(
        r"[，,；;、|]",
        value,
    )

    return {
        re.sub(
            r"\s+",
            "",
            item
        )
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

    query_items = split_structure_values(
        query_value
    )

    target_items = split_structure_values(
        target_value
    )

    if (
        not query_items
        or
        not target_items
    ):
        return 0.0

    if query_items == target_items:
        return 1.0

    common_count = len(
        query_items
        &
        target_items
    )

    return (
        2.0
        *
        common_count
        /
        (
            len(query_items)
            +
            len(target_items)
        )
    )


# ============================================================
# Tier2规则分
# ============================================================

def calculate_tier2_rule_score(
    query_fields,
    target_fields,
):

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
                query_fields.get(
                    field_name,
                    ""
                ),
                target_fields.get(
                    field_name,
                    ""
                ),
            )
        )

        score += (
            field_score
            *
            field_weight
        )

    return score


# ============================================================
# Tier3规则分
# ============================================================

def calculate_tier3_rule_score(
    query_fields,
    target_fields,
):

    field_weights = {
        "瓶颈": 0.20,
        "瓶肩": 0.20,
        "瓶身": 0.40,
        "瓶底": 0.20,
    }

    score = 0.0

    for (
        field_name,
        field_weight
    ) in field_weights.items():

        field_score = (
            calculate_value_similarity(
                query_fields.get(
                    field_name,
                    ""
                ),
                target_fields.get(
                    field_name,
                    ""
                ),
            )
        )

        score += (
            field_score
            *
            field_weight
        )

    return score


# ============================================================
# cosine
# ============================================================

def cosine_similarity(
    vec1,
    vec2
):

    a = np.asarray(
        vec1,
        dtype=np.float32
    )

    b = np.asarray(
        vec2,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# 下载深度图
# ============================================================

def download_depth_image(
    original_image_path: str
) -> str:

    """
    例如：

    data/image_input/03.png

    key:
    03

    保存：
    data/depth_input/03_depth.png
    """

    original_path = Path(
        original_image_path
    )

    depth_key = (
        original_path.stem
    )

    output_dir = Path(
        "data/depth_input"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        /
        f"{depth_key}_depth.png"
    )


    # ========================================================
    # 已经下载过就直接使用缓存
    # ========================================================

    if output_path.exists():

        print(
            f"  使用已有深度图: "
            f"{output_path}"
        )

        return str(
            output_path
        )


    print(
        f"  正在下载深度图，"
        f"key={depth_key}"
    )


    if not DEPTH_DOWNLOAD_URL:
        raise RuntimeError("请先设置 DEPTH_DOWNLOAD_URL，再运行深度图检索脚本")

    response = requests.get(
        DEPTH_DOWNLOAD_URL,
        params={
            "key": depth_key,
            "bucket": DEPTH_BUCKET,
        },
        timeout=30,
    )


    if response.status_code != 200:

        raise RuntimeError(
            f"深度图下载失败："
            f"key={depth_key}, "
            f"status={response.status_code}, "
            f"response={response.text}"
        )


    content_type = (
        response.headers.get(
            "Content-Type",
            ""
        )
    )


    if not content_type.startswith(
        "image/"
    ):

        raise RuntimeError(
            f"深度图返回格式异常："
            f"Content-Type={content_type}"
        )


    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            response.content
        )


    print(
        f"  深度图已保存: "
        f"{output_path}"
    )


    return str(
        output_path
    )


# ============================================================
# 主搜索
# ============================================================

def search_similar_packaging(
    image_path: str,
    top_k: int = TOP_K,
    clip_ratio: float = CLIP_RATIO,
    depth_ratio: float = DEPTH_RATIO,
) -> tuple[list[dict], str]:

    """
    CLIP + Depth 双视觉检索。
    """

    # ========================================================
    # 检查比例
    # ========================================================

    ratio_sum = (
        clip_ratio
        +
        depth_ratio
    )

    if ratio_sum <= 0:

        raise ValueError(
            "clip_ratio + depth_ratio 必须 > 0"
        )


    # 归一化
    clip_ratio = (
        clip_ratio
        /
        ratio_sum
    )

    depth_ratio = (
        depth_ratio
        /
        ratio_sum
    )


    # ========================================================
    # 1/7 原图CLIP
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[1/7] 正在生成原图 CLIP 向量..."
    )

    print(
        f"{'=' * 60}"
    )


    clip_vec = get_clip_embedding(
        image_path
    )


    print(
        f"  原图 CLIP 向量维度: "
        f"{len(clip_vec)}"
    )


    # ========================================================
    # 2/7 下载Depth + 生成Depth CLIP
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[2/7] 正在获取并编码深度图..."
    )

    print(
        f"{'=' * 60}"
    )


    depth_image_path = (
        download_depth_image(
            image_path
        )
    )


    depth_vec = (
        get_clip_embedding(
            depth_image_path
        )
    )


    print(
        f"  Depth CLIP 向量维度: "
        f"{len(depth_vec)}"
    )


    # ========================================================
    # 3/7 Qwen-VL结构分析
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[3/7] 正在用 Qwen-VL 分析瓶型结构..."
    )

    print(
        f"{'=' * 60}"
    )


    query_description = (
        analyze_full(
            image_path
        )
    )


    print(
        query_description
    )


    # ========================================================
    # 4/7 Tier文本
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[4/7] 正在解析并生成三层文本向量..."
    )

    print(
        f"{'=' * 60}"
    )


    fields = parse_description(
        query_description
    )


    t1_text = build_tier_text(
        fields,
        TIER1_FIELDS
    )

    t2_text = build_tier_text(
        fields,
        TIER2_FIELDS
    )

    t3_text = build_tier_text(
        fields,
        TIER3_FIELDS
    )


    print(
        f"  Visual内部比例: "
        f"CLIP={clip_ratio:.0%} "
        f"| Depth={depth_ratio:.0%}"
    )


    print(
        f"  Visual总权重: "
        f"{TIER_WEIGHTS['clip']:.0%}"
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


    vec2 = get_embedding(
        t2_text
    )

    vec3 = get_embedding(
        t3_text
    )


    # ========================================================
    # flags
    # ========================================================

    flags = extract_struct_flags(
        fields
    )


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
    # 5/7 CLIP + Depth 双路召回
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"[5/7] CLIP + Depth 双路召回"
        f"（软约束: "
        f"{pump_label} + {neck_label}）..."
    )

    print(
        f"{'=' * 60}"
    )


    qdrant = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )


    candidate_pool = max(
        top_k,
        100
    )


    # ========================================================
    # 原图CLIP
    # ========================================================

    results_clip = (
        qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=clip_vec,
            using=CLIP_VECTOR_NAME,
            query_filter=struct_filter,
            limit=candidate_pool,
            with_payload=True,
        )
    )


    print(
        f"  CLIP候选池: "
        f"{len(results_clip.points)}"
    )


    # ========================================================
    # Depth CLIP
    # ========================================================

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


    # ========================================================
    # 候选并集
    # ========================================================

    payloads = {}


    for p in results_clip.points:

        payloads[
            p.id
        ] = (
            p.payload
            or
            {}
        )


    for p in results_depth.points:

        payloads[
            p.id
        ] = (
            p.payload
            or
            {}
        )


    candidate_ids = list(
        payloads.keys()
    )


    print(
        f"  合并去重后候选池: "
        f"{len(candidate_ids)}"
    )


    # ========================================================
    # 6/7 获取并集候选两个视觉向量
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[6/7] 正在统一计算 CLIP / Depth 视觉分..."
    )

    print(
        f"{'=' * 60}"
    )


    retrieved_points = (
        qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=candidate_ids,
            with_payload=False,
            with_vectors=[
                CLIP_VECTOR_NAME,
                DEPTH_VECTOR_NAME,
            ],
        )
    )


    visual_scores = {}


    for point in retrieved_points:

        vectors = (
            point.vector
        )


        if not isinstance(
            vectors,
            dict
        ):
            continue


        target_clip_vec = (
            vectors.get(
                CLIP_VECTOR_NAME
            )
        )


        target_depth_vec = (
            vectors.get(
                DEPTH_VECTOR_NAME
            )
        )


        if (
            target_clip_vec is None
            or
            target_depth_vec is None
        ):
            continue


        clip_score = (
            cosine_similarity(
                clip_vec,
                target_clip_vec
            )
        )


        depth_score = (
            cosine_similarity(
                depth_vec,
                target_depth_vec
            )
        )


        visual_score = (
            clip_score
            *
            clip_ratio
            +
            depth_score
            *
            depth_ratio
        )


        visual_scores[
            point.id
        ] = {
            "clip": clip_score,
            "depth": depth_score,
            "visual": visual_score,
        }


    print(
        f"  成功计算双视觉分: "
        f"{len(visual_scores)} / "
        f"{len(candidate_ids)}"
    )


    # ========================================================
    # 7/7 Tier融合
    # ========================================================

    print(
        f"\n{'=' * 60}"
    )

    print(
        "[7/7] CLIP + Depth + Tier1/2/3 加权排序..."
    )

    print(
        f"{'=' * 60}"
    )


    scores = defaultdict(
        float
    )


    tier_scores = defaultdict(
        lambda: {
            "clip": 0.0,
            "depth": 0.0,
            "visual": 0.0,
            "tier1": 0.0,
            "tier2": 0.0,
            "tier3": 0.0,
        }
    )


    visual_weight = (
        TIER_WEIGHTS[
            "clip"
        ]
    )

    w1 = (
        TIER_WEIGHTS[
            "tier1"
        ]
    )

    w2 = (
        TIER_WEIGHTS[
            "tier2"
        ]
    )

    w3 = (
        TIER_WEIGHTS[
            "tier3"
        ]
    )


    valid_ids = []


    # ========================================================
    # 双视觉分
    # ========================================================

    for pid in candidate_ids:

        if pid not in visual_scores:
            continue


        valid_ids.append(
            pid
        )


        vs = visual_scores[
            pid
        ]


        tier_scores[
            pid
        ]["clip"] = (
            vs["clip"]
        )


        tier_scores[
            pid
        ]["depth"] = (
            vs["depth"]
        )


        tier_scores[
            pid
        ]["visual"] = (
            vs["visual"]
        )


        scores[
            pid
        ] += (
            vs["visual"]
            *
            visual_weight
        )


    # ========================================================
    # Tier2 / Tier3
    # ========================================================

    for pid in valid_ids:

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
        # Tier2 vector
        # ====================================================

        tier2_vector_score = float(
            np.dot(
                vec2,
                target_vec2
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


        # ====================================================
        # Tier3 vector
        # ====================================================

        tier3_vector_score = float(
            np.dot(
                vec3,
                target_vec3
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


        # ====================================================
        # rule
        # ====================================================

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


        # ====================================================
        # 保持原逻辑：
        # 70% vector + 30% rule
        # ====================================================

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


        scores[
            pid
        ] += (
            tier2_score
            *
            w2
        )


        scores[
            pid
        ] += (
            tier3_score
            *
            w3
        )


    # ========================================================
    # Tier1
    # ========================================================

    for pid in valid_ids:

        desc = payloads[
            pid
        ].get(
            "description",
            ""
        )


        tier1_score = (
            calculate_tier1_score(
                fields,
                desc
            )
        )


        tier_scores[
            pid
        ]["tier1"] = (
            tier1_score
        )


        scores[
            pid
        ] += (
            tier1_score
            *
            w1
        )


    # ========================================================
    # soft penalty
    # ========================================================

    for pid in valid_ids:

        target = payloads[
            pid
        ]


        if (
            target.get(
                "has_pump"
            )
            !=
            flags[
                "has_pump"
            ]
        ):

            scores[
                pid
            ] -= 0.15


        if (
            target.get(
                "has_neck"
            )
            !=
            flags[
                "has_neck"
            ]
        ):

            scores[
                pid
            ] -= 0.05


    # ========================================================
    # 最终排序
    # ========================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[
        :top_k
    ]


    matches = []


    for (
        pid,
        final_score
    ) in ranked:

        payload = payloads[
            pid
        ]

        ts = tier_scores[
            pid
        ]


        matches.append(
            {
                "score": final_score,

                "score_detail": ts,

                "filename": payload.get(
                    "filename",
                    ""
                ),

                "image_path": payload.get(
                    "image_path",
                    ""
                ),

                "description": payload.get(
                    "description",
                    ""
                ),
            }
        )


    return (
        matches,
        query_description
    )


# ============================================================
# 输出
# ============================================================

def print_results(
    query_image,
    query_desc,
    matches
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        "CLIP + Depth 搜索结果"
    )

    print(
        f"{'=' * 60}"
    )

    print(
        f"查询图片: "
        f"{query_image}\n"
    )


    for i, item in enumerate(
        matches,
        start=1
    ):

        sd = item[
            "score_detail"
        ]


        print(
            f"--- 第 {i} 名 "
            f"(加权总分: "
            f"{item['score']:.4f}) ---"
        )


        print(
            f"  分层得分: "
            f"CLIP={sd['clip']:.4f} "
            f"| Depth={sd['depth']:.4f} "
            f"| 视觉融合={sd['visual']:.4f} "
            f"| 类别={sd['tier1']:.4f} "
            f"| 外形={sd['tier2']:.4f} "
            f"| 细节={sd['tier3']:.4f}"
        )


        print(
            f"  文件: "
            f"{item['filename']}"
        )


        print(
            f"  描述:\n"
            f"{item['description']}"
        )


        print()


# ============================================================
# main
# ============================================================

def main():

    if len(
        sys.argv
    ) < 2:

        print(
            "用法："
            "python search_full_clip_depth.py "
            "<图片路径> [返回数量]"
        )

        print(
            "示例："
            "python search_full_clip_depth.py "
            "\"data/image_input/03.png\" 10"
        )

        sys.exit(
            1
        )


    image_path = (
        sys.argv[
            1
        ]
    )


    top_k = (
        int(
            sys.argv[
                2
            ]
        )
        if len(
            sys.argv
        ) > 2
        else TOP_K
    )


    if not Path(
        image_path
    ).exists():

        print(
            f"[ERROR] "
            f"图片不存在: "
            f"{image_path}"
        )

        sys.exit(
            1
        )


    matches, query_desc = (
        search_similar_packaging(
            image_path,
            top_k
        )
    )


    print_results(
        image_path,
        query_desc,
        matches
    )


if __name__ == "__main__":
    main()
