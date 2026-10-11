"""
批量分析包材库图片 → 生成描述 → 三层向量化 → 存入 Qdrant。

支持断点续传：已分析过的图片（描述缓存在 JSON 文件中）不会重复调用 VL 模型。

命令行：
  python -m optimized.build_index_full
      全量处理 data/images/ 下所有图片，点 id 为 1..N（与排序一致），已存在的 id 会被覆盖。

  python -m optimized.build_index_full --recreate
      删除当前集合内全部向量数据后重新建库（集合定义会重建）。

  python -m optimized.build_index_full --skip-indexed
      跳过 Qdrant 里已有 filename 的图片，只为新图建向量；新点 id 接在当前最大 id 之后。
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from optimized.analyze_full import analyze_full
from optimized.config_full import (
    CLIP_DIM,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    IMAGES_DIR,
    QDRANT_HOST,
    QDRANT_PORT,
    TIER1_FIELDS,
    TIER2_FIELDS,
    TIER3_FIELDS,
)
from optimized.text_embedding import get_embedding
from optimized.clip_embedding import get_image_embedding

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "descriptions_cache_full.json")


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def parse_description(description: str) -> dict[str, str]:
    """将结构化描述解析为 {字段名: 值} 字典。"""
    fields = {}
    for line in description.strip().splitlines():
        line = line.strip()
        m = re.match(r"【(.+?)】(.+)", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def extract_struct_flags(fields: dict[str, str]) -> dict[str, bool]:
    """从解析后的字段中提取结构性布尔标志，用于 Qdrant 硬过滤。"""
    pump_val = fields.get("泵头", "无").strip()
    neck_val = fields.get("瓶颈", "无").strip()
    return {
        "has_pump": pump_val != "无",
        "has_neck": not neck_val.startswith("无"),
    }


def build_tier_text(fields: dict[str, str], tier_keys: list[str]) -> str:
    """根据字段名列表，拼接该层的文本。"""
    parts = []
    for key in tier_keys:
        val = fields.get(key, "")
        if val:
            parts.append(f"【{key}】{val}")
    return " ".join(parts)


def get_image_files() -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp","PNG"}
    files = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if Path(fname).suffix.lower() in exts:
            files.append(fname)
    return files


def ensure_collection(qdrant: QdrantClient):
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "clip": VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
                "tier1": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                "tier2": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
                "tier3": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            },
        )
        print(f"[INFO] 已创建集合(CLIP + 三层文本向量): {COLLECTION_NAME}")
    else:
        print(f"[INFO] 集合已存在: {COLLECTION_NAME}")


def get_indexed_filenames_and_max_id(qdrant: QdrantClient) -> tuple[set[str], int]:
    """扫描集合中已有点的 filename 与数值型 id 的最大值。"""
    indexed: set[str] = set()
    max_id = 0
    offset = None
    while True:
        result, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in result:
            pl = p.payload or {}
            fn = pl.get("filename")
            if fn:
                indexed.add(fn)
            pid = p.id
            if isinstance(pid, int):
                max_id = max(max_id, pid)
        if next_offset is None:
            break
        offset = next_offset
    return indexed, max_id


def build(*, recreate: bool = False, skip_indexed: bool = False):
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    if recreate:
        names = [c.name for c in qdrant.get_collections().collections]
        if COLLECTION_NAME in names:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
            print(f"[INFO] 已删除集合: {COLLECTION_NAME}")
        ensure_collection(qdrant)
    else:
        ensure_collection(qdrant)

    image_files = get_image_files()
    if not image_files:
        print("[WARN] data/images/ 下没有可索引的图片")
        return

    indexed: set[str] = set()
    max_id = 0
    if skip_indexed:
        indexed, max_id = get_indexed_filenames_and_max_id(qdrant)
        print(f"[INFO] Qdrant 中已有 {len(indexed)} 个 filename，当前最大点 id = {max_id}")

    if skip_indexed:
        to_process = [f for f in image_files if f not in indexed]
        skipped_n = len(image_files) - len(to_process)
        print(f"[INFO] 将跳过已索引 {skipped_n} 张，待处理 {len(to_process)} 张")
        if not to_process:
            print("[DONE] 没有新图需要入库")
            return
    else:
        to_process = image_files

    total = len(to_process)
    print(f"[INFO] 本次处理 {total} 张包材图片（目录中共 {len(image_files)} 张）")

    cache = load_cache()
    points = []

    for offset, fname in enumerate(to_process):
        point_id = max_id + offset + 1 if skip_indexed else offset + 1
        prog = offset + 1
        image_path = os.path.join(IMAGES_DIR, fname)

        if fname in cache and cache[fname].get("description"):
            description = cache[fname]["description"]
            print(f"[{prog}/{total}] id={point_id} {fname} - 使用缓存描述")
        else:
            print(f"[{prog}/{total}] id={point_id} {fname} - 正在分析...")
            try:
                description = analyze_full(image_path)
                cache[fname] = {"description": description}
                save_cache(cache)
            except Exception as e:
                print(f"  [ERROR] 分析失败: {e}")
                time.sleep(2)
                continue

        fields = parse_description(description)
        t1_text = build_tier_text(fields, TIER1_FIELDS)
        t2_text = build_tier_text(fields, TIER2_FIELDS)
        t3_text = build_tier_text(fields, TIER3_FIELDS)

        try:
            clip_vec = get_image_embedding(image_path)
        except Exception as e:
            print(f"  [ERROR] CLIP向量化失败: {e}")
            continue

        try:
            vec1 = get_embedding(t1_text)
            vec2 = get_embedding(t2_text)
            vec3 = get_embedding(t3_text)
        except Exception as e:
            print(f"  [ERROR] 文本向量化失败: {e}")
            time.sleep(1)
            continue

        flags = extract_struct_flags(fields)
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "clip": clip_vec,
                    "tier1": vec1,
                    "tier2": vec2,
                    "tier3": vec3,
                },
                payload={
                    "filename": fname,
                    "image_path": image_path,
                    "description": description,
                    "has_pump": flags["has_pump"],
                    "has_neck": flags["has_neck"],
                },
            )
        )

        if len(points) >= 10:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  [INFO] 已写入 {len(points)} 条向量")
            points = []

        time.sleep(0.3)

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  [INFO] 已写入最后 {len(points)} 条向量")

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\n[DONE] 集合 {COLLECTION_NAME} 共有 {info.points_count} 条记录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建索引：VL + CLIP + 文本向量写入 Qdrant")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除当前集合（清空全部向量）后重新创建并全量建索引",
    )
    parser.add_argument(
        "--skip-indexed",
        action="store_true",
        help="跳过 Qdrant 中 payload.filename 已存在的图片，仅处理新文件并续接点 id",
    )
    args = parser.parse_args()
    build(recreate=args.recreate, skip_indexed=args.skip_indexed)
