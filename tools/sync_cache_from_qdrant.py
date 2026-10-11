import os
import json
from qdrant_client import QdrantClient

from shared.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME
)


# =========================
# cache文件位置
# =========================

CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "descriptions_cache.json"
)


# =========================
# 读取旧cache
# =========================

def load_cache():

    if os.path.exists(CACHE_FILE):

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    return {}



# =========================
# 保存cache
# =========================

def save_cache(cache):

    os.makedirs(
        os.path.dirname(CACHE_FILE),
        exist_ok=True
    )

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2
        )



# =========================
# 从Qdrant同步
# =========================

def sync():

    print("连接Qdrant...")


    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT
    )


    print(
        "读取collection:",
        COLLECTION_NAME
    )


    cache = load_cache()


    print(
        "当前cache数量:",
        len(cache)
    )


    next_page = None

    total = 0
    added = 0


    while True:


        points, next_page = client.scroll(

            collection_name=COLLECTION_NAME,

            limit=100,

            offset=next_page,

            with_payload=True,

            with_vectors=False
        )


        if not points:
            break



        for point in points:


            payload = point.payload or {}


            filename = payload.get(
                "filename"
            )

            description = payload.get(
                "description"
            )


            if not filename:
                continue


            if not description:
                continue



            total += 1



            # 已存在的不覆盖
            if filename in cache:

                continue



            cache[filename] = {

                "description": description

            }


            added += 1



        print(
            f"扫描 {total} 条，新增 {added} 条"
        )



        if next_page is None:
            break



    save_cache(cache)


    print("====================")
    print("同步完成")
    print(
        "Qdrant读取:",
        total
    )
    print(
        "新增cache:",
        added
    )
    print(
        "最终cache:",
        len(cache)
    )



if __name__ == "__main__":

    sync()
