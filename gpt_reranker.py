import base64
import json
import mimetypes
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)

GPT_BASE_URL = os.getenv("GPT_BASE_URL")
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_API_KEY = os.getenv("GPT_API_KEY")


SYSTEM_PROMPT = (
    "你是化妆品包材结构相似度评审模型。"
    "候选图可能包含多个独立包材主体。你必须逐一观察每个完整主体，"
    "选择与搜索主体最相似的一个主体，只评价该最佳匹配主体。"
    "不得综合评价整张候选图，不得对多个主体取平均分；"
    "不得因为存在其他瓶子、罐子、泵头、瓶盖或分离零件而扣分。"
    "reason必须说明选中的是左侧、右侧、中间或第几个主体。"
    "禁止用‘包含多种结构’、‘多数主体不同’或‘结构混杂’作为扣分理由。"
    "只能比较几何结构，包括整体轮廓、高宽比例、瓶盖、泵头、瓶颈、"
    "瓶肩、瓶身、瓶底和部件连接关系。必须忽略颜色、透明度、材质、"
    "反光、品牌、文字、标签、图案、内容物和背景。"
    "透明罩与不透明罩如果外轮廓和连接结构相同，必须视为相同结构。"
    "structure_score使用0到100分，并按分数从高到低输出全部候选。"
)


def _image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_image_path(match: dict, images_dir: Path) -> Path:
    candidates = []
    if match.get("image_path"):
        candidates.append(Path(match["image_path"]))
    if match.get("filename"):
        candidates.append(images_dir / match["filename"])

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    raise FileNotFoundError(
        f"候选图片不存在：filename={match.get('filename', '')}, "
        f"image_path={match.get('image_path', '')}"
    )


def _build_content(query_path: Path, ordered_candidates: list[dict]) -> list[dict]:
    content = [
        {
            "type": "text",
            "text": (
                "第一张是搜索图。后面是候选图。对每张多主体候选图，"
                "只选择其中与搜索主体最相似的一个完整主体评分。"
            ),
        },
        {"type": "text", "text": "【搜索图】"},
        {
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(query_path), "detail": "high"},
        },
    ]

    for number, candidate in enumerate(ordered_candidates, start=1):
        content.extend(
            [
                {"type": "text", "text": f"【候选图 {number}】"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(candidate["resolved_path"]),
                        "detail": "high",
                    },
                },
            ]
        )

    content.append(
        {
            "type": "text",
            "text": (
                "必须输出全部候选，每个编号只能出现一次。只输出JSON。"
                '格式：{"ranking":[{"candidate":1,'
                '"structure_score":0,"reason":"所选主体及简短理由"}]}'
            ),
        }
    )
    return content


def _parse_ranking(answer: str, ordered_candidates: list[dict]) -> list[dict]:
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    ranking = json.loads(cleaned)["ranking"]
    if len(ranking) != len(ordered_candidates):
        raise ValueError("GPT没有返回全部候选")

    seen = set()
    parsed = []
    for rank, item in enumerate(ranking, start=1):
        number = int(item["candidate"])
        if number in seen or not 1 <= number <= len(ordered_candidates):
            raise ValueError(f"GPT候选编号无效或重复：{number}")
        seen.add(number)
        candidate = ordered_candidates[number - 1]
        parsed.append(
            {
                "original_index": candidate["original_index"],
                "rank": rank,
                "score": float(item["structure_score"]),
                "reason": item.get("reason", ""),
            }
        )
    return parsed

def _request_one_round(
    client: OpenAI,
    query_path: Path,
    ordered_candidates: list[dict],
) -> list[dict]:
    response = client.chat.completions.create(
        model=GPT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_content(query_path, ordered_candidates)},
        ],
        max_tokens=4000,
    )
    answer = response.choices[0].message.content or ""
    return _parse_ranking(answer, ordered_candidates)

def rerank_with_gpt(
    query_image_path: str,
    matches: list[dict],
    images_dir: str | Path,
    rounds: int = 3,
) -> list[dict]:
    
    """GPT多顺序投票分组；组内始终保留Qwen原始顺序。"""
    if not matches or len(matches) < 2:
        return matches
    if rounds != 3:
        raise ValueError("当前稳定重排固定使用3轮")
    if not all([GPT_BASE_URL, GPT_MODEL, GPT_API_KEY]):
        raise RuntimeError("GPT配置不完整，请检查.env")

    query_path = Path(query_image_path)
    if not query_path.exists():
        raise FileNotFoundError(f"搜索图片不存在：{query_path}")

    images_dir = Path(images_dir)
    candidates = []
    for index, match in enumerate(matches):
        candidates.append(
            {
                "original_index": index,
                "match": match,
                "resolved_path": _resolve_image_path(match, images_dir),
            }
        )

    original_order = list(candidates)
    reverse_order = list(reversed(candidates))
    random_order = list(candidates)
    random.Random(20260827).shuffle(random_order)

    client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)
    results = []
    for round_number, order in enumerate(
        [original_order, reverse_order, random_order], start=1
    ):
        print(f"  GPT重排第{round_number}/3轮...")
        results.append(_request_one_round(client, query_path, order))

    # “晋级组”只表示三轮中至少两次进入候选池前半区，不是绝对相似度标签。
    promoted_size = (len(candidates) + 1) // 2
    stats = {
        index: {"ranks": [], "scores": [], "promoted_count": 0, "reasons": []}
        for index in range(len(candidates))
    }
    for result in results:
        for item in result:
            stat = stats[item["original_index"]]
            stat["ranks"].append(item["rank"])
            stat["scores"].append(item["score"])
            stat["reasons"].append(item["reason"])
            if item["rank"] <= promoted_size:
                stat["promoted_count"] += 1

    promoted = []
    remaining = []
    for candidate in original_order:
        index = candidate["original_index"]
        stat = stats[index]
        enriched = dict(candidate["match"])
        enriched["gpt_rerank"] = {
            "round_ranks": stat["ranks"],
            "average_rank": sum(stat["ranks"]) / len(stat["ranks"]),
            "round_scores": stat["scores"],
            "average_score": sum(stat["scores"]) / len(stat["scores"]),
            "promoted_count": stat["promoted_count"],
            "group": "promoted" if stat["promoted_count"] >= 2 else "remaining",
            "reasons": stat["reasons"],
        }
        if stat["promoted_count"] >= 2:
            promoted.append(enriched)
        else:
            remaining.append(enriched)

    return promoted + remaining