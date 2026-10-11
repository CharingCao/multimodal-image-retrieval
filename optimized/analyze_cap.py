import base64
import mimetypes
from openai import OpenAI
from optimized.config_full import (

    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    QWEN_VL_MODEL
)

client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_mime_type(image_path):
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/jpeg"

CAP_PROMPT = """
你是一名化妆品包材瓶盖结构分析师。
第一步：判断是否存在独立瓶盖。
最高优先级判断规则：
透明或半透明保护罩：不要要求明显颜色差异、接缝或分界线。

如果满足：① 覆盖瓶口区域 ②包围泵头或喷头 ③高度超过功能结构 ④形成完整外部封闭轮廓
任意一个条件：直接判断：【盖子存在】有

瓶盖定义：覆盖瓶口并且作为外部保护结构存在。
属于瓶盖：透明保护盖 | 半透明保护盖 | 套盖 | 螺纹盖 | 旋盖 | 翻盖 | 外部保护罩
不属于：单独暴露的泵头、单独暴露的喷雾头、滴管帽、泵头底座、瓶口连接环。
如果泵头或喷头外部存在保护罩：保护罩属于瓶盖。

输出：【盖子存在】有 / 无

第二步：
如果【盖子存在】=无：

输出：【盖子描述】无

如果【盖子存在】=有：只分析独立瓶盖结构。

不要分析：
- 泵头
- 喷嘴
- 滴管
- 瓶颈
- 瓶口螺纹

输出：
【盖子描述】盖子形状，盖顶形态，盖身高度，盖径关系，盖身过渡，盖壁轮廓，盖面纹理，盖体装饰

注意：
1.判断盖径关系：盖子最大直径vs瓶身中部直筒区域最宽处（如果盖子与瓶口大小接近，但明显小于瓶身主体，应判断为内缩）
2.判断盖身高度只计算独立瓶盖主体高度。起点：瓶盖顶部；终点：瓶盖与瓶身连接的位置。（不包含：瓶颈、瓶口连接环、泵头底座、装饰圈）
3.判断优先级：如果顶部只是圆柱盖边缘因为透视产生弧线，一律判断为平顶。
可选值：
盖子形状：圆柱圆形盖（盖体横截面为圆形，但正面具有明显垂直侧壁，顶部通常为平面） | 方形盖（四边直线结构） | 圆角方形盖（四角圆弧处理） | 椭圆形盖（正面轮廓呈椭圆，但存在明确盖壁） | 异形盖 | 球形盖（顶部没有水平平面，整体由连续弧面组成） | 锥形盖
盖顶：平顶 | 微弧顶 | 弧顶(穹顶) | 高穹顶 | 尖顶 | 阶梯顶 | 内凹顶
盖高：高盖（盖高约占瓶身主体高度1/2以上）|中高盖（盖高约占瓶身主体高度1/3~1/2）|中盖（盖高约占瓶身主体高度1/5~1/3）|矮盖（盖高约占瓶身主体高度1/8~1/5）|超矮盖（盖体接近平面，仅有很小高度）
盖径：外扩（盖子最大宽度明显超过瓶身主体宽度） | 齐平（盖子最大宽度与瓶身主体宽度接近） | 内缩（盖子最大宽度明显小于瓶身主体宽度）
盖身过渡：无缝齐平 | 凹槽分隔 | 阶梯过渡 | 卡环过渡 | 外翻唇边 | 收窄过渡
盖壁：直壁 | 上宽下窄 | 上窄下宽 | 弧形外凸壁 | 弧形内凹壁
纹理：光滑 | 竖条防滑纹 | 横条纹 | 菱形纹 | 滚花纹 | 螺纹
装饰：无装饰 | 有装饰环/带 | 有分割线 | 有logo凹印
禁止：颜色、品牌、文字、材质描述
"""

def analyze_cap(image_path):
    b64 = encode_image_to_base64(image_path)
    mime = get_mime_type(image_path)
    response = client.chat.completions.create(
        model=QWEN_VL_MODEL,
        messages=[
            {
             "role":"system",
             "content":CAP_PROMPT
            },
            {
             "role":"user",
             "content":[
                 {
                 "type":"image_url",
                 "image_url":{
                     "url":f"data:{mime};base64,{b64}"
                 }
                 }
             ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
