from analyze_structure import analyze_packaging_image
from analyze_cap import analyze_cap

def analyze_full(image_path):

    # ==========================
    # 第一阶段：整体结构分析
    # ==========================
    structure_result = analyze_packaging_image(image_path)

    # ==========================
    # 提取structure瓶盖判断
    # ==========================
    structure_cap = extract_structure_cap(structure_result)

    # ==========================
    # 如果structure已经判断无盖
    # 直接返回
    # ==========================
    if structure_cap == "无":

        return structure_result

    # ==========================
    # 第二阶段：瓶盖专项分析
    # ==========================
    cap_result = analyze_cap(image_path)

    # ==========================
    # 用cap补充盖子细节
    # ==========================
    final_result = merge_cap_result(
        structure_result,
        cap_result
    )

    return final_result

def extract_structure_cap(structure):

    """
    从structure结果中提取：

    【瓶盖】无
    或
    【瓶盖】xxx
    """

    for line in structure.split("\n"):

        if line.startswith("【瓶盖】"):

            value = line.replace(
                "【瓶盖】",
                ""
            ).strip()

            return value

    return None

def merge_cap_result(structure, cap_result):

    structure_cap_line = None

    # 找structure中的瓶盖字段
    for line in structure.split("\n"):

        if line.startswith("【瓶盖】"):

            structure_cap_line = line
            break

    if structure_cap_line is None:

        return structure

    # ==========================
    # 提取cap描述
    # ==========================

    cap_desc = None

    for line in cap_result.split("\n"):

        if line.startswith("【盖子描述】"):

            cap_desc = line.replace(
                "【盖子描述】",
                ""
            ).strip()

            break

    # ==========================
    # cap没有有效描述
    # 保留structure
    # ==========================

    if (
        cap_desc is None
        or cap_desc == ""
        or cap_desc == "无"
    ):

        return structure

    # ==========================
    # 替换瓶盖字段
    # ==========================

    new_cap_line = (
        "【瓶盖】"
        +
        cap_desc
    )

    structure = structure.replace(
        structure_cap_line,
        new_cap_line
    )

    return structure