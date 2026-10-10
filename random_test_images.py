import os
import random
import shutil

# 原始图库
source_dir = "data/images"

# 测试集目录
target_dir = "data/image_input"

# 抽取数量
sample_num = 30

# 支持的图片格式
image_ext = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
]

# 获取所有图片
images = [
    f for f in os.listdir(source_dir)
    if os.path.splitext(f)[1].lower() in image_ext
]

print(f"图库共有 {len(images)} 张图片")

# 随机抽取
print("图片数量:", len(images))
print("抽取数量:", sample_num)

selected = random.sample(images, sample_num)

print("抽取结果:", selected)

# 如果测试目录不存在，创建
os.makedirs(target_dir, exist_ok=True)


# 复制并自动重命名
for idx, img in enumerate(selected, start=1):

    src = os.path.join(source_dir, img)

    # 获取原图片后缀
    ext = os.path.splitext(img)[1].lower()

    # 新文件名：01.png、02.jpg...
    new_name = f"{idx:02d}{ext}"

    dst = os.path.join(target_dir, new_name)

    shutil.copy(src, dst)

print("\n抽取完成:")
for img in selected:
    print(img)

print(f"\n共复制 {len(selected)} 张图片到 {target_dir}")