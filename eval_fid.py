import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# 1. 检查参数
if len(sys.argv) < 2:
    print("用法: python simple_fid_resized.py <图片文件夹路径>")
    sys.exit(1)
folder_path = sys.argv[1]

# 2. 设置设备和 FID 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# feature=2048 是标准 InceptionV3 特征层
fid = FrechetInceptionDistance(feature=2048).to(device)

# 3. 定义预处理：转 Tensor 并 强制缩放到 299x299 (对应 downsample)
# PIL 读取原本就是 0-255 的 uint8，所以不需要 * 255
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),  # 显式缩放到 Inception 标准尺寸
    transforms.PILToTensor()  # 转为 (C, H, W) 的 uint8 Tensor
])

print(f"处理路径: {folder_path} | 设备: {device}")

# 4. 遍历并计算
files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

if not files:
    print("该路径下没有图片。")
    sys.exit(1)

for f in files:
    try:
        # 读取
        img = Image.open(os.path.join(folder_path, f)).convert('RGB')
        w, h = img.size

        # 切割：左边 Fake，右边 Real
        fake_part = img.crop((0, 0, w // 2, h))
        real_part = img.crop((w // 2, 0, w, h))

        # 预处理：缩放 + 转Tensor + 增加Batch维度 -> [1, 3, 299, 299]
        fake_tensor = preprocess(fake_part).unsqueeze(0).to(device)
        real_tensor = preprocess(real_part).unsqueeze(0).to(device)

        # 放入 FID 更新队列
        fid.update(real_tensor, real=True)
        fid.update(fake_tensor, real=False)

    except Exception as e:
        print(f"跳过坏图 {f}: {e}")

# 5. 计算并打印
print("正在计算结果...")
print(f"FID: {fid.compute().item()}")