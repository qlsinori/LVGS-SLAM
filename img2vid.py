import cv2
import glob
import os

# 图片所在目录
img_dir = "/qls/code/neurad-studio/outputs/unnamed/splatad/2025-11-19_031526/fid/pred_rgb/60600"     # 改成你的图片目录

# 输出视频文件名
output_video = "output.mp4"

# 视频帧率
fps = 10                 # 根据需要修改

def main():
    # 匹配类似 000000_0.png 的文件
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    if len(img_paths) == 0:
        print("没有找到 PNG 图片，请检查路径")
        return

    # 读取第一张，确定宽高
    first_img = cv2.imread(img_paths[0])
    height, width, _ = first_img.shape

    # 定义视频编解码器与输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"跳过无法读取的图片：{path}")
            continue
        video_writer.write(img)

    video_writer.release()
    print(f"视频已生成：{output_video}")

if __name__ == "__main__":
    main()
