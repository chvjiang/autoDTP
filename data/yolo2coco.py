import os
import json
import cv2
from tqdm import tqdm
from collections import defaultdict

def yolo2coco(image_dir, label_dir, output_json, class_names):
    """
    将 YOLO 格式标签转换为 COCO 格式的 JSON 文件。
    
    参数:
        image_dir (str): 图像文件目录（如 `images/train`）。
        label_dir (str): YOLO 标签文件目录（如 `labels/train`）。
        output_json (str): 输出的 COCO JSON 文件路径（如 `annotations/train.json`）。
        class_names (list): 类别名称列表（如 `['cat', 'dog']`）。
    """
    # 初始化 COCO 数据结构
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 添加类别信息
    for i, name in enumerate(class_names):
        coco["categories"].append({
            "id": i + 1,  # COCO 类别 ID 从 1 开始
            "name": name,
            "supercategory": "object"
        })

    # 统计标注 ID
    annotation_id = 1
    image_id = 1

    # 遍历图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        # 读取图像尺寸
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # 添加图像信息到 COCO
        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # 解析对应的 YOLO 标签文件
        label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
        if not os.path.exists(label_file):
            continue  # 跳过无标签的图像

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过无效行

            class_id, x_center, y_center, w, h = map(float, parts)
            class_id = int(class_id)

            # 将 YOLO 归一化坐标转换为 COCO 的绝对坐标
            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w = w * width
            h = h * height

            # 添加标注信息到 COCO
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,  # YOLO 类别 ID 从 0 开始，COCO 从 1 开始
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []  # 目标检测任务留空
            })
            annotation_id += 1

        image_id += 1

    # 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"转换完成！结果已保存到: {output_json}")

if __name__ == "__main__":
    # 示例用法（修改为你的实际路径和类别）
    IMAGE_DIR = "DATA/images/val"      # 图像目录
    LABEL_DIR = "DATA/labels/val"      # YOLO 标签目录
    OUTPUT_JSON = "DATA/val_coco.json"  # 输出 JSON 路径
    CLASS_NAMES = ["people"]             # 类别名称列表（与 YOLO 的 class_id 对应）

    yolo2coco(IMAGE_DIR, LABEL_DIR, OUTPUT_JSON, CLASS_NAMES)