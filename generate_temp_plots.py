import os
import shutil
from PIL import Image, ImageDraw, ImageFont

def process_images(source_dir, target_dir, suffix):
    """
    读取源目录中的图片，添加标题后保存到目标目录，并按要求重命名。
    """
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} not found.")
        return

    os.makedirs(target_dir, exist_ok=True)
    
    # 定义文件名映射关系
    file_map = {
        "confusion_matrix.png": f"confusion_matrix_{suffix}.png",
        "confusion_matrix_normalized.png": f"confusion_matrix_normalized_{suffix}.png",
        "F1_curve.png": f"F1_curve_{suffix}.png",
        "P_curve.png": f"P_curve_{suffix}.png",
        "PR_curve.png": f"PR_curve_{suffix}.png",
        "R_curve.png": f"R_curve_{suffix}.png",
        "val_batch0_labels.jpg": f"val_batch0_labels_{suffix}.jpg",
        "val_batch0_pred.jpg": f"val_batch0_pred_{suffix}.jpg",
        "val_batch1_labels.jpg": f"val_batch1_labels_{suffix}.jpg",
        "val_batch1_pred.jpg": f"val_batch1_pred_{suffix}.jpg",
        "val_batch2_labels.jpg": f"val_batch2_labels_{suffix}.jpg",
        "val_batch2_pred.jpg": f"val_batch2_pred_{suffix}.jpg",
    }

    for src_name, target_name in file_map.items():
        src_path = os.path.join(source_dir, src_name)
        if os.path.exists(src_path):
            try:
                # 使用 PIL 打开图片并添加标题
                img = Image.open(src_path)
                
                # 简单的图片处理：在顶部添加一个白边并写入标题（模拟 plt.title）
                title_text = target_name.split('.')[0]
                
                # 创建新图，高度增加 50 像素用于放置标题
                new_img = Image.new('RGB', (img.width, img.height + 60), (255, 255, 255))
                new_img.paste(img, (0, 60))
                
                draw = ImageDraw.Draw(new_img)
                # 尝试加载默认字体，如果失败则使用基础字体
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 30)
                except:
                    font = ImageFont.load_default()
                
                # 计算文字位置（居中）
                bbox = draw.textbbox((0, 0), title_text, font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(((img.width - w) / 2, (60 - h) / 2), title_text, fill=(0, 0, 0), font=font)
                
                # 保存到 temp 文件夹
                target_path = os.path.join(target_dir, target_name)
                new_img.save(target_path)
                print(f"Processed: {src_name} -> {target_name}")
            except Exception as e:
                print(f"Error processing {src_name}: {e}")
                # 如果处理失败，直接复制
                shutil.copy(src_path, os.path.join(target_dir, target_name))
        else:
            print(f"Skip: {src_name} not found in {source_dir}")

def main():
    # 基础目录（假设在项目根目录下运行）
    base_dir = os.getcwd()
    temp_dir = os.path.join(base_dir, "temp")
    
    # 处理 Baseline 模型图片
    print("\n>>> Processing Baseline images...")
    baseline_src = os.path.join(base_dir, "runs/baseline_test")
    # 如果 runs 下没有，尝试从附件解压的路径读取（用于演示）
    if not os.path.exists(baseline_src):
        baseline_src = "/home/ubuntu/paper_req/附件2/baseline_test"
    process_images(baseline_src, temp_dir, "baseline")
    
    # 处理 Improved 模型图片
    print("\n>>> Processing Improved images...")
    improved_src = os.path.join(base_dir, "runs/improved_test")
    if not os.path.exists(improved_src):
        improved_src = "/home/ubuntu/paper_req/附件2/improved_test"
    process_images(improved_src, temp_dir, "improved")
    
    print(f"\nDone! All images are saved in: {temp_dir}")

if __name__ == "__main__":
    main()
