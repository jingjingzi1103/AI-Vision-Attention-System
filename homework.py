import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import requests
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

print(">>> 步骤1/5: 正在加载预训练模型...")
try:
    segmentation_model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).eval()
    classification_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
except Exception as e:
    print(f"模型权重加载失败，可能由于网络问题或torchvision版本更新。错误: {e}")
    print("正在尝试使用旧的 'pretrained=True' 方法...")
    segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    classification_model = models.resnet50(pretrained=True).eval()
print(">>> 模型加载完毕！")

try:
    response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
    response.raise_for_status()
    imagenet_labels = response.text.split("\n")
except requests.RequestException as e:
    print(f"无法在线下载ImageNet标签: {e}")
    imagenet_labels = ["未知类别"] * 1000

COCO_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        return Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        return Image.open(image_path_or_url).convert("RGB")

def preprocess_for_classification(image_pil):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_pil).unsqueeze(0)

def preprocess_for_segmentation(image_pil):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image_pil).unsqueeze(0)

def classify_image(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    results = []
    for i in range(top5_prob.size(0)):
        category_name = imagenet_labels[top5_catid[i]]
        probability = top5_prob[i].item()
        results.append(f"  - {category_name:<25} (置信度: {probability:.4f})")
    return "\n".join(results)

def create_noise_background(size):
    noise = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(noise)

def create_color_background(size, color=(128, 128, 128)):
    return Image.new('RGB', size, color)

def crop_object(image_pil, mask_pil, padding=10):
    mask_np = np.array(mask_pil)
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return image_pil

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    width, height = image_pil.size
    xmin = max(0, xmin - padding)
    xmax = min(width, xmax + padding)
    ymin = max(0, ymin - padding)
    ymax = min(height, ymax + padding)

    cropped_image = image_pil.crop((xmin, ymin, xmax, ymax))
    return cropped_image

def visualize_and_save(original_image, mask_pil, masked_image_pil, target_class, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("1. Original Image", fontsize=14); plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_pil, cmap='gray')
    plt.title(f"2. Extracted Mask for '{target_class}'", fontsize=14); plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(masked_image_pil)
    plt.title("3. Composited Image (Target Only)", fontsize=14); plt.axis('off')
    
    plt.suptitle(f"Segmentation-Guided Classification Experiment: Focusing on '{target_class}'", fontsize=18, y=0.95)
    
    filename = f"result_{target_class}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"\n>>> 结果图已保存到: {filepath}")
    plt.show()

def run_full_experiment(image_path, segmentation_target, output_dir="results"):
    print("\n" + "="*80)
    print(f"🚀 最终实验: '{segmentation_target}' 的注意力热力图分析 (Grad-CAM)")
    print("="*80)
    
    original_pil = get_image(image_path)
    seg_input = preprocess_for_segmentation(original_pil)
    
    with torch.no_grad():
        output = segmentation_model(seg_input)['out'][0]
    output_predictions = output.argmax(0)
    
    try:
        target_index = COCO_CLASSES.index(segmentation_target.lower())
    except ValueError:
        print(f"错误: 不支持类别 {segmentation_target}"); return

    mask_np = (output_predictions == target_index).cpu().numpy().astype(np.uint8)
    if mask_np.shape != original_pil.size[::-1]:
         mask_pil = Image.fromarray(mask_np * 255).resize(original_pil.size, resample=Image.NEAREST)
    else:
         mask_pil = Image.fromarray(mask_np * 255)

    if np.array(mask_pil).sum() == 0:
        print(f"未检测到目标，跳过。"); return

    bg_black = Image.new('RGB', original_pil.size, (0, 0, 0))
    bg_noise = create_noise_background(original_pil.size)
    
    scenarios = [
        ("Original", original_pil),
        ("Black BG", Image.composite(original_pil, bg_black, mask_pil)),
        ("Noise BG", Image.composite(original_pil, bg_noise, mask_pil))
    ]
    
    target_layers = [classification_model.layer4[-1]]
    cam = GradCAM(model=classification_model, target_layers=target_layers)

    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Attention Map Analysis: Target = '{segmentation_target}'", fontsize=20, y=0.98)
    
    for idx, (title, img) in enumerate(scenarios):
        input_tensor = preprocess_for_classification(img)
        
        res_str = classify_image(classification_model, input_tensor)
        top1_line = res_str.strip().split('\n')[0]
        pred_name = top1_line.split('(')[0].replace('-', '').strip()
        
        targets = None 
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        img_resized = img.resize((224, 224))
        rgb_img = np.float32(img_resized) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img)
        plt.title(f"{title}\nPred: {pred_name}", fontsize=12)
        plt.axis('off')
        
        plt.subplot(2, 3, idx + 4)
        plt.imshow(visualization)
        plt.title(f"Attention Map", fontsize=12, color='red')
        plt.axis('off')

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, f"gradcam_{segmentation_target}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f">>> ✅ 热力图已生成: {save_path}")
    plt.show()

if __name__ == '__main__':
    image_url_1 = "image/image01.jpg"
    run_full_experiment(image_path=image_url_1, segmentation_target="cat")
    run_full_experiment(image_path=image_url_1, segmentation_target="dog")

    image_url_2 = "image/image02.jpg"
    run_full_experiment(image_path=image_url_2, segmentation_target="person")
    run_full_experiment(image_path=image_url_2, segmentation_target="horse")

    print("\n" + "*"*80); print("✅ 所有实验已完成！检查 'results' 文件夹查看所有生成的图像。"); print("*"*80)