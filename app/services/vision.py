import traceback
from typing import List, Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageEnhance, ImageFilter
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models
import torchvision.transforms as T


def load_imagenet_labels() -> List[str]:
    try:
        resp = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.text.split("\n")
    except Exception:
        # 保证长度为 1000，避免索引报错
        return ["Label Error"] * 1000


COCO_CLASSES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def load_models():
    """在应用启动时调用，一次性加载模型和标签。"""
    print(">>> 正在初始化模型 (DeepLabV3-ResNet101 / ResNet101)...")
    seg_model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT").eval()
    cls_model = models.resnet101(weights="DEFAULT").eval()
    imagenet_labels = load_imagenet_labels()
    print(">>> 模型与标签加载完成。")
    return seg_model, cls_model, imagenet_labels


def preprocess_cls(img: Image.Image) -> torch.Tensor:
    t = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return t(img).unsqueeze(0)


def preprocess_seg(img: Image.Image) -> torch.Tensor:
    t = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return t(img).unsqueeze(0)


def generate_cam_resized(
    model: torch.nn.Module,
    img_pil: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    """生成缩放到 target_size 的 Grad-CAM 热力图 PIL 图像。"""
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = preprocess_cls(img_pil)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    img_small = img_pil.resize((224, 224))
    rgb_img = np.float32(img_small) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    vis_pil = Image.fromarray(visualization).resize(
        target_size, resample=Image.BICUBIC
    )
    return vis_pil


def fix_holes_ultimate(
    img: Image.Image,
    mask_pil: Image.Image,
    padding: int = 20,
) -> Image.Image:
    """从 app_v3.py 迁移过来的“强力补洞 + 证件照化”处理。"""
    mask_fat = mask_pil.filter(ImageFilter.MaxFilter(21))
    mask_soft = mask_fat.filter(ImageFilter.GaussianBlur(radius=5))

    patch_layer = img.filter(ImageFilter.GaussianBlur(radius=40))

    bg_white = Image.new("RGB", img.size, (255, 255, 255))
    filled_hole_img = Image.composite(patch_layer, bg_white, mask_soft)
    final_composite = Image.composite(img, filled_hole_img, mask_pil)

    mask_arr = np.array(mask_fat)
    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)
    if not np.any(rows) or not np.any(cols):
        return final_composite

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    w, h = img.size
    xmin = max(0, xmin - padding)
    xmax = min(w, xmax + padding)
    ymin = max(0, ymin - padding)
    ymax = min(h, ymax + padding)

    cropped = final_composite.crop((xmin, ymin, xmax, ymax))

    w_crop, h_crop = cropped.size
    max_side = max(w_crop, h_crop)
    square_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    offset_x = (max_side - w_crop) // 2
    offset_y = (max_side - h_crop) // 2
    square_img.paste(cropped, (offset_x, offset_y))

    enhancer = ImageEnhance.Contrast(square_img)
    square_img = enhancer.enhance(1.15)

    return square_img


def run_inference_pipeline(
    original_image: Image.Image,
    target_class: str,
    seg_model: torch.nn.Module,
    cls_model: torch.nn.Module,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    后端版本推理流程：
    返回 (mask_pil, heatmap_orig, img_final, heatmap_final) 四张 PIL 图。
    """
    if original_image is None:
        raise ValueError("original_image is None")

    try:
        original_image = original_image.convert("RGB")
        orig_size = original_image.size

        # 语义分割
        input_tensor = preprocess_seg(original_image)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"][0]
        predictions = output.argmax(0)

        try:
            idx = COCO_CLASSES.index(target_class)
        except ValueError:
            raise ValueError(f"类别不支持: {target_class}")

        mask_np = (predictions == idx).cpu().numpy().astype(np.uint8)
        mask_pil = Image.fromarray(mask_np * 255)
        if mask_pil.size != original_image.size:
            mask_pil = mask_pil.resize(original_image.size, resample=Image.NEAREST)

        if np.array(mask_pil).sum() == 0:
            raise ValueError(f"在图中未检测到目标: {target_class}")

        # 修复 + 聚焦
        img_final = fix_holes_ultimate(original_image, mask_pil, padding=20)

        # Grad-CAM
        heatmap_orig = generate_cam_resized(cls_model, original_image, orig_size)
        heatmap_final = generate_cam_resized(cls_model, img_final, img_final.size)

        return mask_pil, heatmap_orig, img_final, heatmap_final
    except Exception as e:
        traceback.print_exc()
        raise e

