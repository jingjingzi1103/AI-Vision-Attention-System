import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import OUTPUT_DIR
from app.services.vision import COCO_CLASSES, run_inference_pipeline


router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze")
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="待分析的图片文件"),
    target_class: str = Form(..., description="目标类别（如 cat / dog / person 等）"),
):
    # 1. 检查类别是否在支持的 20 类列表中
    if target_class not in COCO_CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的类别: {target_class}。支持的类别有: {', '.join(COCO_CLASSES[1:])}",
        )

    # 2. 读取图片为 PIL
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无法解析上传图片，请确认文件格式是否正确。",
        )

    # 3. 从 app.state 中取出模型
    app = request.app
    seg_model = getattr(app.state, "seg_model", None)
    cls_model = getattr(app.state, "cls_model", None)

    if seg_model is None or cls_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型尚未加载完成，请稍后重试。",
        )

    # 4. 执行推理流程
    try:
        mask_pil, heatmap_orig, img_final, heatmap_final = run_inference_pipeline(
            img, target_class, seg_model, cls_model
        )
    except ValueError as e:
        # 业务级异常（如未检测到目标）
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"服务内部错误: {e}",
        )

    # 5. 保存图片到 static/outputs/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex
    mask_path = OUTPUT_DIR / f"{uid}_mask.png"
    heatmap_orig_path = OUTPUT_DIR / f"{uid}_heatmap_orig.png"
    focus_path = OUTPUT_DIR / f"{uid}_focus.png"
    heatmap_final_path = OUTPUT_DIR / f"{uid}_heatmap_final.png"

    mask_pil.save(mask_path)
    heatmap_orig.save(heatmap_orig_path)
    img_final.save(focus_path)
    heatmap_final.save(heatmap_final_path)

    # 6. 构造可被前端访问的完整 URL
    def build_url(p: Path) -> str:
        # /static/outputs/xxx.png
        static_root: Path = Path(app.state.static_dir)
        static_relative = p.relative_to(static_root)
        url = request.url_for("static", path=str(static_relative).replace("\\", "/"))
        return str(url)

    try:
        mask_url = build_url(mask_path)
        heatmap_orig_url = build_url(heatmap_orig_path)
        focus_url = build_url(focus_path)
        heatmap_final_url = build_url(heatmap_final_path)
    except Exception:
        # 回退：直接拼接 /static/outputs，相对简单但依赖部署路径
        base = str(request.base_url).rstrip("/")
        rel_root = "/static/outputs"
        mask_url = f"{base}{rel_root}/{mask_path.name}"
        heatmap_orig_url = f"{base}{rel_root}/{heatmap_orig_path.name}"
        focus_url = f"{base}{rel_root}/{focus_path.name}"
        heatmap_final_url = f"{base}{rel_root}/{heatmap_final_path.name}"

    return JSONResponse(
        {
            "target_class": target_class,
            "mask_url": mask_url,
            "heatmap_orig_url": heatmap_orig_url,
            "focus_url": focus_url,
            "heatmap_final_url": heatmap_final_url,
        }
    )
