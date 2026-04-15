from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import OUTPUT_DIR, STATIC_DIR, settings
from app.services.vision import load_models, load_imagenet_labels


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期：
    - 启动时加载模型并创建输出目录
    - 关闭时如有需要可在此释放资源
    """
    # 确保静态目录存在
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载模型和标签，只在启动时加载一次
    seg_model, cls_model, imagenet_labels = load_models()
    app.state.seg_model = seg_model
    app.state.cls_model = cls_model
    app.state.imagenet_labels = imagenet_labels
    app.state.static_dir = STATIC_DIR

    yield

    # 如果需要，可以在这里手动释放显存或关闭资源
    # del app.state.seg_model
    # del app.state.cls_model


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# 静态文件挂载：/static 对应到本地 STATIC_DIR
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# 注册 API 路由
app.include_router(api_router)


@app.get("/", tags=["root"])
async def read_root():
    return {"message": "AI Vision FastAPI backend is running."}

