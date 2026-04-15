from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent

# 静态资源目录，例如 /static
STATIC_DIR: Path = BASE_DIR / "static"

# 输出图片目录，例如 /static/outputs
OUTPUT_DIR: Path = STATIC_DIR / "outputs"


class Settings:
    """应用配置。后续如果需要可以接入环境变量。"""

    app_name: str = "AI Vision Backend"
    cors_allow_origins: List[str] = ["*"]  # 如需限制可改为前端实际域名
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]


settings = Settings()

