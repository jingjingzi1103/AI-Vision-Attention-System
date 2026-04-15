## VisionX - AI 视觉注意力分析与纠偏系统

基于 **FastAPI + PyTorch + Grad-CAM + React + TailwindCSS** 的前后端分离小项目，用语义分割结果引导分类模型“重新聚焦”，并可视化前后注意力热力图变化，适合作为课程作业或 AI 产品 Demo。

---

### ✨ 核心功能

- **目标显著性重构**
  - 使用 `DeepLabV3-ResNet101` 对输入图像做语义分割，支持 20 类 COCO 子集（cat、dog、person、car 等）。
  - 根据目标类别生成掩码，自动补洞、裁剪并重构为“白底证件照风格”图像，强化主体信息。

- **注意力可视化与对比**
  - 使用 `ResNet101` + Grad-CAM，对 **原始图像** 与 **聚焦后图像** 生成注意力热力图。
  - 通过前后对比展示：模型是否从背景干扰转而聚焦到真实目标。

- **可视化交互控制台（前端）**
  - 左侧：拖拽上传 / 点击选择图片 + 目标类别选择（`target_class`，默认 `dog`）+ 一键“开始分析”按钮。
  - 右侧：以 2×2 网格展示：
    - 原始热力图
    - 分割掩码
    - 修复后聚焦图
    - 最终热力图
  - 下方：**📊 视觉注意力迁移诊断报告** 卡片，自动生成原始环境诊断与焦点优化说明，整体风格对标大型互联网公司 AI 控制台。

---

### 🧱 项目结构

```text
d:\homework1
├─ app/                    # FastAPI 后端
│  ├─ main.py              # 应用入口（挂载路由、静态文件、CORS、生命周期加载模型）
│  ├─ api/
│  │  └─ routes.py         # /api/analyze 接口定义
│  ├─ core/
│  │  └─ config.py         # 全局配置（静态目录、输出目录、CORS 设置）
│  └─ services/
│     └─ vision.py         # 模型加载与推理逻辑（分割、补洞、Grad-CAM 等）
│
├─ app_v3.py               # 早期 Gradio 版 Demo（可选运行）
├─ homework.py             # 实验脚本版（批量生成结果图片）
├─ image/                  # 示例图片
│   ├─ image01.jpg
│   └─ image02.jpg
├─ frontend/               # React + Vite + TailwindCSS 前端
│  ├─ src/
│  │  ├─ main.jsx          # React 入口
│  │  ├─ App.jsx           # VisionX 控制台 UI
│  │  └─ index.css         # Tailwind 引导 + 全局暗色背景
│  ├─ tailwind.config.js   # Tailwind 配置
│  └─ package.json         # 前端依赖与脚本
└─ requirements.txt        # 后端 Python 依赖（见下）
```

---

### ⚙️ 后端环境准备（FastAPI + PyTorch）

1. **创建虚拟环境并安装依赖**

在项目根目录 `d:\homework1`：

```bash
pip install -r requirements.txt
```

> 如遇到 `torch` 或 `torchvision` 安装问题，可根据自己机器的 CUDA / CPU 情况，改用官方命令重新安装：
> 参考 `https://pytorch.org/` 给出的安装指令。

2. **启动 FastAPI 后端**

```bash
uvicorn app.main:app --reload
```

启动成功后：

- 健康检查：`GET http://127.0.0.1:8000/`
- 核心接口：`POST http://127.0.0.1:8000/api/analyze`

---

### 📡 接口说明：`POST /api/analyze`

- **URL**

```text
POST http://127.0.0.1:8000/api/analyze
```

- **请求方式**

`multipart/form-data`

- **请求参数**

| 字段名        | 类型        | 位置 | 必填 | 说明                                      |
| ------------- | ----------- | ---- | ---- | ----------------------------------------- |
| `file`        | UploadFile  | body | 是   | 上传的图片文件（JPG/PNG/WebP 等）        |
| `target_class`| string      | body | 是   | 目标类别，如 `dog`、`cat`、`person` 等   |

- **支持的目标类别**（与 COCO 子集保持一致）：

```text
[
  "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
  "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
  "motorbike", "person", "pottedplant", "sheep", "sofa",
  "train", "tvmonitor"
]
```

- **返回示例**

```json
{
  "target_class": "dog",
  "mask_url": "http://127.0.0.1:8000/static/outputs/xxxx_mask.png",
  "heatmap_orig_url": "http://127.0.0.1:8000/static/outputs/xxxx_heatmap_orig.png",
  "focus_url": "http://127.0.0.1:8000/static/outputs/xxxx_focus.png",
  "heatmap_final_url": "http://127.0.0.1:8000/static/outputs/xxxx_heatmap_final.png"
}
```

- **说明**
  - 所有图片均保存到后端的 `static/outputs/` 目录中。
  - FastAPI 已挂载静态目录 `/static`，可直接通过返回的 URL 在浏览器访问。

---

### 🖥 前端运行（React + Vite + TailwindCSS）

1. **安装依赖**

```bash
cd frontend
npm install
```

2. **启动开发服务器**

```bash
npm run dev
```

默认会在 `http://127.0.0.1:5173/`（或控制台显示的端口）启动前端。

3. **前端主要技术点**

- 使用 `axios` 调用后端 `POST /api/analyze` 接口，并通过 `FormData` 上传图片与 `target_class`。
- TailwindCSS 风格：深色科技感、带渐变背景与玻璃拟态卡片。
- 布局结构：
  - 左侧：上传/预览/参数选择/按钮
  - 右侧：2×2 结果图片网格 + 诊断报告卡片

---

### 🧠 核心算法说明（简要）

- **语义分割：** 使用 `torchvision.models.segmentation.deeplabv3_resnet101` 得到每个像素的类别，构造目标掩码。
- **掩码补洞与证件照化：**
  - 膨胀 + 高斯模糊掩码，填补掩码漏洞。
  - 将背景替换为模糊/白底背景，裁剪出目标区域并居中，增强对比度。
- **注意力可视化：** 使用 `pytorch-grad-cam` 在 ResNet 的 `layer4[-1]` 上做 Grad-CAM，生成伪彩色热力图叠加在图像上。
- **实验脚本：** `homework.py` 支持对多张图像/多类别进行批量实验，对比不同背景场景下的 Grad-CAM。

---

### ✅ CORS 与静态资源

- 在 `app/main.py` 中已配置：
  - `CORSMiddleware`，允许来自任意源（`*`）的跨域请求，方便前后端本地联调。
  - 静态目录挂载：
    - 本地路径：`app/core/config.py` 中的 `STATIC_DIR` 和 `OUTPUT_DIR`
    - 访问路径：`/static/...`

如需在生产环境收紧安全策略，可将 `cors_allow_origins` 改为具体的前端域名。

---

### 📦 requirements.txt（后端依赖）

项目根目录下的 `requirements.txt` 推荐内容如下（已在仓库中提供）：

```text
fastapi
uvicorn[standard]
torch
torchvision
pillow
numpy
requests
pytorch-grad-cam
matplotlib
```

如需完全可复现的环境，可以在本地通过 `pip freeze > requirements-lock.txt` 额外生成一份锁定版本文件。

---

### 🔍 可选：保留的 Gradio Demo 与实验脚本

- `app_v3.py`：
  - 使用 Gradio 搭建的交互式 Web Demo，逻辑与当前 FastAPI 版本相近。
  - 可单独运行：
    - `python app_v3.py`
- `homework.py`：
  - 用于课程作业实验的脚本版，自动对多张图片、多种 `segmentation_target` 生成完整的 Grad-CAM 对比图，输出到 `results/` 目录。

---

### 📎 License & 说明

- 本项目主要用于技术演示，不建议直接用于生产环境。
- 如需商用，请根据业务需求补充：
  - 模型版本管理与推理服务化（如 GPU 部署）
  - 用户鉴权与限流
  - 日志与监控

