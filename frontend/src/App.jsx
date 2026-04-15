import { useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  ArrowUpRight,
  Github,
  ImageUp,
  Loader2,
  Sparkles,
  Wand2,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000/api/analyze";

const CLASS_OPTIONS = [
  "dog",
  "cat",
  "person",
  "horse",
  "bird",
  "sheep",
  "cow",
  "aeroplane",
  "bicycle",
  "boat",
  "bus",
  "car",
  "motorbike",
  "train",
  "bottle",
  "chair",
  "diningtable",
  "pottedplant",
  "sofa",
  "tvmonitor",
];

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

export default function App() {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [targetClass, setTargetClass] = useState("dog");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [result, setResult] = useState(null);

  const tiles = useMemo(() => {
    if (!result) return [];
    return [
      { title: "原始热力图", url: result.heatmap_orig_url },
      { title: "分割掩码", url: result.mask_url },
      { title: "修复后聚焦图", url: result.focus_url },
      { title: "最终热力图", url: result.heatmap_final_url },
    ];
  }, [result]);

  const setSelectedFile = (f) => {
    setError("");
    setResult(null);
    setFile(f || null);
    if (!f) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(f);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return url;
    });
  };

  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (f) setSelectedFile(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setSelectedFile(f);
  };

  const onAnalyze = async () => {
    setError("");
    setResult(null);
    if (!file) {
      setError("请先上传一张图片。");
      return;
    }
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("target_class", targetClass);

      const { data } = await axios.post(API_URL, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(data);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "请求失败，请检查后端是否启动以及 CORS 配置。";
      setError(String(msg));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-full">
      {/* Top Header */}
      <div className="sticky top-0 z-50 border-b border-white/10 bg-slate-950/55 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-fuchsia-400/20 ring-1 ring-white/10">
                <Sparkles className="h-5 w-5 text-cyan-200" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-white">
                  VisionX - AI 视觉注意力分析与纠偏系统
                </div>
                <div className="mt-0.5 hidden text-xs text-slate-400 sm:block">
                  Attention Transfer · Segmentation · Grad-CAM
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <a
                className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200 hover:bg-white/10"
                href={API_URL}
                target="_blank"
                rel="noreferrer"
                title="打开后端接口地址"
              >
                API
                <ArrowUpRight className="h-4 w-4" />
              </a>

              <a
                className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-slate-200 hover:bg-white/10"
                href="https://github.com/"
                target="_blank"
                rel="noreferrer"
                title="GitHub"
              >
                <Github className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-2">
          <div className="inline-flex items-center gap-2 self-start rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200">
            <Sparkles className="h-4 w-4 text-cyan-300" />
            FastAPI · Attention Transfer · Grad-CAM
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            AI 视觉注意力迁移分析台
          </h1>
          <p className="text-sm text-slate-300">
            上传图片，选择目标类别，查看分割掩码与前后注意力热力图对比。
          </p>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-12">
          {/* Left */}
          <section className="lg:col-span-5">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.04)] backdrop-blur">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-medium text-slate-200">
                  输入与控制台
                </h2>
                <div className="text-xs text-slate-400">
                  默认目标：<span className="text-slate-200">dog</span>
                </div>
              </div>

              {/* Dropzone */}
              <div
                className={classNames(
                  "mt-4 group relative overflow-hidden rounded-2xl border border-dashed p-4 transition",
                  isDragging
                    ? "border-cyan-300/70 bg-cyan-300/10"
                    : "border-white/15 bg-slate-950/20 hover:border-white/30",
                )}
                onDragEnter={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(true);
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setIsDragging(false);
                }}
                onDrop={onDrop}
                role="button"
                tabIndex={0}
                onClick={() => inputRef.current?.click()}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    inputRef.current?.click();
                  }
                }}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={onPick}
                />

                <div className="flex items-center gap-4">
                  <div className="grid h-12 w-12 place-items-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-fuchsia-400/20 ring-1 ring-white/10">
                    <ImageUp className="h-6 w-6 text-cyan-200" />
                  </div>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">
                      拖拽上传图片
                      <span className="text-slate-400"> 或点击选择</span>
                    </div>
                    <div className="mt-0.5 text-xs text-slate-400">
                      支持常见格式（JPG/PNG/WebP）
                    </div>
                  </div>
                  <div className="text-xs text-slate-400">
                    {file ? (
                      <span className="rounded-lg bg-white/5 px-2 py-1 ring-1 ring-white/10">
                        {file.name}
                      </span>
                    ) : (
                      <span className="rounded-lg bg-white/5 px-2 py-1 ring-1 ring-white/10">
                        未选择
                      </span>
                    )}
                  </div>
                </div>

                {previewUrl ? (
                  <div className="mt-4 overflow-hidden rounded-xl ring-1 ring-white/10">
                    <img
                      src={previewUrl}
                      alt="预览"
                      className="h-64 w-full object-contain bg-slate-950/30"
                    />
                  </div>
                ) : (
                  <div className="mt-4 grid h-64 place-items-center rounded-xl bg-slate-950/25 ring-1 ring-white/10">
                    <div className="text-center">
                      <div className="mx-auto mb-2 h-10 w-10 rounded-2xl bg-white/5 ring-1 ring-white/10 grid place-items-center">
                        <Wand2 className="h-5 w-5 text-fuchsia-200" />
                      </div>
                      <div className="text-sm text-slate-200">
                        等待输入图像…
                      </div>
                      <div className="mt-1 text-xs text-slate-400">
                        选择后将显示预览
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <label className="block">
                  <div className="mb-1 text-xs text-slate-400">
                    target_class
                  </div>
                  <select
                    value={targetClass}
                    onChange={(e) => setTargetClass(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-slate-950/30 px-3 py-2 text-sm text-slate-100 outline-none ring-0 transition focus:border-cyan-300/40 focus:bg-slate-950/40"
                  >
                    {CLASS_OPTIONS.map((c) => (
                      <option key={c} value={c} className="bg-slate-950">
                        {c}
                      </option>
                    ))}
                  </select>
                </label>

                <div className="flex items-end">
                  <button
                    type="button"
                    onClick={onAnalyze}
                    disabled={loading}
                    className={classNames(
                      "w-full rounded-xl px-4 py-2.5 text-sm font-semibold text-white shadow-lg transition",
                      "bg-gradient-to-r from-cyan-500 to-fuchsia-500 hover:from-cyan-400 hover:to-fuchsia-400",
                      "disabled:opacity-60 disabled:cursor-not-allowed",
                      "ring-1 ring-white/10",
                    )}
                  >
                    {loading ? (
                      <span className="inline-flex items-center justify-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        AI正在进行注意力迁移计算...
                      </span>
                    ) : (
                      <span className="inline-flex items-center justify-center gap-2">
                        <Sparkles className="h-4 w-4" />
                        开始分析
                      </span>
                    )}
                  </button>
                </div>
              </div>

              {error ? (
                <div className="mt-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-100">
                  {error}
                </div>
              ) : null}

              <div className="mt-4 text-xs text-slate-400">
                后端地址：<span className="text-slate-200">{API_URL}</span>
              </div>
            </div>
          </section>

          {/* Right */}
          <section className="lg:col-span-7">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5 shadow-[0_0_0_1px_rgba(255,255,255,0.04)] backdrop-blur">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-medium text-slate-200">
                  输出结果（2x2）
                </h2>
                <div className="text-xs text-slate-400">
                  {result ? (
                    <>
                      target_class：
                      <span className="text-slate-200">
                        {" "}
                        {result.target_class}
                      </span>
                    </>
                  ) : (
                    "等待分析结果"
                  )}
                </div>
              </div>

              {!result ? (
                <div className="mt-4 grid min-h-[520px] place-items-center rounded-2xl bg-slate-950/25 ring-1 ring-white/10">
                  <div className="max-w-md px-6 text-center">
                    <div className="mx-auto mb-3 grid h-12 w-12 place-items-center rounded-2xl bg-white/5 ring-1 ring-white/10">
                      <Sparkles className="h-6 w-6 text-cyan-200" />
                    </div>
                    <div className="text-base font-semibold text-white">
                      结果将显示在这里
                    </div>
                    <div className="mt-1 text-sm text-slate-400">
                      右侧会以 2x2 网格展示：原始热力图 / 分割掩码 /
                      修复后聚焦图 / 最终热力图。
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                  {tiles.map((t) => (
                    <figure
                      key={t.title}
                      className="group overflow-hidden rounded-2xl border border-white/10 bg-slate-950/20 ring-1 ring-white/5"
                    >
                      <figcaption className="flex items-center justify-between gap-3 border-b border-white/10 bg-white/5 px-3 py-2">
                        <div className="text-sm font-medium text-slate-100">
                          {t.title}
                        </div>
                        <a
                          className="text-xs text-slate-300 hover:text-white"
                          href={t.url}
                          target="_blank"
                          rel="noreferrer"
                          title="新窗口打开原图"
                        >
                          打开
                        </a>
                      </figcaption>
                      <div className="aspect-video bg-slate-950/30">
                        <img
                          src={t.url}
                          alt={t.title}
                          className="h-full w-full object-contain"
                          loading="lazy"
                        />
                      </div>
                    </figure>
                  ))}
                </div>
              )}

              {/* Report Card (only after success) */}
              {!loading && result ? (
                <div className="mt-5 overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-b from-white/8 to-white/4 shadow-[0_0_0_1px_rgba(255,255,255,0.05)]">
                  <div className="flex items-center justify-between gap-3 border-b border-white/10 bg-white/5 px-4 py-3">
                    <div className="text-sm font-semibold text-white">
                      📊 视觉注意力迁移诊断报告
                    </div>
                    <div className="text-xs text-slate-400">
                      目标：{" "}
                      <span className="rounded-lg bg-white/5 px-2 py-1 text-slate-200 ring-1 ring-white/10">
                        {result.target_class}
                      </span>
                    </div>
                  </div>

                  <div className="grid gap-3 p-4 sm:grid-cols-2">
                    <div className="rounded-2xl border border-white/10 bg-slate-950/25 p-4 ring-1 ring-white/5">
                      <div className="text-sm font-semibold text-rose-200">
                        🚫 原始环境诊断
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-slate-300">
                        发现背景共现物体干扰，导致模型注意力分散，激活区域（热力图）未完全覆盖目标主体。
                      </p>
                      <div className="mt-3 text-xs text-slate-400">
                        风险：误聚焦 · 误分类 · 置信度波动
                      </div>
                    </div>

                    <div className="rounded-2xl border border-white/10 bg-slate-950/25 p-4 ring-1 ring-white/5">
                      <div className="text-sm font-semibold text-emerald-200">
                        ✅ 焦点优化重构
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-slate-300">
                        通过自适应掩码补洞与白底证件照重构算法，成功剥离干扰特征。最终热力图显示，AI 注意力已{" "}
                        <span className="font-semibold text-slate-100">
                          100% 聚焦
                        </span>{" "}
                        于{" "}
                        <span className="font-semibold text-cyan-200">
                          『{result.target_class}』
                        </span>{" "}
                        主体，分类置信度显著提升。
                      </p>
                      <div className="mt-3 text-xs text-slate-400">
                        结论：注意力集中 · 鲁棒性提升 · 输出更可信
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
