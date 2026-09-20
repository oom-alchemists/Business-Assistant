"""下载项目所需的模型文件。

仓库不含模型权重与训练产物（体积超 GitHub 限制），队友首次拉取代码后运行本脚本获取。

用法:
    python scripts/download_models.py              # 下载全部
    python scripts/download_models.py --only bert  # 只下 BERT 预训练模型
    python scripts/download_models.py --only bge   # 只下 Embedding 模型
    python scripts/download_models.py --only ckpt  # 只下训练好的意图分类器权重
    python scripts/download_models.py --force      # 已存在也重新下载

网络说明:
    - bert / bge 来自 HuggingFace，国内可设镜像:  set HF_ENDPOINT=https://hf-mirror.com
    - ckpt 来自本仓库的 GitHub Release（约 390MB）；若报「连接被重置」，
      说明网络直连 GitHub 不通，请先配置代理再重试（详见 README「网络问题」章节）
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- HuggingFace 预训练模型 ----------
MODELS = {
    # 键名      HuggingFace 仓库 ID            本地目标目录
    "bert": ("google-bert/bert-base-chinese", ROOT / "models" / "bert-base-chinese"),
    "bge": ("BAAI/bge-small-zh-v1.5", ROOT / "models" / "bge-small-zh-v1.5"),
}

# ---------- 训练产物：从本仓库 Release 获取（无法从公网模型库下载）----------
GH_REPO = "oom-alchemists/Business-Assistant"
GH_TAG = "model-v1"
CKPT_URL = f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}/best.pt"
CKPT_DST = ROOT / "checkpoints" / "intent_bert" / "best.pt"

CHUNK = 1024 * 256          # 256KB 一块
MB = 1024 * 1024


def _fmt(n: int) -> str:
    return f"{n / MB:.1f} MB"


def download_hf(key: str, force: bool = False) -> bool:
    """从 HuggingFace 下载预训练模型。"""
    repo_id, local_dir = MODELS[key]
    local_dir.mkdir(parents=True, exist_ok=True)

    if not force and (local_dir / "model.safetensors").exists():
        print(f"[跳过] {key:5s} 已存在 -> {local_dir}")
        return True

    print(f"[下载] {key:5s} {repo_id}")
    print(f"       目标 -> {local_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [错误] 缺少依赖，请先执行: pip install huggingface_hub")
        return False

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            # 只取推理必需文件，跳过 tf/flax/onnx 等冗余权重
            ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "tf_model*", "flax_model*"],
        )
        print(f"  [完成] {key}")
        return True
    except Exception as e:
        print(f"  [失败] {key} -> {e}")
        print("         国内网络可尝试: set HF_ENDPOINT=https://hf-mirror.com")
        return False


def download_ckpt(force: bool = False) -> bool:
    """从 GitHub Release 下载训练好的分类器权重，支持断点续传。"""
    if not force and CKPT_DST.exists():
        print(f"[跳过] ckpt  已存在 -> {CKPT_DST}  ({_fmt(CKPT_DST.stat().st_size)})")
        return True

    CKPT_DST.parent.mkdir(parents=True, exist_ok=True)
    part = CKPT_DST.with_suffix(".pt.part")

    try:
        import requests
    except ImportError:
        print("  [错误] 缺少依赖，请先执行: pip install requests")
        return False

    # 断点续传：已下载部分则从断点继续
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}

    print(f"[下载] ckpt  {CKPT_URL}")
    if done:
        print(f"       检测到未完成文件，从 {_fmt(done)} 处续传 ...")
    else:
        print("       约 390MB，视网速可能需要几分钟，中途 Ctrl+C 后重跑可续传")

    try:
        with requests.get(CKPT_URL, headers=headers, stream=True, timeout=60) as r:
            if r.status_code == 404:
                print("  [失败] 未找到该 Release 文件（404）")
                print(f"         请确认维护者已发布 tag 「{GH_TAG}」并上传 best.pt")
                print(f"         或直接访问: https://github.com/{GH_REPO}/releases")
                return False
            r.raise_for_status()

            remaining = int(r.headers.get("content-length", 0))
            total = done + remaining if remaining else 0

            mode = "ab" if done and r.status_code == 206 else "wb"
            if mode == "wb":
                done = 0

            with open(part, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r       进度 {pct:3d}%   {_fmt(done)} / {_fmt(total)}", end="")
            print()

        part.replace(CKPT_DST)
        print(f"  [完成] ckpt  -> {CKPT_DST}  ({_fmt(CKPT_DST.stat().st_size)})")
        return True

    except Exception as e:
        print(f"\n  [失败] ckpt -> {e}")
        print("         若提示 Connection reset / 连接被重置，说明直连 GitHub 不通：")
        print('           git config --global "http.https://github.com.proxy" "http://127.0.0.1:7890"')
        print("         或用浏览器手动下载后放到:")
        print(f"           {CKPT_DST}")
        print(f"         下载地址: {CKPT_URL}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载项目所需的模型文件")
    parser.add_argument("--only", choices=list(MODELS) + ["ckpt", "all"], default="all",
                        help="只下载指定项（默认 all）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    results = {}
    if args.only in ("all", "bert", "bge"):
        for k in (["bert", "bge"] if args.only == "all" else [args.only]):
            results[k] = download_hf(k, force=args.force)
    if args.only in ("all", "ckpt"):
        results["ckpt"] = download_ckpt(force=args.force)

    print("\n" + "=" * 50)
    for k, ok in results.items():
        print(f"  {k:5s} : {'OK' if ok else 'FAILED'}")
    print("=" * 50)

    if not all(results.values()):
        print("\n部分文件下载失败。也可离线获取后手动放置：")
        for k, (_, d) in MODELS.items():
            print(f"  {k:5s} -> {d}")
        print(f"  ckpt  -> {CKPT_DST}")
        sys.exit(1)

    print("\n下一步:")
    print("  1. 构建向量索引: python scripts/build_index.py")
    print("  2. 启动服务:     双击 start_fastapi.bat")
    print("\n（若想自己训练分类器而非用 Release 权重，可执行：")
    print("  python scripts/train_intent.py --epochs 3 --bs 64 ）")


if __name__ == "__main__":
    main()
