"""下载项目所需的预训练模型。

模型权重不进 Git 仓库（体积超 GitHub 限制），队友首次拉取代码后运行本脚本获取。

用法:
    python scripts/download_models.py              # 下载全部
    python scripts/download_models.py --only bert  # 只下 BERT
    python scripts/download_models.py --only bge   # 只下 embedding

国内网络建议先设镜像（可选）:
    set HF_ENDPOINT=https://hf-mirror.com
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    # 键名      HuggingFace 仓库 ID            本地目标目录
    "bert": ("google-bert/bert-base-chinese", ROOT / "models" / "bert-base-chinese"),
    "bge": ("BAAI/bge-small-zh-v1.5", ROOT / "models" / "bge-small-zh-v1.5"),
}


def download(key: str) -> bool:
    repo_id, local_dir = MODELS[key]
    local_dir.mkdir(parents=True, exist_ok=True)

    # 已存在则跳过（判断标志文件）
    if (local_dir / "model.safetensors").exists():
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


def main():
    parser = argparse.ArgumentParser(description="下载项目预训练模型")
    parser.add_argument("--only", choices=list(MODELS) + ["all"], default="all",
                        help="只下载指定模型（默认 all）")
    args = parser.parse_args()

    keys = list(MODELS) if args.only == "all" else [args.only]
    results = {k: download(k) for k in keys}

    print("\n" + "=" * 46)
    for k, ok in results.items():
        print(f"  {k:5s} : {'OK' if ok else 'FAILED'}")
    print("=" * 46)

    if not all(results.values()):
        print("\n部分模型下载失败。若已离线拿到模型，可手动放到:")
        for k, (_, d) in MODELS.items():
            print(f"  {k:5s} -> {d}")
        sys.exit(1)

    print("\n下一步:")
    print("  1. 训练意图分类器: python scripts/train_intent.py --epochs 3 --bs 64")
    print("  2. 构建向量索引:   python scripts/build_index.py")
    print("  3. 启动服务:       双击 start_fastapi.bat")


if __name__ == "__main__":
    main()
