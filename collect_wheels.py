# coding=gbk
import os
import sys
import subprocess
from pathlib import Path

def download_deps():
    # 项目根目录
    root = Path(__file__).parent
    # 依赖清单路径
    req_file = root / "requirements.txt"
    # 依赖保存目录（自动创建）
    deps_dir = root / "deps" / "wheels"
    deps_dir.mkdir(parents=True, exist_ok=True)

    # 确定当前平台的 wheel 标签（适配 Windows/macOS/Linux）
    platform_tag = ""
    if sys.platform.startswith("win"):
        # Windows x64
        platform_tag = "win_amd64"
    elif sys.platform.startswith("darwin"):
        # macOS Intel/ARM 通用
        platform_tag = "macosx_10_9_universal2"
    elif sys.platform.startswith("linux"):
        # Linux x64
        platform_tag = "manylinux2014_x86_64"

    # 调用 pip download 下载依赖
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-r", str(req_file),
        "-d", str(deps_dir),
        "--platform", platform_tag,  # 指定平台
        "--only-binary=:all:",       # 只下载二进制 wheel 包
        "--no-deps"                  # 不下载依赖的依赖（避免重复）
    ]

    try:
        subprocess.check_call(cmd)
        print(f"依赖已下载到 {deps_dir}")
    except subprocess.CalledProcessError as e:
        print(f"下载依赖失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    download_deps()