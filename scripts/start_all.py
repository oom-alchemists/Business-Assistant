"""单窗口统一启动器：Ollama 检查 + Streamlit(8501) + FastAPI(8001, 2 workers)。

用法：
    conda python scripts/start_all.py

一个控制台窗口内启动全部服务；日志写入根目录 server_out.log / server_err.log；
按 q 退出并终止所有子进程。
"""
import msvcrt
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT_LOG = ROOT / "server_out.log"
ERR_LOG = ROOT / "server_err.log"
SKIP_PREFIXES = ("127.", "169.254.", "10.254.", "172.28.", "192.168.88.")
LAN_PREFIXES = ("192.168.", "10.", "172.")


def get_lan_ip():
    """获取局域网 IPv4：排除虚拟网卡/PPP 拨号/APIPA，优先 192.168/10.x 段。"""
    try:
        cands = []
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith(SKIP_PREFIXES):
                continue
            cands.append(ip)
        for ip in cands:
            if ip.startswith(LAN_PREFIXES):
                return ip
        return cands[0] if cands else "127.0.0.1"
    except Exception:
        return "127.0.0.1"


def is_up(url, timeout=2):
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def wait_up(url, name, secs=60):
    for _ in range(secs):
        if is_up(url):
            print(f"  [OK] {name} 就绪")
            return True
        time.sleep(1)
    print(f"  [WARN] {name} 未在 {secs}s 内就绪，请查看日志")
    return False


def spawn(name, args):
    print(f"  启动 {name} ...")
    stamp = f"\n===== {name} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
    with open(OUT_LOG, "a", encoding="utf-8") as fo, open(ERR_LOG, "a", encoding="utf-8") as fe:
        fo.write(stamp)
        fe.write(stamp)
        return subprocess.Popen(args, cwd=str(ROOT), stdout=fo, stderr=fe)


def stop(proc, name):
    if proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  已停止 {name}")


def check_ollama():
    if is_up("http://127.0.0.1:11434/api/tags"):
        print("  [OK] Ollama 运行中")
        return
    print("  启动 Ollama ...")
    subprocess.Popen(["ollama", "serve"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)


def main():
    ip = get_lan_ip()
    print("=" * 54)
    print("  Enterprise AI Assistant - 单窗口启动")
    print("=" * 54)
    print()
    print("[1/4] 检查 Ollama ...")
    check_ollama()

    print("[2/4] 启动 Streamlit (8501) ...")
    p_streamlit = spawn("Streamlit", [PY, "-m", "streamlit", "run", "app/webui.py",
                                      "--server.address", "0.0.0.0",
                                      "--server.port", "8501",
                                      "--server.headless", "true"])
    print("[3/4] 启动 FastAPI (8001, 2 workers) ...")
    p_fastapi = spawn("FastAPI", [PY, "-m", "uvicorn", "app.api:app",
                                  "--host", "0.0.0.0", "--port", "8001",
                                  "--workers", "2"])
    print("[4/4] 等待服务就绪 ...")
    wait_up("http://127.0.0.1:8501", "Streamlit 8501")
    wait_up("http://127.0.0.1:8001/api/health", "FastAPI 8001", secs=90)

    print()
    print("=" * 54)
    print("  Streamlit（单用户界面）")
    print(f"    本机    http://localhost:8501")
    print(f"    局域网  http://{ip}:8501")
    print()
    print("  FastAPI（并发界面，2 workers）")
    print(f"    本机    http://localhost:8001")
    print(f"    局域网  http://{ip}:8001")
    print(f"    API 文档 http://localhost:8001/docs")
    print("=" * 54)
    print()
    print("  日志文件: server_out.log / server_err.log")
    print("  按 q 退出（同时停止两个服务）")
    print()
    try:
        while True:
            if msvcrt.kbhit() and msvcrt.getch().lower() in (b"q",):
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print("正在停止服务 ...")
    stop(p_fastapi, "FastAPI")
    stop(p_streamlit, "Streamlit")
    print("已全部退出")


if __name__ == "__main__":
    main()
