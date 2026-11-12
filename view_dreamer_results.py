"""Open the existing Dreamer training results in TensorBoard."""

import argparse
import importlib.util
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT = Path(r"D:\USTB\Code\dreamer-TensorFlow 2")


def main():
    parser = argparse.ArgumentParser(description="打开 Dreamer 的 TensorBoard 训练结果")
    parser.add_argument("--logdir", type=Path, default=PROJECT, help="训练日志目录")
    args = parser.parse_args()
    logdir = args.logdir.expanduser().resolve()
    if not logdir.is_dir() or not any(logdir.rglob("events.out.tfevents.*")):
        sys.exit(f"这个目录里没有 TensorBoard 日志：{logdir}")

    executable = shutil.which("tensorboard")
    if importlib.util.find_spec("tensorboard") is not None:
        command = [sys.executable, "-m", "tensorboard.main"]
    elif executable:
        command = [executable]
    else:
        sys.exit("当前 Python 环境没有 TensorBoard。请激活训练 Dreamer 时用的环境，或在当前环境运行 python -m pip install tensorboard。")

    host = "127.0.0.1"
    with socket.socket() as listener:
        listener.bind((host, 0))
        port = listener.getsockname()[1]
    url = f"http://{host}:{port}/"
    print(f"训练日志：{logdir}\n浏览器地址：{url}\n按 Ctrl+C 关闭。", flush=True)
    with subprocess.Popen(command + ["--logdir", str(logdir), "--host", host, "--port", str(port)]) as process:
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    sys.exit(f"TensorBoard 启动失败（退出码 {process.returncode}），请查看上方日志。")
                try:
                    with urlopen(url, timeout=1):
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                sys.exit("TensorBoard 启动超时，请查看上方日志。")
            if not webbrowser.open(url):
                print(f"请在浏览器手动打开 {url}", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    sys.exit(main())
