"""Provision the pinned UVR/RVC runtimes in a directory owned by this app."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import uuid
from pathlib import Path

from studio import DATA, MODEL, MODEL_HASH, ROOT, require_file, sha256, write_json

RVC_SOURCE = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_REVISION = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
UVR_URL = f"https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{MODEL}"
HUBERT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/0d7ebae452fb102c695b08f4e6f546be00603425/rmvpe.pt"
RMVPE_HASH = "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193"
HUBERT = {
    "config.json": "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf",
    "preprocessor_config.json": "7c1976a680fb7acc757cd36fb08eef878fa36c70b4c9d2d595df9c608bbbbf0e",
    "pytorch_model.bin": "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5",
}


def provision_file(url, path, expected, verify_only):
    if path.exists() or verify_only:
        if sha256(require_file(path)) != expected:
            raise ValueError(f"文件 SHA-256 不符，请核对并移走该文件后重试：{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{uuid.uuid4().hex}.download")
    print(f"下载 {path.name}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=90) as response, temp.open("xb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(temp) != expected:
            raise ValueError(f"下载内容的 SHA-256 不符：{path.name}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def verify_checkout(git, root, revision):
    top_level = subprocess.check_output([*git, "-C", root, "rev-parse", "--show-toplevel"],
                                        encoding="utf-8").strip()
    if Path(top_level).resolve() != root:
        raise ValueError(f"请使用独立的引擎工作目录：{root}")
    head = subprocess.check_output([*git, "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"引擎提交不匹配：{head}。请使用指定版本。")
    subprocess.run([*git, "-C", root, "diff", "--quiet", "HEAD", "--"], check=True)


def provision_rvc(root, verify_only):
    git = [shutil.which("git"), "-c", "core.hooksPath=NUL"]
    if not verify_only and (not root.exists() or set(item.name for item in root.iterdir()) <= {".git"}):
        if not (root / ".git").is_dir():
            subprocess.run([*git, "init", root], check=True)
        subprocess.run([*git, "-C", root, "fetch", "--depth", "1", RVC_SOURCE, RVC_REVISION], check=True)
        subprocess.run([*git, "-C", root, "checkout", "--detach", "FETCH_HEAD"], check=True)
    verify_checkout(git, root, RVC_REVISION)
    for name, digest in HUBERT.items():
        provision_file(f"{HUBERT_URL}/{name}", root / "assets" / "hubert_base" / name, digest, verify_only)


def register_ace(runtime, env):
    from ace_original import ACE_FILES, ACE_REVISION

    root, python = runtime / "ace-step", runtime / "ace-env" / "Scripts" / "python.exe"
    require_file(python)
    verify_checkout([shutil.which("git"), "-c", "core.hooksPath=NUL"], root, ACE_REVISION)
    for name, digest in ACE_FILES.items():
        provision_file("", root / "checkpoints" / name, digest, verify_only=True)
    subprocess.run([python, "-B", "-c",
                    "import torch; from acestep.handler import AceStepHandler; "
                    "from acestep.inference import generate_music; "
                    "assert torch.version.cuda is None and torch.version.hip is None"],
                   cwd=root, env={**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}, check=True)
    return {"ace_python": setting_path(python), "ace_root": setting_path(root)}


def setup(runtime, verify_only, ace_runtime, with_rmvpe=False):
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11) or platform.machine() != "AMD64":
        raise ValueError("引擎安装使用 Windows x64 的 Python 3.11。")
    for tool in ("git", "ffmpeg", "uv") if not verify_only else ("git", "ffmpeg"):
        if not shutil.which(tool):
            raise ValueError(f"找不到 {tool}，请安装并加入 PATH。")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(UV_CACHE_DIR=str(runtime / "uv-cache"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    python = runtime / "audio-env" / "Scripts" / "python.exe"
    rvc, models = runtime / "rvc", runtime / "uvr-models"
    if not verify_only:
        uv = shutil.which("uv")
        install_temp = runtime / "install-temp"
        install_temp.mkdir(parents=True, exist_ok=True)
        env.update(TEMP=str(install_temp), TMP=str(install_temp))
        if not python.is_file():
            subprocess.run([uv, "venv", "--python", sys.executable, python.parents[1]], env=env, check=True)
        subprocess.run([uv, "pip", "sync", "--python", python, "--strict", "--no-progress", "--link-mode", "copy",
                        "--index", "https://download.pytorch.org/whl/cpu",
                        "--default-index", "https://pypi.org/simple", ROOT / "engines-lock.txt"], env=env, check=True)
    require_file(python)
    provision_rvc(rvc, verify_only)
    provision_file(UVR_URL, models / MODEL, MODEL_HASH, verify_only)
    if with_rmvpe:
        provision_file(RMVPE_URL, rvc / "assets" / "rmvpe" / "rmvpe.pt", RMVPE_HASH, verify_only)
    subprocess.run([python, "-B", "-c", "from audio_separator.separator.architectures.mdx_separator "
                    "import MDXSeparator; import onnxruntime, torch; assert torch.version.cuda is None "
                    "and torch.version.hip is None; assert 'CPUExecutionProvider' in "
                    "onnxruntime.get_available_providers()"], env=env, check=True)
    subprocess.run([python, "-B", "-m", "infer.cli", "--help"], cwd=rvc, env=env, check=True,
                   stdout=subprocess.DEVNULL)
    config_path = DATA / "engines.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    config.update(audio_python=setting_path(python), uvr_dir=setting_path(models),
                  rvc_root=setting_path(rvc), rvc_python=setting_path(python))
    config.setdefault("uvr_backend", "auto")
    if with_rmvpe:
        config["f0_method"] = "rmvpe"
    else:
        config.setdefault("f0_method", "pm")
    if ace_runtime is not None:
        config.update(register_ace(ace_runtime, env))
    write_json(config_path, config)
    print(f"引擎配置已写入 {config_path}")


def setting_path(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DATA / "engines",
                        help="引擎目录。默认在应用目录下。")
    parser.add_argument("--register-existing", action="store_true",
                        help="核对现有运行时并写入配置。此模式不下载或安装。")
    parser.add_argument("--ace-existing", type=Path, help="核对并登记已有的 ace-env 和 ace-step。")
    parser.add_argument("--with-rmvpe", action="store_true", help="校验并启用实测使用的 RMVPE 音高模型。")
    args = parser.parse_args()
    try:
        setup(args.runtime.resolve(), args.register_existing,
              args.ace_existing.resolve() if args.ace_existing else None, args.with_rmvpe)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"引擎准备失败：{exc}\n")


if __name__ == "__main__":
    main()
