"""Local, serial audio jobs for AI Singing Studio."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
from array import array
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    from winjob import ProcessTree

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "studio-data"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
MODEL_HASH = "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc"
MEDIA_MODES = {"separate", "cover", "stems", "remix"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def require_file(path: Path, suffix: str | None = None) -> Path:
    if not path.is_file() or path.stat().st_size == 0 or (suffix and path.suffix.lower() != suffix):
        raise ValueError(f"文件缺失、为空或格式不符：{path}")
    return path


@dataclass(frozen=True)
class Voice:
    name: str
    permission: str
    weights: str
    weights_hash: str
    index: str = ""
    index_hash: str = ""
    speaker: int = 0
    pitch: int = 0
    index_rate: float = 0.6

    @classmethod
    def register(cls, name, permission, weights, index="", speaker=0, pitch=0, index_rate=0.6):
        if not name.strip() or not permission.strip():
            raise ValueError("请填写角色名称和授权依据。")
        model = require_file(Path(weights).resolve(), ".pth")
        feature = require_file(Path(index).resolve(), ".index") if index else None
        voice = cls(name.strip(), permission.strip(), str(model), sha256(model),
                    str(feature) if feature else "", sha256(feature) if feature else "",
                    speaker, pitch, index_rate)
        voice.check_settings()
        return voice

    def check_settings(self):
        if not self.name.strip() or not self.permission.strip():
            raise ValueError("角色配置缺少名称或授权依据。")
        if type(self.speaker) is not int or self.speaker < 0:
            raise ValueError("说话人编号必须是非负整数。")
        if type(self.pitch) is not int or not -24 <= self.pitch <= 24:
            raise ValueError("移调范围为 -24 至 +24 个半音。")
        if not 0 <= self.index_rate <= 1:
            raise ValueError("索引比重范围为 0 至 1。")
        if self.index and self.index_rate and "trained" in self.index:
            raise ValueError("请选择已添加检索向量的索引。索引完整路径不能含 trained，上游 RVC 会替换这段路径。")

    def verify(self):
        self.check_settings()
        files = [(self.weights, self.weights_hash, ".pth")]
        if self.index:
            files.append((self.index, self.index_hash, ".index"))
        for filename, expected, suffix in files:
            path = require_file(Path(filename), suffix)
            if sha256(path) != expected:
                raise ValueError(f"角色文件已变化，请核对授权并重新登记：{path.name}")


@dataclass(frozen=True)
class Settings:
    audio_python: str = ""
    uvr_dir: str = ""
    rvc_root: str = ""
    rvc_python: str = ""
    uvr_backend: str = "cpu"
    f0_method: str = "pm"
    ace_python: str = ""
    ace_root: str = ""

    @classmethod
    def load(cls):
        config = DATA / "engines.json"
        if not config.exists():
            return cls()
        values = json.loads(config.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("引擎配置应为 JSON 对象。")
        for key in ("audio_python", "uvr_dir", "rvc_root", "rvc_python", "ace_python", "ace_root"):
            if values.get(key):
                values[key] = str((ROOT / values[key]).resolve())
        return cls(**values)


@dataclass(frozen=True)
class Request:
    mode: str
    source: Path | None = None
    backing: Path | None = None
    start: float = 0.0
    seconds: float | None = 30.0
    voice: Voice | None = None
    caption: str = ""
    lyrics: str = ""
    language: str = "unknown"
    vocal_gain: float | None = None
    backing_gain: float = 0.5
    formant_shift: int = 0
    source_title: str = ""
    source_voice: str = ""

    def validate(self):
        if self.mode not in MEDIA_MODES | {"demo", "original"}:
            raise ValueError("未知任务类型。")
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("起点请填 0 或更大的数字。0 表示从头开始。")
        if self.mode in MEDIA_MODES:
            if self.seconds is None:
                if self.start != 0:
                    raise ValueError("处理整首须从头开始。")
            elif self.mode == "remix":
                if not math.isfinite(self.seconds) or self.seconds <= 0:
                    raise ValueError("原成品时长无效。")
            elif not math.isfinite(self.seconds) or self.seconds < 1:
                raise ValueError("片段时长须至少为 1 秒。")
        elif self.seconds is None or not 1 <= self.seconds <= 30:
            raise ValueError("片段时长须为 1–30 秒。")
        if (self.vocal_gain is not None and not 0 < self.vocal_gain <= 1.5) or not 0 <= self.backing_gain <= 1.5:
            raise ValueError("人声增益须大于 0，且两条音轨增益均不得高于 150%。")
        if type(self.formant_shift) is not int or not -2 <= self.formant_shift <= 2:
            raise ValueError("音色厚薄范围为 -2 至 +2。")
        if self.mode == "original":
            if not 15 <= self.seconds <= 30 or not self.caption.strip() or not self.lyrics.strip():
                raise ValueError("原创片段为 15–30 秒，请填写曲风和歌词。")
            if self.language not in {"unknown", "zh", "en", "ja"} or self.voice is not None:
                raise ValueError("原创任务的语言设置无效。生成后可通过翻唱工作流使用角色音色。")
        if self.mode in MEDIA_MODES:
            if self.source is None:
                raise ValueError("请选择输入音频。")
            require_file(self.source)
        if self.mode == "stems" and self.backing is not None:
            require_file(self.backing)
        if self.mode == "remix":
            if self.backing is None or self.voice is not None or self.formant_shift or self.start:
                raise ValueError("快速混音只接受已有的人声和伴奏，以及新的音量。")
            require_file(self.backing)
        if self.mode == "cover" and self.voice is None:
            raise ValueError("请先选择已登记的角色模型。")
        if self.mode in {"demo", "separate"} and self.voice is not None:
            raise ValueError("此工作流不使用角色音色。")


class Cancelled(Exception):
    pass


@dataclass
class Job:
    request: Request
    settings: Settings
    folder: Path
    state: str = "queued"
    progress: int = 0
    detail: str = "等待"
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    metadata: dict = field(default_factory=dict)
    started_at: float | None = None
    stage_at: float | None = None

    @property
    def result(self):
        return self.folder / "result.wav"

    def tracks(self):
        if self.snapshot()[0] != "done":
            return {}
        mode = self.request.mode
        source_label = "原曲" if self.request.seconds is None else "原曲片段"
        tracks = {
            "cover": {"混音成品": "result.wav", "调整后人声": "timbre.wav", "AI 人声": "ai-vocal.wav",
                      "分离人声": "vocals.wav", "分离伴奏": "instrumental.wav", source_label: "input.wav"},
            "separate": {"分离人声": "vocals.wav", "分离伴奏": "instrumental.wav",
                         source_label: "input.wav"},
            "stems": {"混音成品": "result.wav", "调整后人声": "timbre.wav", "AI 人声": "ai-vocal.wav",
                      "原始导唱": "input.wav", "导入伴奏": "backing.wav"},
            "remix": {"混音成品": "result.wav", "混音人声": "input.wav", "混音伴奏": "backing.wav"},
            "original": {"歌曲草稿 · 歌词待听验": "result.wav"},
            "demo": {"测试音": "result.wav"},
        }[mode]
        return {label: self.folder / name for label, name in tracks.items()
                if (self.folder / name).is_file()}

    @property
    def source_title(self):
        return (self.metadata.get("source_title") or self.request.source_title
                or (self.request.source.stem if self.request.source else ""))

    @property
    def voice_title(self):
        return (self.metadata.get("voice") or self.request.source_voice
                or (self.request.voice.name if self.request.voice else ""))

    def mix_tracks(self):
        tracks = self.tracks()
        if "混音成品" not in tracks:
            return None
        vocal = next((tracks[label] for label in ("调整后人声", "AI 人声", "混音人声", "原始导唱")
                      if label in tracks), None)
        backing = next((tracks[label] for label in ("分离伴奏", "混音伴奏", "导入伴奏")
                        if label in tracks), None)
        return (vocal, backing) if vocal and backing else None

    def remix(self, vocal_gain, backing_gain):
        if (tracks := self.mix_tracks()) is None:
            raise ValueError("这个任务缺少可用于混音的人声或伴奏。")
        return Request("remix", *tracks, seconds=wav_info(self.result)[0],
                       vocal_gain=vocal_gain, backing_gain=backing_gain,
                       source_title=self.source_title or "未记录歌名", source_voice=self.voice_title)

    def stop(self):
        with self.lock:
            if self.state in {"queued", "running"}:
                self.cancel.set()
                self.state = "cancelling"
                self.detail = "正在取消"

    def snapshot(self):
        with self.lock:
            return self.state, self.progress, self.detail

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()

    def stage(self, progress, detail):
        with self.lock:
            self.check_cancel()
            self.progress, self.detail = progress, detail
            self.stage_at = time.monotonic()


def recent_jobs(data, limit=20):
    """Reopen completed local results; older records may not have a title or clip choice."""
    records = []
    for folder in (data / "jobs").glob("*"):
        if folder.is_symlink() or not folder.is_dir():
            continue
        try:
            result = folder / "result.wav"
            if not result.is_file() or not result.stat().st_size:
                continue
            record = folder / "result.json"
            records.append((record.stat().st_mtime_ns, record))
        except OSError:
            continue
    jobs = []
    for _, record in sorted(records, reverse=True):
        try:
            meta = json.loads(record.read_text(encoding="utf-8"))
            mode, duration = meta["mode"], meta["seconds"]
            if (mode not in MEDIA_MODES | {"demo", "original"} or type(duration) not in {int, float}
                    or not math.isfinite(duration) or duration <= 0):
                continue
            # Use the actual exported duration if the previous version did not record the requested range.
            seconds = None if mode in MEDIA_MODES and meta.get("whole_song") is True else duration
            for field_name in ("source_title", "voice"):
                if not isinstance(meta.get(field_name), str):
                    meta[field_name] = ""
            detail = f"{duration:.2f} 秒" + (" · 歌词待听验" if mode == "original" else "")
            jobs.append(Job(Request(mode, seconds=seconds), Settings(), record.parent,
                            state="done", progress=100, detail=detail, metadata=meta))
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            continue
        if len(jobs) >= limit:
            break
    return jobs


class JobQueue:
    def __init__(self, data: Path = DATA):
        self.data = data
        self.pending = queue.Queue()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def submit(self, request: Request, settings: Settings) -> Job:
        request.validate()
        folder = self.data / "jobs" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        job = Job(request, settings, folder)
        self.pending.put(job)
        return job

    def _work(self):
        while (job := self.pending.get()) is not None:
            execute(job)

    def close(self):
        self.pending.put(None)


def local_env(folder):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    cache = str((DATA / "cache" / "numba").resolve())
    if sys.platform == "win32":
        cache = ("\\\\?\\UNC\\" + cache[2:]) if cache.startswith("\\\\") else ("\\\\?\\" + cache)
    env.update(PYTHONUTF8="1", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HOME=str(folder / "hf"), TORCH_HOME=str(folder / "torch"),
               XDG_CACHE_HOME=str(folder / "cache"), TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
               NUMBA_CACHE_DIR=cache,
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return env


def run(job, args, cwd=None, timeout=900):
    job.check_cancel()
    args = [str(item) for item in args]
    with (job.folder / "job.log").open("ab", buffering=0) as log:
        log.write(("\n> " + subprocess.list2cmdline(args) + "\n").encode("utf-8"))
        with subprocess.Popen(args, cwd=cwd or job.folder, env=local_env(job.folder),
                              stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                              creationflags=NO_WINDOW) as proc:
            try:
                if sys.platform == "win32":
                    with ProcessTree() as tree:
                        tree.attach(proc)
                        wait_process(job, proc, args, timeout)
                else:
                    wait_process(job, proc, args, timeout)
            finally:
                if proc.poll() is None:
                    proc.kill()


def wait_process(job, proc, args, timeout):
    deadline = time.monotonic() + timeout
    while proc.poll() is None:
        if job.cancel.wait(0.1):
            raise Cancelled()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{Path(args[0]).name} 超过 {timeout} 秒。")
    job.check_cancel()
    if proc.returncode:
        raise RuntimeError(f"{Path(args[0]).name} 退出码 {proc.returncode}，详见 job.log。")


def ffmpeg(job, output, *args):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FileNotFoundError("找不到 FFmpeg。请安装后加入 PATH。")
    run(job, [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
              *args, "-map_metadata", "-1", "-map_chapters", "-1",
              "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", output],
        timeout=180)


def duration_args(seconds):
    return ("-t", str(seconds)) if seconds is not None else ()


def audio_file_stamp(source):
    try:
        info = source.stat()
    except OSError:
        return None
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def probe_audio(source):
    stamp = audio_file_stamp(source)
    if stamp is not None:
        duration = read_audio_duration(source)
        if duration is not None and stamp == audio_file_stamp(source):
            return duration, stamp
    return None, None


def read_audio_duration(source):
    """Read a local file's duration for the form. FFmpeg still checks the actual audio when rendering."""
    try:
        if not source.is_file():
            return None
        if source.suffix.lower() == ".wav":
            with suppress(OSError, EOFError, ValueError, wave.Error):
                return wav_info(source)[0]
        probe = shutil.which("ffprobe")
        if not probe:
            return None
        output = subprocess.run([probe, "-v", "error", "-protocol_whitelist", "file,pipe",
                                 "-select_streams", "a:0", "-show_entries", "stream=duration:format=duration",
                                 "-of", "json", "-i", str(source.resolve())],
                                capture_output=True, check=True, timeout=4, stdin=subprocess.DEVNULL,
                                creationflags=NO_WINDOW)
        info = json.loads(output.stdout)
        if not info.get("streams"):
            return None
        for value in (info["streams"][0].get("duration"), info.get("format", {}).get("duration")):
            with suppress(TypeError, ValueError):
                seconds = float(value)
                if math.isfinite(seconds) and seconds > 0:
                    return seconds
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def clip(job, source, output):
    ffmpeg(job, output, "-ss", str(job.request.start), "-protocol_whitelist", "file,pipe",
           "-i", source.resolve(), "-map", "0:a:0", *duration_args(job.request.seconds))
    try:
        wav_info(output)
    except ValueError as exc:
        if job.request.start > 0:
            raise ValueError(f"在「{source.name}」的指定起点读不到音频。请把起点调早再试。") from exc
        raise


def wav_info(path):
    require_file(path, ".wav")
    with wave.open(str(path), "rb") as wav:
        frames, rate, channels, width = (wav.getnframes(), wav.getframerate(),
                                         wav.getnchannels(), wav.getsampwidth())
        if frames:
            wav.setpos(frames - 1)
        if not frames or len(wav.readframes(1)) != channels * width:
            raise ValueError(f"WAV 为空或数据不完整：{path.name}")
    return frames / rate, rate, channels, width


def require_vocal_signal(path, label):
    """Reject effectively silent PCM without claiming the signal contains lyrics."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{label}的 PCM 格式无法检查。")
        samples = array("h", wav.readframes(wav.getnframes()))
    if not samples or max(map(abs, samples)) < 32 or (
            sum(sample * sample for sample in samples) / len(samples) < 16):
        raise ValueError(f"{label}近乎静音。请选一段有清晰歌声的片段，再试换声。")


def separate(job, source):
    config = job.settings
    if not config.audio_python or not config.uvr_dir:
        raise ValueError("UVR 分轨引擎尚未准备。请查看《启动与回滚》。")
    model_dir = Path(config.uvr_dir)
    python = require_file(Path(config.audio_python), ".exe")
    require_file(model_dir / MODEL)
    if sha256(model_dir / MODEL) != MODEL_HASH:
        raise ValueError("UVR 权重 SHA-256 与本版本不匹配。")
    vocals, backing = job.folder / "vocals.wav", job.folder / "instrumental.wav"
    if config.uvr_backend not in {"auto", "directml", "cpu"}:
        raise ValueError("UVR 后端应为 auto、directml 或 cpu。")
    backends = ("directml", "cpu") if config.uvr_backend == "auto" else (config.uvr_backend,)
    for backend in backends:
        job.stage(30, f"UVR 分轨 · {backend}")
        try:
            run(job, [python, "-u", ROOT / "uvr_worker.py", source, model_dir,
                      job.folder, backend])
            wav_info(vocals)
            wav_info(backing)
            return vocals, backing, backend
        except (RuntimeError, ValueError, FileNotFoundError, wave.Error, TimeoutError):
            if backend == backends[-1]:
                raise
            vocals.unlink(missing_ok=True)
            backing.unlink(missing_ok=True)
            with (job.folder / "job.log").open("ab") as log:
                log.write(b"\nDirectML did not complete; retrying with CPU.\n")


def check_rvc(job):
    voice, config = job.request.voice, job.settings
    if not config.rvc_python or not config.rvc_root:
        raise ValueError("RVC 换声引擎尚未准备。请查看《启动与回滚》。")
    root = Path(config.rvc_root).resolve()
    require_file(Path(config.rvc_python), ".exe")
    require_file(root / "infer" / "cli.py")
    for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        require_file(root / "assets" / "hubert_base" / filename)
    if config.f0_method not in {"pm", "rmvpe"}:
        raise ValueError("RVC 音高提取应为 pm 或 rmvpe。")
    if config.f0_method == "rmvpe":
        require_file(root / "assets" / "rmvpe" / "rmvpe.pt")
    voice.verify()


def convert(job, source):
    voice, config = job.request.voice, job.settings
    voice.verify()
    root = Path(config.rvc_root).resolve()
    engine = Path(config.rvc_python)
    output = job.folder / "converted.wav"
    args = [engine, "-u", "-m", "infer.cli", "--input", source, "--output", output, "--format", "wav",
            "--model", voice.weights, "--speaker-id", voice.speaker, "--pitch", voice.pitch,
            "--f0-method", config.f0_method, "--index-rate", voice.index_rate if voice.index else 0]
    if voice.index and voice.index_rate:
        args.extend(["--index", voice.index])
    log = job.folder / "job.log"
    previous_size = log.stat().st_size
    run(job, args, cwd=root)
    if voice.index and voice.index_rate:
        with log.open("rb") as handle:
            handle.seek(previous_size)
            messages = handle.read()
        # Pinned RVC catches these FAISS failures and otherwise continues without the selected index.
        if b"index = faiss.read_index(" in messages or b"index_vectors = index.reconstruct_n(" in messages:
            raise ValueError("RVC 未能载入登记的检索索引，详见 job.log。")
    require_file(output, ".wav")
    return output


def original(job, output):
    from ace_original import ACE_FILES

    settings, req = job.settings, job.request
    if not settings.ace_python or not settings.ace_root:
        raise ValueError("ACE-Step CPU 引擎尚未准备。请查看《启动与回滚》。")
    engine = require_file(Path(settings.ace_python), ".exe")
    root = Path(settings.ace_root)
    require_file(root / "acestep" / "handler.py")
    for file in ACE_FILES:
        require_file(root / "checkpoints" / file)
    prompt = job.folder / "prompt.json"
    raw = job.folder / "generated.wav"
    write_json(prompt, {"caption": req.caption, "lyrics": req.lyrics, "language": req.language,
                        "seconds": req.seconds})
    job.stage(35, "ACE-Step · CPU 生成与解码")
    run(job, [engine, "-u", ROOT / "ace_original.py", root, prompt, raw])
    require_file(raw, ".wav")
    job.stage(85, "转换为标准 WAV")
    ffmpeg(job, output, "-i", raw, "-map", "0:a:0", "-t", str(req.seconds))


def shape_vocal(job, source):
    output = job.folder / "timbre.wav"
    duration, rate, _, _ = wav_info(source)
    shifted_rate = round(rate * 2 ** (job.request.formant_shift / 12))
    restore = rate / shifted_rate
    ffmpeg(job, output, "-i", source, "-af",
           f"asetrate={shifted_rate},aresample={rate},"
           f"rubberband=tempo={restore}:pitch={restore}:formant=preserved,apad",
           "-t", str(duration))
    return output


def render(job):
    req = job.request
    job.stage(5, "检查文件")
    backend = ""
    output = job.folder / "preview.wav"
    if req.mode == "demo":
        job.stage(70, "生成测试音")
        ffmpeg(job, output, "-f", "lavfi", "-i", f"sine=frequency=440:duration={req.seconds}",
               "-af", "volume=0.35")
        return output, backend
    if req.mode == "original":
        original(job, output)
        return output, backend
    if req.voice is not None:
        job.stage(5, "检查换声引擎和音色文件")
        check_rvc(job)

    job.stage(15, "读取整首音频" if req.seconds is None else "截取音频")
    duration_limit = duration_args(req.seconds)
    primary = job.folder / "input.wav"
    clip(job, req.source, primary)
    vocals, backing = primary, None
    if req.mode in {"cover", "separate"}:
        vocals, backing, backend = separate(job, primary)
        if req.mode == "separate" or req.seconds is None:
            input_duration = wav_info(primary)[0]
            if any(abs(wav_info(track)[0] - input_duration) > 0.05 for track in (vocals, backing)):
                raise ValueError("分离结果时长与输入不一致，请查看任务日志。")
        if req.mode == "separate":
            job.stage(85, "保存分轨")
            shutil.copyfile(backing, output)
            return output, backend
    if req.mode in {"stems", "remix"} and req.backing is not None:
        backing = job.folder / "backing.wav"
        clip(job, req.backing, backing)
    if req.voice is not None:
        require_vocal_signal(vocals, "导唱或分离人声")
        job.stage(60, "RVC 转换角色音色")
        converted = convert(job, vocals)
        conversion_limit = duration_limit
        if req.seconds is None:
            vocal_duration = wav_info(vocals)[0]
            if abs(wav_info(converted)[0] - vocal_duration) > 0.1:
                raise ValueError("AI 人声时长与导唱不一致，请查看任务日志。")
            # RVC may round to its nearest analysis frame. Align the tail to the source.
            conversion_limit = ("-af", "apad", *duration_args(vocal_duration))
        vocals = job.folder / "ai-vocal.wav"
        ffmpeg(job, vocals, "-i", converted, "-map", "0:a:0", *conversion_limit)
        require_vocal_signal(vocals, "AI 人声")
    if req.formant_shift:
        job.stage(75, "调整人声音色")
        vocals = shape_vocal(job, vocals)

    job.stage(85, "快速混音与 WAV 编码" if req.mode == "remix" else "混音与 WAV 编码")
    vocal_gain = req.vocal_gain
    if vocal_gain is None:
        vocal_gain = 0.5 if backing else 1.0
    if backing is None:
        ffmpeg(job, output, "-i", vocals, "-map", "0:a:0", *duration_limit,
               "-af", f"volume={vocal_gain},alimiter=limit=0.95:level=false:latency=true")
    else:
        mix_length = "first" if req.mode == "cover" else "longest"
        ffmpeg(job, output, "-i", backing, "-i", vocals, "-filter_complex",
               f"[0:a][1:a]amix=inputs=2:duration={mix_length}:"
               f"weights='{req.backing_gain} {vocal_gain}':normalize=0,"
               "alimiter=limit=0.95:level=false:latency=true[m]",
               "-map", "[m]", *duration_limit)
    return output, backend


def execute(job):
    start = time.monotonic()
    try:
        with job.lock:
            job.check_cancel()
            job.state = "running"
            job.started_at = job.stage_at = start
        audio, backend = render(job)
        duration, rate, channels, width = wav_info(audio)
        limit = job.request.seconds
        if (rate, channels, width) != (44100, 2, 2) or (limit is not None and duration > limit + 0.05):
            raise ValueError("结果 WAV 超出规格。")
        voice = job.request.voice
        with job.lock:
            job.check_cancel()
            meta = {
                "mode": job.request.mode, "seconds": round(duration, 6),
                "source_title": job.source_title or None, "whole_song": job.request.seconds is None,
                "processing_seconds": round(time.monotonic() - start, 3),
                "original_backend": "ace-step-v15-turbo/cpu" if job.request.mode == "original" else None,
                "uvr_backend": backend or None, "voice": job.voice_title or None,
                "voice_sha256": voice.weights_hash if voice else None,
                "index_sha256": voice.index_hash if voice else None,
                "speaker": voice.speaker if voice else None, "pitch": voice.pitch if voice else None,
                "f0_method": job.settings.f0_method if voice else None,
                "index_rate": voice.index_rate if voice and voice.index else 0,
                "vocal_gain": job.request.vocal_gain, "backing_gain": job.request.backing_gain,
                "formant_shift": job.request.formant_shift,
                "lyrics_review": "pending_listen" if job.request.mode == "original" else None,
                "wav_sha256": sha256(audio),
            }
            write_json(job.folder / "result.json", meta)
            audio.replace(job.result)
            job.metadata = meta
            detail = f"{duration:.2f} 秒"
            if job.request.mode == "original":
                detail += " · 歌词待听验"
            job.state, job.progress, job.detail = "done", 100, detail
    except Exception as exc:
        with job.lock:
            for path in (job.result, job.folder / "result.json"):
                with suppress(OSError):
                    path.unlink(missing_ok=True)
            if isinstance(exc, Cancelled) or job.cancel.is_set():
                job.state, job.detail = "cancelled", "已取消"
            else:
                job.state, job.detail = "failed", str(exc)
