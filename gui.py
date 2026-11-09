"""Windows desktop interface for AI Singing Studio."""

import json
import math
import os
import re
import shutil
import subprocess
import time
import tkinter as tk
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from preview import PreviewPlayer
from studio import (DATA, MEDIA_MODES, NO_WINDOW, ROOT, JobQueue, Request, Settings, Voice,
                    audio_file_stamp, probe_audio, recent_jobs, write_json)

MODES = {
    "自动翻唱": "cover",
    "仅分离人声与伴奏": "separate",
    "用干声与伴奏直接换声 / 混音": "stems",
    "ACE-Step 歌曲草稿": "original",
    "设备测试音": "demo",
}
STATE = {"queued": "排队", "running": "运行", "cancelling": "正在取消",
         "cancelled": "已取消", "failed": "失败", "done": "完成"}
TERMINAL = {"cancelled", "failed", "done"}
NO_VOICE = "保留输入人声"
FORMANTS = {"偏厚": -2, "稍厚": -1, "原样": 0, "稍亮": 1, "偏亮": 2}
AUDIO = [("音频", "*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.aac *.wma"), ("所有文件", "*.*")]
MODE_LABELS = {value: label for label, value in MODES.items()} | {"remix": "快速混音"}
PREVIEW_ERRORS = (OSError, ValueError, EOFError, wave.Error)


def choose_file(parent, target, filters=AUDIO):
    path = filedialog.askopenfilename(parent=parent, filetypes=filters)
    if path:
        target.set(path)


def matching_index(model):
    stem = model.stem.casefold()
    try:
        candidates = [path for path in model.parent.iterdir()
                      if path.suffix.lower() == ".index" and path.is_file() and "trained" not in str(path)
                      and (path.stem.casefold() == stem or re.fullmatch(
                          rf"added_ivf\d+_flat_nprobe_\d+_{re.escape(stem)}_v[12]", path.stem.casefold()))]
    except OSError:
        return ""
    return str(candidates[0]) if len(candidates) == 1 else ""


def clock_text(seconds):
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}:{remainder:02}"


class VoiceDialog(simpledialog.Dialog):
    def body(self, master):
        master.columnconfigure(1, weight=1)
        self.fields = {}
        self.suggestions = {"name": "", "index": ""}

        def field(parent, row, key, label, value=""):
            var = self.fields[key] = tk.StringVar(value=value)
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
            ttk.Entry(parent, textvariable=var, width=42).grid(row=row, column=1, sticky="ew", padx=6)

        field(master, 0, "weights", "音色模型 (.pth)")
        first = ttk.Button(master, text="选模型", command=self.pick_model)
        first.grid(row=0, column=2)
        field(master, 1, "name", "显示名称")
        field(master, 2, "index", "检索索引（可空）")
        ttk.Button(master, text="选索引", command=lambda: choose_file(
            self, self.fields["index"], [("索引", "*.index")])).grid(row=2, column=2)
        field(master, 3, "permission", "授权依据")
        ttk.Label(master, text="填模型作者写明许可的页面，或你的实际授权记录。\n"
                              "例：本人录音并训练；作者允许翻唱（附原帖链接）。",
                  style="Secondary.TLabel").grid(row=4, column=1, columnspan=2, sticky="w", padx=6)
        self.allowed = tk.BooleanVar(value=False)
        ttk.Checkbutton(master, variable=self.allowed, text="我确认有权用这个音色进行翻唱",
                        ).grid(row=5, column=0, columnspan=3, sticky="w", padx=8, pady=9)
        advanced = ttk.Frame(master)
        field(advanced, 0, "speaker", "说话人编号", "0")
        field(advanced, 1, "pitch", "默认移调（半音）", "0")
        field(advanced, 2, "index_rate", "索引比重", "0.6")
        advanced.grid(row=7, column=0, columnspan=3, sticky="w")
        advanced.grid_remove()
        expanded = tk.BooleanVar(value=False)
        ttk.Checkbutton(master, text="高级设置", variable=expanded,
                        command=lambda: advanced.grid() if expanded.get() else advanced.grid_remove()
                        ).grid(row=6, column=0, columnspan=3, sticky="w", padx=8)
        return first

    def pick_model(self):
        chosen = filedialog.askopenfilename(parent=self, filetypes=[("RVC 模型", "*.pth")])
        if not chosen:
            return
        path = Path(chosen)
        self.fields["weights"].set(chosen)
        for key, value in {"name": path.stem, "index": matching_index(path)}.items():
            field = self.fields[key]
            if field.get() in {"", self.suggestions[key]}:
                field.set(value)
                self.suggestions[key] = value

    def validate(self):
        try:
            if not self.allowed.get():
                raise ValueError("请确认模型授权。")
            v = {key: var.get() for key, var in self.fields.items()}
            self.result = Voice.register(v["name"], v["permission"], v["weights"], v["index"],
                                         int(v["speaker"]), int(v["pitch"]), float(v["index_rate"]))
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法登记角色", str(exc), parent=self)
            return False
        return True


class RemixDialog(simpledialog.Dialog):
    def __init__(self, app, job):
        self.app, self.job = app, job
        super().__init__(app.root, title="调整混音")

    def body(self, master):
        ttk.Label(master, text="使用这次成品的人声和伴奏，只调整音量。\n会另存一个版本。"
                  ).pack(anchor="w", padx=10, pady=(6, 12))
        tracks = ttk.Frame(master, padding=(10, 0))
        tracks.pack(fill="x")
        self.levels = []
        for label, key, fallback, minimum in (("人声", "vocal_gain", 0.5, 1),
                                                ("伴奏", "backing_gain", 0.5, 0)):
            gain = self.job.metadata.get(key)
            if type(gain) not in {int, float} or not minimum / 100 <= gain <= 1.5:
                gain = fallback
            level = tk.DoubleVar(value=100 * gain)
            row = ttk.Frame(tracks)
            row.pack(anchor="w", pady=5)
            self.app.level_slider(row, label, level, minimum)
            self.levels.append(level)

    def buttonbox(self):
        box = ttk.Frame(self, padding=10)
        ttk.Button(box, text="取消", command=self.cancel).pack(side="left", padx=4)
        ttk.Button(box, text="生成新混音", style="Primary.TButton", command=self.ok).pack(side="left", padx=4)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        box.pack()

    def apply(self):
        self.result = tuple(level.get() / 100 for level in self.levels)


class OriginalDialog(simpledialog.Dialog):
    def body(self, master):
        ttk.Label(master, text="曲风与编配").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.caption = ttk.Entry(master, width=58)
        self.caption.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(master, text="歌词").grid(row=1, column=0, sticky="nw", padx=8, pady=4)
        self.lyrics = tk.Text(master, width=55, height=9, wrap="word")
        self.lyrics.grid(row=1, column=1, sticky="ew")
        ttk.Label(master, text="语言").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.languages = {"中文": "zh", "英语": "en", "日语": "ja", "自动": "unknown"}
        self.language = ttk.Combobox(master, values=list(self.languages), state="readonly")
        self.language.current(0)
        self.language.grid(row=2, column=1, sticky="w", pady=4)
        return self.caption

    def validate(self):
        caption, lyrics = self.caption.get().strip(), self.lyrics.get("1.0", "end-1c").strip()
        if not caption or not lyrics:
            messagebox.showerror("缺少创作内容", "请填写曲风和歌词。", parent=self)
            return False
        self.result = (caption, lyrics, self.languages[self.language.get()])
        return True


def export_audio(source, target):
    suffix = target.suffix.lower()
    if suffix not in {".wav", ".mp3"}:
        raise ValueError("请选择 .wav 或 .mp3 文件名。")
    if target.resolve() == source.resolve():
        return
    staging = target.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        if suffix == ".wav":
            shutil.copyfile(source, staging)
        else:
            executable = shutil.which("ffmpeg")
            if executable is None:
                raise ValueError("导出 MP3 需要 FFmpeg。")
            try:
                subprocess.run([executable, "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
                                "-i", str(source), "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "160k",
                                "-f", "mp3", str(staging)], check=True, capture_output=True, timeout=30,
                               creationflags=NO_WINDOW)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("MP3 编码超时。") from exc
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.decode("utf-8", errors="replace").strip()
                raise ValueError(f"MP3 编码失败。\n{detail}") from exc
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)


def studio_style(root):
    paper, ink, muted, edge = "#ffffff", "#151515", "#666666", "#dddddd"
    root.configure(bg=paper)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=paper, foreground=ink, bordercolor=edge,
                    lightcolor=paper, darkcolor=edge, focuscolor="#555555")
    style.configure("Secondary.TLabel", foreground=muted)
    style.configure("TButton", background="#f4f4f4", padding=(10, 6))
    style.configure("TScale", troughcolor="#e1e1e1", background=paper, bordercolor="#c7c7c7")
    style.map("TButton", background=[("pressed", "#d8d8d8"), ("active", "#eaeaea")])
    style.configure("Primary.TButton", background=ink, foreground=paper, bordercolor=ink)
    style.map("Primary.TButton", background=[("disabled", "#bbbbbb"), ("pressed", "#444444"),
                                            ("active", "#303030")],
              foreground=[("disabled", paper), ("!disabled", paper)])
    style.configure("TCombobox", fieldbackground=paper, background="#f3f3f3", padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", paper), ("disabled", "#f5f5f5")],
              selectbackground=[("readonly", paper)], selectforeground=[("readonly", ink)])
    style.configure("TLabelframe", padding=(13, 8))
    style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))
    style.configure("Treeview", rowheight=28, fieldbackground=paper, borderwidth=0)
    style.map("Treeview", background=[("selected", ink)], foreground=[("selected", paper)])
    style.configure("Treeview.Heading", background="#f5f5f5", padding=6)


class StudioApp:
    def __init__(self, root):
        self.root = root
        self.queue = JobQueue()
        self.jobs = {}
        self.voices = {}
        self.closing = False
        self.preview = PreviewPlayer()
        self.preview_target = None
        self.preview_error = ""
        self.drag_resume = None
        self.duration_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-info")
        self.duration_task = None
        self.probed_source = ""
        self.source_duration = None
        self.duration_stamp = None
        root.title("AI Singing Studio")
        root.geometry("960x710")
        root.minsize(850, 690)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
        studio_style(root)
        header = ttk.Frame(root, padding=(24, 11, 24, 4))
        header.pack(fill="x")
        ttk.Label(header, text="AI Singing Studio", font=("Segoe UI", 26, "bold")).pack(anchor="w")
        ttk.Label(header, text="导入歌曲  →  分离双轨  →  换声混音  →  逐轨试听",
                  style="Secondary.TLabel").pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(root, padding=(18, 8))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        form = ttk.LabelFrame(body, text="01   音频与角色")
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        self.mode = tk.StringVar(value=next(iter(MODES)))
        self.source = tk.StringVar()
        self.backing = tk.StringVar()
        self.voice = tk.StringVar(value=NO_VOICE)
        self.start = tk.StringVar(value="0")
        self.seconds = tk.StringVar(value="30")
        self.full_song = tk.BooleanVar(value=True)
        self.note = tk.StringVar()
        self.track = tk.StringVar()
        self.pitch = tk.StringVar(value="0")
        self.formant = tk.StringVar(value="原样")
        self.vocal_level = tk.DoubleVar(value=95)
        self.backing_level = tk.DoubleVar(value=55)
        self.track_job = None

        ttk.Label(form, text="工作流").grid(row=0, column=0, sticky="w", padx=(0, 12))
        mode_box = ttk.Combobox(form, textvariable=self.mode, values=list(MODES), state="readonly")
        mode_box.grid(row=0, column=1, sticky="ew")
        mode_box.bind("<<ComboboxSelected>>", self.update_form)
        _, *self.source_controls = self.path_row(form, 1, "输入音频", self.source)
        self.backing_row = self.path_row(form, 2, "可选伴奏", self.backing)
        ttk.Label(form, text="角色音色").grid(row=3, column=0, sticky="w")
        self.voice_box = ttk.Combobox(form, textvariable=self.voice, state="readonly")
        self.voice_box.grid(row=3, column=1, sticky="ew")
        ttk.Button(form, text="导入音色模型", command=self.add_voice).grid(row=3, column=2, padx=(8, 0))
        controls = ttk.Frame(form)
        controls.grid(row=4, column=1, sticky="w", pady=(6, 0))
        self.clip_label = ttk.Label(form, text="片段（秒）")
        self.clip_label.grid(row=4, column=0, sticky="w")
        self.full_song_check = ttk.Checkbutton(form, text="处理整首", variable=self.full_song,
                                              command=self.update_form)
        self.full_song_check.grid(row=4, column=0, sticky="w")
        self.timing = ttk.Frame(controls)
        self.timing.pack(side="left")
        ttk.Label(self.timing, text="起点（秒）").pack(side="left")
        self.start_entry = ttk.Entry(self.timing, textvariable=self.start, width=8)
        self.start_entry.pack(side="left", padx=(5, 14))
        ttk.Label(self.timing, text="时长（秒）").pack(side="left")
        self.seconds_entry = ttk.Entry(self.timing, textvariable=self.seconds, width=8)
        self.seconds_entry.pack(side="left", padx=5)
        self.pitch_label = ttk.Label(controls, text="额外移调")
        self.pitch_label.pack(side="left", padx=(14, 0))
        self.pitch_box = ttk.Spinbox(controls, from_=-12, to=12, width=5, textvariable=self.pitch)
        self.pitch_box.pack(side="left", padx=5)
        ttk.Label(controls, text="音色").pack(side="left", padx=(12, 0))
        self.formant_box = ttk.Combobox(controls, textvariable=self.formant, values=list(FORMANTS),
                                        state="readonly", width=7)
        self.formant_box.pack(side="left", padx=5)
        self.submit_button = ttk.Button(form, text="开始翻唱", command=self.submit, style="Primary.TButton")
        self.submit_button.grid(row=4, column=2, padx=(8, 0), pady=(4, 0))
        mixer = ttk.Frame(form)
        mixer.grid(row=5, column=1, sticky="w", pady=(8, 0))
        ttk.Label(form, text="混音音量").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.level_controls = (self.level_slider(mixer, "人声", self.vocal_level, 10),
                               self.level_slider(mixer, "伴奏", self.backing_level, 0))
        ttk.Button(form, text="录音与训练", command=self.open_voice_guide).grid(row=5, column=2, padx=(8, 0))
        ttk.Label(form, textvariable=self.note, style="Secondary.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))

        pane = ttk.LabelFrame(body, text="02   任务队列与近期成品")
        pane.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.tree = ttk.Treeview(pane, show="headings", selectmode="browse", height=3,
                                 columns=("mode", "state", "progress", "detail"))
        for key, heading, width in (("mode", "歌曲 / 音色", 340), ("state", "状态", 78),
                                    ("progress", "进度", 68), ("detail", "步骤 / 时长", 240)):
            self.tree.heading(key, text=heading, anchor="w")
            self.tree.column(key, width=width, minwidth=width // 2, stretch=key in {"mode", "detail"})
        self.tree.bind("<<TreeviewSelect>>", self.update_actions)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(pane, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        bar = ttk.LabelFrame(body, text="03   试听与导出")
        bar.grid(row=2, column=0, sticky="ew", pady=(9, 0))
        self.actions = {}
        transport = ttk.Frame(bar)
        transport.pack(fill="x", pady=(0, 6))
        self.track_box = ttk.Combobox(transport, textvariable=self.track, width=19, state="disabled")
        self.track_box.pack(side="left", padx=(0, 8))
        self.track_box.bind("<<ComboboxSelected>>", self.update_actions)
        self.position = tk.DoubleVar(value=0.0)
        self.timeline = ttk.Scale(transport, from_=0.0, to=1.0, variable=self.position, state="disabled")
        self.timeline.pack(side="left", expand=True, fill="x", padx=(0, 8))
        for sequence, handler in (("<ButtonPress-1>", self.start_seek), ("<B1-Motion>", self.move_seek),
                                  ("<ButtonRelease-1>", self.finish_seek), ("<Left>", self.key_seek),
                                  ("<Right>", self.key_seek), ("<Home>", self.key_seek), ("<End>", self.key_seek)):
            self.timeline.bind(sequence, handler)
        self.timecode = ttk.Label(transport, text="0:00 / 0:00", width=15, anchor="e")
        self.timecode.pack(side="right")

        buttons = ttk.Frame(bar)
        buttons.pack(fill="x")
        for name, label, command in (("play", "播放", self.play), ("stop", "停止", self.stop_preview),
                                     ("export", "导出音频", self.export),
                                     ("remix", "调整混音", self.remix),
                                     ("reuse", "用这组分轨", self.reuse_stems),
                                     ("folder", "任务目录", self.open_folder),
                                     ("cancel", "取消任务", self.cancel_job)):
            button = self.actions[name] = ttk.Button(buttons, text=label, command=command)
            button.pack(side="right" if name == "cancel" else "left", padx=(0, 6))
        footer = ttk.Frame(bar)
        footer.pack(fill="x", pady=(5, 0))
        self.action_hint = tk.StringVar(value="拖动进度或切换音轨，接着听同一段。")
        ttk.Label(footer, textvariable=self.action_hint, style="Secondary.TLabel").pack(side="left")
        self.actions["followup"] = ttk.Button(footer, command=self.follow_up)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(body, textvariable=self.status, wraplength=810).grid(
            row=3, column=0, sticky="ew", pady=(5, 0))
        self.load_voices()
        self.update_form()
        for job in recent_jobs(DATA):
            self.add_job(job, select=False)
        self.tick()

    def path_row(self, form, row, label, var):
        caption = ttk.Label(form, text=label)
        caption.grid(row=row, column=0, sticky="w")
        entry = ttk.Entry(form, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=(4, 0))
        button = ttk.Button(form, text="选择文件", command=lambda: choose_file(self.root, var))
        button.grid(row=row, column=2, padx=(8, 0))
        return caption, entry, button

    def level_slider(self, container, label, variable, minimum):
        ttk.Label(container, text=label).pack(side="left")
        readout = ttk.Label(container, width=5, text=f"{variable.get():.0f}%")
        scale = ttk.Scale(container, from_=minimum, to=150, length=88, variable=variable,
                          command=lambda value: readout.configure(text=f"{float(value):.0f}%"))
        scale.pack(side="left", padx=(4, 2))
        readout.pack(side="left", padx=(0, 10))
        return scale

    def update_form(self, *_):
        mode = MODES[self.mode.get()]
        full_song = mode in MEDIA_MODES and self.full_song.get()
        if mode in MEDIA_MODES:
            self.clip_label.grid_remove()
            self.full_song_check.grid()
        else:
            self.full_song_check.grid_remove()
            self.clip_label.grid()
        if full_song:
            self.timing.pack_forget()
        elif not self.timing.winfo_manager():
            self.timing.pack(side="left", before=self.pitch_label)
        for control in self.source_controls:
            control.configure(state="disabled" if mode in {"demo", "original"} else "normal")
        self.start_entry.configure(state="disabled" if full_song or mode in {"demo", "original"} else "normal")
        self.seconds_entry.configure(state="disabled" if full_song else "normal")
        for control in self.backing_row:
            if mode == "stems":
                control.grid()
            else:
                control.grid_remove()
        self.voice_box.configure(state="readonly" if mode in {"cover", "stems"} else "disabled")
        self.pitch_box.configure(state="normal" if mode in {"cover", "stems"} else "disabled")
        self.formant_box.configure(state="readonly" if mode in {"cover", "stems"} else "disabled")
        for scale in self.level_controls:
            scale.configure(state="normal" if mode in {"cover", "stems"} else "disabled")
        self.submit_button.configure(text={"cover": "开始翻唱" if full_song else "试唱片段",
                                           "separate": "开始分离", "stems": "处理分轨",
                                           "original": "填写曲风与歌词", "demo": "生成测试音"}[mode])
        self.note_base = {
            "demo": "生成可试听的测试音，检查队列和音频导出。",
            "separate": ("整首分离无需填写时长。" if full_song else "起点填 0 从头开始，时长至少 1 秒。"),
            "cover": ("整首翻唱无需填写时长。" if full_song else "建议从有歌声的位置开始试唱。"),
            "stems": ("导入对齐的干声与伴奏，整首换声或混音。" if full_song else
                      "导入对齐的干声与伴奏，按起点和时长处理一段。"),
            "original": "生成 15–30 秒歌曲草稿。请试听确认歌词后再用于翻唱。",
        }[mode]
        self.update_duration_hint()

    def update_duration_hint(self):
        mode = MODES[self.mode.get()]
        source = self.source.get() if mode in MEDIA_MODES else ""
        if self.duration_task is not None and self.duration_task.done():
            self.source_duration, self.duration_stamp = self.duration_task.result()
            self.duration_task = None
        if not self.closing and self.duration_task is None and source != self.probed_source:
            self.probed_source, self.source_duration = source, None
            self.duration_stamp = None
            if source:
                self.duration_task = self.duration_worker.submit(probe_audio, Path(source))
        hint = self.note_base
        if source == self.probed_source and self.source_duration is not None:
            duration = self.source_duration
            hint = f"输入音频 {clock_text(duration)} · {hint}"
            if not self.full_song.get():
                try:
                    start, seconds = float(self.start.get()), float(self.seconds.get())
                except ValueError:
                    pass
                else:
                    if start >= duration:
                        hint = f"起点须早于音频结尾（{duration:.1f} 秒）。"
                    elif math.isfinite(start + seconds) and start >= 0 and seconds >= 1 and start + seconds > duration:
                        scope = "干声" if mode == "stems" else "片段"
                        hint = f"输入音频 {clock_text(duration)} · {scope}会处理到音频结尾。"
        if self.note.get() != hint:
            self.note.set(hint)

    def load_voices(self):
        self.voices.clear()
        for path in sorted((DATA / "voices").glob("*.json")):
            try:
                voice = Voice(**json.loads(path.read_text(encoding="utf-8")))
                voice.check_settings()
                self.voices[f"{voice.name} · {path.stem[:8]}"] = voice
            except (OSError, ValueError, TypeError, AttributeError):
                self.status.set(f"有一份角色配置无法读取：{path.name}")
        self.voice_box["values"] = [NO_VOICE, *self.voices]

    def add_voice(self):
        dialog = VoiceDialog(self.root, title="登记已获授权的角色音色")
        if dialog.result is not None:
            key = uuid.uuid4().hex
            try:
                write_json(DATA / "voices" / f"{key}.json", asdict(dialog.result))
                self.load_voices()
                self.voice.set(f"{dialog.result.name} · {key[:8]}")
            except OSError as exc:
                messagebox.showerror("无法保存角色", str(exc), parent=self.root)

    def submit(self):
        try:
            mode = MODES[self.mode.get()]
            settings = Settings() if mode == "demo" else Settings.load()
            full_song = mode in MEDIA_MODES and self.full_song.get()
            seconds = None if full_song else float(self.seconds.get())
            if mode == "original":
                if not settings.ace_python or not settings.ace_root:
                    raise ValueError("ACE-Step CPU 引擎尚未准备。请查看《启动与回滚》。")
                if not 15 <= seconds <= 30:
                    raise ValueError("原创片段时长为 15–30 秒。")
                dialog = OriginalDialog(self.root, title="原创片段")
                if dialog.result is None:
                    return
                caption, lyrics, language = dialog.result
                request = Request(mode, seconds=seconds, caption=caption, lyrics=lyrics, language=language)
            elif mode == "demo":
                request = Request(mode, seconds=seconds)
            else:
                voice = self.voices.get(self.voice.get()) if mode in {"cover", "stems"} else None
                if voice is not None:
                    voice = replace(voice, pitch=voice.pitch + int(self.pitch.get()))
                    voice.check_settings()
                source = Path(self.source.get()).resolve() if self.source.get() else None
                backing = Path(self.backing.get()).resolve() if self.backing.get() and mode == "stems" else None
                start = 0.0 if full_song else float(self.start.get())
                if (not full_song and self.probed_source == self.source.get() and self.source_duration is not None
                        and start >= self.source_duration):
                    if audio_file_stamp(source) == self.duration_stamp:
                        raise ValueError(f"起点须早于音频结尾（{self.source_duration:.1f} 秒）。")
                    self.probed_source = ""
                    self.update_duration_hint()
                request = Request(mode, source, backing, start, seconds, voice,
                                  vocal_gain=self.vocal_level.get() / 100,
                                  backing_gain=self.backing_level.get() / 100,
                                  formant_shift=FORMANTS[self.formant.get()] if mode in {"cover", "stems"} else 0)
            job = self.queue.submit(request, settings)
        except (TypeError, ValueError, OSError) as exc:
            messagebox.showerror("无法加入任务", str(exc), parent=self.root)
            return
        self.add_job(job)

    def add_job(self, job, select=True):
        key = job.folder.name
        workflow = MODE_LABELS[job.request.mode]
        label = job.source_title or ("未记录歌名" if job.request.mode in MEDIA_MODES else workflow)
        if job.voice_title:
            label += f" · {job.voice_title}"
        if job.request.mode == "remix":
            label += " · 快速混音"
        self.jobs[key] = (label, job)
        self.tree.insert("", 0 if select else "end", iid=key)
        if select:
            self.tree.selection_set(key)
            self.tree.see(key)
        state, progress, detail = job.snapshot()
        self.tree.item(key, values=(label, STATE[state], f"{progress}%", self.row_detail(job, state, detail)))

    @staticmethod
    def row_detail(job, state, detail):
        if state == "done" and job.request.mode == "cover":
            if job.request.source is not None or type(job.metadata.get("whole_song")) is bool:
                return ("整首 · " if job.request.seconds is None else "片段 · ") + detail
        return detail

    def selected(self):
        selection = self.tree.selection()
        return self.jobs[selection[0]][1] if selection else None

    def tick(self):
        self.update_duration_hint()
        for key, (label, job) in self.jobs.items():
            state, progress, detail = job.snapshot()
            self.tree.item(key, values=(label, STATE[state], f"{progress}%", self.row_detail(job, state, detail)))
        try:
            self.preview.update()
        except PREVIEW_ERRORS as exc:
            self.preview_failed(exc)
        self.update_actions()
        if self.closing and all(job.snapshot()[0] in TERMINAL for _, job in self.jobs.values()):
            self.queue.close()
            self.duration_worker.shutdown(wait=False, cancel_futures=True)
            self.root.destroy()
            return
        self.tick_callback = self.root.after(200, self.tick)

    def update_actions(self, *_):
        job = self.selected()
        state, progress, detail = job.snapshot() if job else (None, 0, "")
        tracks = job.tracks() if job else {}
        labels = tuple(tracks)
        folder = job.folder if job else None
        same_job = self.track_job == folder
        if not same_job or self.track.get() not in labels:
            self.track.set(next(iter(labels), ""))
        self.track_job = folder
        self.track_box.configure(values=labels, state="readonly" if labels else "disabled")
        target = tracks.get(self.track.get())
        if target != self.preview_target:
            if same_job and self.preview.playing:
                try:
                    self.preview.update()
                except PREVIEW_ERRORS as exc:
                    self.preview_failed(exc)
            position = self.preview.position if same_job else 0.0
            resume = same_job and (self.preview.playing or self.drag_resume is True)
            self.drag_resume = None
            self.preview_target = target
            self.preview_error = ""
            try:
                self.preview.load(target, position, resume)
            except PREVIEW_ERRORS as exc:
                self.preview_failed(exc)
            self.timeline.configure(to=max(0.001, self.preview.duration),
                                    state="normal" if self.preview.path else "disabled")
        playable = self.preview.path is not None
        followup = self.followup_label(job, state)
        button = self.actions["followup"]
        if followup:
            if not button.winfo_manager():
                button.pack(side="right", padx=(0, 6))
            button.configure(text=followup)
        elif button.winfo_manager():
            button.pack_forget()
        hint = {"翻唱整首": "沿用这次试唱的设置，从歌曲开头翻唱整首。",
                "重试任务": "按原参数重新加入队列。"}.get(followup, "拖动进度或切换音轨，接着听同一段。")
        if self.action_hint.get() != hint:
            self.action_hint.set(hint)
        allowed = {"cancel": state in {"queued", "running"}, "play": playable,
                   "export": bool(tracks), "stop": playable, "folder": job is not None,
                   "remix": bool(job and job.mix_tracks()), "followup": bool(followup),
                   "reuse": "分离人声" in tracks and "分离伴奏" in tracks}
        for key, button in self.actions.items():
            button.configure(state="normal" if allowed[key] and not self.closing else "disabled")
        if self.drag_resume is None:
            self.show_position(self.preview.position)
        self.actions["play"].configure(text="暂停" if self.preview.playing else "播放")
        if job:
            elapsed = ""
            now = time.monotonic()
            if state == "running" and job.started_at is not None and job.stage_at is not None:
                elapsed = f" · 本步 {clock_text(now - job.stage_at)} / 总用时 {clock_text(now - job.started_at)}"
            elif state == "done":
                seconds = job.metadata.get("processing_seconds")
                if type(seconds) in {int, float} and 0 <= seconds < 10**9:
                    elapsed = f" · 用时 {clock_text(seconds)}"
            self.status.set(f"{STATE[state]} · {progress}%  {detail}{elapsed}" +
                            (f"   |   {self.preview_error}" if self.preview_error else ""))
        else:
            self.status.set("就绪")

    def cancel_job(self):
        if (job := self.selected()) is not None:
            job.stop()

    @staticmethod
    def followup_label(job, state):
        if job and state in {"failed", "cancelled"}:
            return "重试任务"
        if job and state == "done" and job.request.mode == "cover" and job.request.seconds is not None:
            if job.request.source is not None and job.request.voice is not None:
                return "翻唱整首"
        return ""

    def follow_up(self):
        job = self.selected()
        if job is None or self.closing:
            return
        state = job.snapshot()[0]
        if not self.followup_label(job, state):
            return
        request, settings = job.request, job.settings
        try:
            if state == "done":
                request = replace(request, start=0.0, seconds=None)
            elif request.mode not in {"demo", "remix"}:
                settings = Settings.load()
            self.add_job(self.queue.submit(request, settings))
        except (TypeError, ValueError, OSError) as exc:
            messagebox.showerror("无法加入任务", str(exc), parent=self.root)

    def play(self):
        if self.preview.path is None:
            return
        self.preview_error = ""
        try:
            self.preview.update()
            if self.preview.playing:
                self.preview.seek(self.preview.position)
            elif self.preview.position >= self.preview.duration:
                self.preview.seek(0, True)
            else:
                self.preview.play()
        except PREVIEW_ERRORS as exc:
            self.preview_failed(exc)
        self.update_actions()

    def preview_failed(self, exc):
        self.preview_error = f"无法试听：{exc}"
        try:
            self.preview.release()
        except OSError:
            pass

    def stop_preview(self):
        self.drag_resume = None
        self.seek_preview(0, False)

    def show_position(self, position):
        self.position.set(position)
        self.timecode.configure(text=f"{clock_text(position)} / {clock_text(self.preview.duration)}")

    def seek_preview(self, position, resume):
        self.preview_error = ""
        try:
            self.preview.seek(position, resume)
        except PREVIEW_ERRORS as exc:
            self.preview_failed(exc)
        self.update_actions()

    def start_seek(self, event):
        if self.preview.path is not None:
            event.widget.focus_set()
            self.drag_resume = self.preview.playing
            self.seek_preview(self.preview.position, False)
            self.move_seek(event)
        return "break"

    def move_seek(self, event):
        if self.drag_resume is not None:
            width = max(event.widget.winfo_width() - 16, 1)
            ratio = min(max((event.x - 8) / width, 0), 1)
            self.show_position(self.preview.duration * ratio)
        return "break"

    def finish_seek(self, event):
        if self.drag_resume is not None:
            self.move_seek(event)
            resume, self.drag_resume = self.drag_resume, None
            self.seek_preview(self.position.get(), resume)
        return "break"

    def key_seek(self, event):
        if self.preview.path is not None:
            position = {"Left": self.preview.position - 5, "Right": self.preview.position + 5,
                        "Home": 0, "End": self.preview.duration}[event.keysym]
            self.seek_preview(position, self.preview.playing)
        return "break"

    def remix(self):
        job = self.selected()
        if not job or not job.mix_tracks():
            return
        dialog = RemixDialog(self, job)
        if dialog.result is None:
            return
        try:
            self.add_job(self.queue.submit(job.remix(*dialog.result), Settings()))
        except (TypeError,) + PREVIEW_ERRORS as exc:
            messagebox.showerror("无法调整混音", str(exc), parent=self.root)

    def export(self):
        if (source := self.selected_track()) is not None:
            path = filedialog.asksaveasfilename(parent=self.root, title="导出音频",
                                               defaultextension=".wav", initialfile=self.track.get(),
                                               filetypes=[("WAV 音频", "*.wav"), ("MP3 聊天分享", "*.mp3")])
            if path:
                try:
                    target = Path(path)
                    export_audio(source, target)
                except (OSError, ValueError) as exc:
                    messagebox.showerror("无法导出", str(exc), parent=self.root)

    def selected_track(self):
        job = self.selected()
        return job.tracks().get(self.track.get()) if job else None

    def reuse_stems(self):
        job = self.selected()
        tracks = job.tracks() if job else {}
        if "分离人声" in tracks and "分离伴奏" in tracks:
            self.mode.set(next(label for label, mode in MODES.items() if mode == "stems"))
            self.source.set(str(tracks["分离人声"]))
            self.backing.set(str(tracks["分离伴奏"]))
            self.start.set("0")
            self.full_song.set(job.request.seconds is None)
            self.seconds.set(str(job.request.seconds if job.request.seconds is not None else 30))
            self.update_form()

    def open_voice_guide(self):
        try:
            instructions = (ROOT / "音色与训练.md").read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("无法打开说明", str(exc), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title("录音与训练")
        window.transient(self.root)
        window.geometry("690x500")
        page = ttk.Frame(window, padding=16)
        page.pack(fill="both", expand=True)
        reader = tk.Text(page, wrap="word", padx=12, pady=10, relief="flat", borderwidth=0)
        scroll = ttk.Scrollbar(page, command=reader.yview)
        reader.configure(yscrollcommand=scroll.set)
        reader.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        reader.insert("1.0", instructions)
        reader.configure(state="disabled")

    def open_folder(self):
        if (job := self.selected()) is not None:
            os.startfile(job.folder)

    def close(self):
        active = [job for _, job in self.jobs.values() if job.snapshot()[0] not in TERMINAL]
        if active and not messagebox.askyesno("退出工作台", "取消尚未完成的任务并退出？", parent=self.root):
            return
        try:
            self.preview.close()
        except OSError as exc:
            self.preview_failed(exc)
        self.closing = True
        self.submit_button.configure(state="disabled")
        for job in active:
            job.stop()


if __name__ == "__main__":
    window = tk.Tk()
    StudioApp(window)
    window.mainloop()
