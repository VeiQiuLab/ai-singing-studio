import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import studio


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for test condition")


def pcm(path, seconds=2, sample=800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * (2 * int(44100 * seconds)))


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        studio.DATA.mkdir(exist_ok=True)
        scratch = tempfile.TemporaryDirectory(prefix="test-", dir=studio.DATA)
        self.addCleanup(scratch.cleanup)
        self.base = Path(scratch.name)

    def queue(self):
        jobs = studio.JobQueue(self.base)
        self.addCleanup(jobs.worker.join, 12)
        self.addCleanup(jobs.close)
        return jobs

    def test_profile_requires_permission_and_rejects_changed_checkpoint(self):
        model = self.base / "voice.pth"
        model.write_bytes(b"first snapshot")
        with self.assertRaisesRegex(ValueError, "授权"):
            studio.Voice.register("role", "", model)
        voice = studio.Voice.register("role", "synthetic test", model, pitch=-2)
        voice.verify()
        model.write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "文件已变化"):
            voice.verify()

    def test_mode_and_clip_validation(self):
        source = self.base / "source.wav"
        pcm(source)
        with self.assertRaisesRegex(ValueError, "角色模型"):
            studio.Request("cover", source).validate()
        for start, seconds in ((-1, 4), (float("nan"), 4), (0, -1), (0, float("inf"))):
            with self.subTest(start=start, seconds=seconds):
                with self.assertRaises(ValueError):
                    studio.Request("stems", source, start=start, seconds=seconds).validate()
        with self.assertRaises(ValueError):
            studio.Request("demo", seconds=31).validate()

    def test_original_request_requires_prompt_lyrics_and_valid_duration(self):
        studio.Request("original", seconds=15, caption="folk", lyrics="morning", language="zh").validate()
        for changes in ({"seconds": 14}, {"caption": " "}, {"lyrics": ""}, {"language": "xx"}):
            settings = {"seconds": 15, "caption": "folk", "lyrics": "morning"} | changes
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                studio.Request("original", **settings).validate()

    def test_audio_workflows_accept_full_song_and_long_segments(self):
        source = self.base / "source.wav"
        pcm(source)
        weights = self.base / "voice.pth"
        weights.write_bytes(b"synthetic weights")
        voice = studio.Voice.register("role", "test fixture", weights)
        for mode in ("separate", "cover", "stems"):
            role = voice if mode == "cover" else None
            with self.subTest(mode=mode):
                studio.Request(mode, source, seconds=None, voice=role).validate()
                studio.Request(mode, source, start=1, seconds=180, voice=role).validate()
        for mode, start, seconds in (("separate", 1, None), ("separate", 0, -1),
                                     ("separate", 0, float("inf")), ("demo", 0, None), ("stems", 1, None)):
            with self.subTest(mode=mode, start=start, seconds=seconds), self.assertRaises(ValueError):
                studio.Request(mode, source, start=start, seconds=seconds).validate()

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_gui_separates_entire_song_and_long_segment(self):
        import tkinter as tk
        import gui

        root = tk.Tk()
        root.withdraw()
        source = self.base / "整首歌曲.wav"
        pcm(source, seconds=48)

        def copy_separation(job, prepared):
            vocals = job.folder / "vocals.wav"
            backing = job.folder / "instrumental.wav"
            shutil.copyfile(prepared, vocals)
            shutil.copyfile(prepared, backing)
            return vocals, backing, "cpu"

        with patch.object(gui, "JobQueue", side_effect=self.queue), patch.object(gui, "DATA", self.base), \
             patch.object(studio, "separate", side_effect=copy_separation), \
             patch.object(gui.messagebox, "showerror") as show_error:
            app = gui.StudioApp(root)
            try:
                app.mode.set(next(label for label, mode in gui.MODES.items() if mode == "separate"))
                app.source.set(str(source))
                app.start.set("1")
                app.seconds.set("44")
                app.update_form()
                self.assertFalse(app.timing.winfo_manager())
                app.submit_button.invoke()
                show_error.assert_not_called()
                job = app.selected()
                wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
                self.assertEqual(job.snapshot()[0], "done", job.snapshot())
                self.assertEqual(studio.wav_info(job.result)[0], 48)
                self.assertTrue(all(studio.wav_info(track)[0] == 48 for track in job.tracks().values()))

                app.full_song_check.invoke()
                self.assertEqual(app.timing.winfo_manager(), "pack")
                app.submit_button.invoke()
                show_error.assert_not_called()
                job = app.selected()
                wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
                self.assertEqual(job.snapshot()[0], "done", job.snapshot())
                self.assertEqual(studio.wav_info(job.result)[0], 44)

                app.reuse_stems()
                self.assertEqual(gui.MODES[app.mode.get()], "stems")
                self.assertEqual(float(app.seconds.get()), 44)
            finally:
                root.after_cancel(app.tick_callback)
                root.update_idletasks()
                root.destroy()

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_gui_full_cover_and_reuse_stems_keep_whole_duration(self):
        import tkinter as tk
        import gui

        root = tk.Tk()
        root.withdraw()
        source = self.base / "整首翻唱.wav"
        pcm(source, seconds=36)
        weights = self.base / "voice.pth"
        weights.write_bytes(b"synthetic weights")
        voice = studio.Voice.register("role", "test fixture", weights)

        def copy_separation(job, prepared):
            vocals, backing = job.folder / "vocals.wav", job.folder / "instrumental.wav"
            shutil.copyfile(prepared, vocals)
            shutil.copyfile(prepared, backing)
            return vocals, backing, "cpu"

        with patch.object(gui, "JobQueue", side_effect=self.queue), patch.object(gui, "DATA", self.base), \
             patch.object(studio, "separate", side_effect=copy_separation), \
             patch.object(studio, "check_rvc"), \
             patch.object(studio, "convert", side_effect=lambda job, source: source) as convert, \
             patch.object(gui.messagebox, "showerror") as show_error:
            app = gui.StudioApp(root)
            try:
                app.source.set(str(source))
                app.voices["role"] = voice
                app.voice.set("role")
                app.full_song.set(False)
                app.update_form()
                app.start.set("5")
                app.seconds.set("12")
                app.pitch.set("2")
                app.submit_button.invoke()
                show_error.assert_not_called()
                trial = app.selected()
                wait_for(lambda: trial.snapshot()[0] in {"done", "failed"})
                self.assertEqual(trial.snapshot()[0], "done", trial.snapshot())
                self.assertEqual(studio.wav_info(trial.result)[0], 12)
                app.source.set("")
                app.pitch.set("-2")
                app.update_actions()
                app.actions["followup"].invoke()
                job = app.selected()
                self.assertIsNot(job, trial)
                self.assertEqual(job.request, replace(trial.request, start=0, seconds=None))
                wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
                self.assertEqual(job.snapshot()[0], "done", job.snapshot())
                self.assertEqual(studio.wav_info(job.result)[0], 36)
                self.assertEqual(studio.wav_info(job.tracks()["AI 人声"])[0], 36)

                app.reuse_stems()
                self.assertTrue(app.full_song.get())
                app.submit_button.invoke()
                reused = app.selected()
                wait_for(lambda: reused.snapshot()[0] in {"done", "failed"})
                self.assertEqual(reused.snapshot()[0], "done", reused.snapshot())
                self.assertEqual(studio.wav_info(reused.result)[0], 36)

                truncated = self.base / "truncated.wav"
                pcm(truncated)
                convert.side_effect = lambda job, source: truncated
                app.submit_button.invoke()
                failed = app.selected()
                wait_for(lambda: failed.snapshot()[0] in {"done", "failed"})
                self.assertEqual(failed.snapshot()[0], "failed")
                self.assertIn("AI 人声时长", failed.snapshot()[2])
                self.assertFalse(failed.result.exists())
            finally:
                root.after_cancel(app.tick_callback)
                root.update_idletasks()
                root.destroy()

    def test_cover_checks_rvc_before_separation(self):
        source = self.base / "song.wav"
        pcm(source)
        weights = self.base / "voice.pth"
        weights.write_bytes(b"synthetic weights")
        voice = studio.Voice.register("role", "test fixture", weights)
        with patch.object(studio, "separate") as separate:
            job = self.queue().submit(studio.Request("cover", source, voice=voice), studio.Settings())
            wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
            separate.assert_not_called()
        self.assertEqual(job.snapshot()[0], "failed")
        self.assertIn("RVC 换声引擎尚未准备", job.snapshot()[2])
        self.assertFalse((job.folder / "input.wav").exists())

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_gui_replaced_source_does_not_use_old_duration(self):
        import tkinter as tk
        import gui

        source = self.base / "song.wav"
        pcm(source, seconds=2)
        root = tk.Tk()
        root.withdraw()
        with patch.object(gui, "JobQueue", side_effect=self.queue), patch.object(gui, "DATA", self.base):
            app = gui.StudioApp(root)
            try:
                app.mode.set(next(label for label, mode in gui.MODES.items() if mode == "stems"))
                app.source.set(str(source))
                app.full_song.set(False)
                app.start.set("3")
                app.seconds.set("1")
                app.update_form()

                def duration_known():
                    app.update_duration_hint()
                    return app.source_duration == 2

                wait_for(duration_known)
                with patch.object(gui.messagebox, "showerror") as error:
                    app.submit_button.invoke()
                    error.assert_called_once()
                    self.assertIsNone(app.selected())
                    error.reset_mock()
                    pcm(source, seconds=5)
                    app.submit_button.invoke()
                    error.assert_not_called()
                job = app.selected()
                wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
                self.assertEqual(job.snapshot()[0], "done", job.snapshot())
                self.assertEqual(studio.wav_info(job.result)[0], 1)
            finally:
                root.after_cancel(app.tick_callback)
                app.duration_worker.shutdown(wait=True, cancel_futures=True)
                root.destroy()

    def test_auto_retries_cpu_and_cleans_incomplete_directml_stems(self):
        folder = self.base / "job"
        folder.mkdir()
        models = self.base / "models"
        models.mkdir()
        model = models / studio.MODEL
        model.write_bytes(b"synthetic placeholder")
        settings = studio.Settings(audio_python=sys.executable, uvr_dir=str(models), uvr_backend="auto")
        job = studio.Job(studio.Request("separate"), settings, folder)
        calls = []

        def run(job, args, **kwargs):
            calls.append(args[-1])
            vocals = job.folder / "vocals.wav"
            if args[-1] == "directml":
                vocals.write_bytes(b"partial")
                raise RuntimeError("GPU unavailable")
            self.assertFalse(vocals.exists())
            pcm(vocals)
            pcm(job.folder / "instrumental.wav")

        with patch.object(studio, "MODEL_HASH", studio.sha256(model)), patch.object(studio, "run", run):
            _, _, backend = studio.separate(job, folder / "input.wav")
        self.assertEqual(calls, ["directml", "cpu"])
        self.assertEqual(backend, "cpu")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_demo_followed_by_stems_with_unicode_paths(self):
        jobs = self.queue()
        source = self.base / "原创 导唱.wav"
        backing = self.base / "原创 伴奏.wav"
        pcm(source, 2, sample=800)
        pcm(backing, 2, sample=1600)
        demo = jobs.submit(studio.Request("demo", seconds=30), studio.Settings())
        stems = jobs.submit(studio.Request("stems", source, backing, start=0.5, seconds=1), studio.Settings())
        wait_for(lambda: stems.snapshot()[0] in {"done", "failed"})
        self.assertEqual(demo.snapshot()[:2], ("done", 100), demo.snapshot())
        self.assertEqual(stems.snapshot()[:2], ("done", 100), stems.snapshot())
        self.assertEqual(studio.wav_info(demo.result), (30.0, 44100, 2, 2))
        self.assertEqual(studio.wav_info(stems.result), (1.0, 44100, 2, 2))
        with wave.open(str(stems.result), "rb") as handle:
            self.assertEqual(int.from_bytes(handle.readframes(1)[:2], "little", signed=True), 1200)
        self.assertEqual(json.loads((stems.folder / "result.json").read_text(encoding="utf-8"))["mode"], "stems")

    def test_invalid_media_does_not_publish_wav(self):
        jobs = self.queue()
        source = self.base / "invalid.wav"
        source.write_bytes(b"this is not audio")
        job = jobs.submit(studio.Request("stems", source), studio.Settings())
        wait_for(lambda: job.snapshot()[0] == "failed")
        self.assertFalse(job.result.exists())
        self.assertFalse((job.folder / "result.json").exists())

    def test_silent_pcm_is_rejected_before_voice_conversion(self):
        voice = self.base / "quiet.wav"
        pcm(voice, sample=1)
        with self.assertRaisesRegex(ValueError, "近乎静音"):
            studio.require_vocal_signal(voice, "分离人声")
        with self.assertRaises(ValueError):
            studio.Request("demo", vocal_gain=0).validate()

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_vocal_shaping_preserves_duration_and_exposes_track(self):
        jobs = self.queue()
        demo = jobs.submit(studio.Request("demo", seconds=2), studio.Settings())
        wait_for(lambda: demo.snapshot()[0] in {"done", "failed"})
        self.assertEqual(demo.snapshot()[0], "done", demo.snapshot())
        changed = jobs.submit(studio.Request("stems", demo.result, seconds=2, formant_shift=2), studio.Settings())
        wait_for(lambda: changed.snapshot()[0] in {"done", "failed"})
        self.assertEqual(changed.snapshot()[0], "done", changed.snapshot())
        self.assertEqual(studio.wav_info(changed.result), (2, 44100, 2, 2))
        self.assertIn("调整后人声", changed.tracks())
        self.assertNotEqual(studio.sha256(changed.tracks()["调整后人声"]), studio.sha256(demo.result))

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_cancel_process_tree_and_pending_job_then_continue(self):
        import ctypes as ct
        from ctypes import wintypes as w

        jobs = self.queue()
        marker = self.base / "child.pid"
        child = self.base / "child.py"
        parent = self.base / "parent.py"
        child.write_text("import time; time.sleep(60)", encoding="utf-8")
        parent.write_text(
            "import subprocess,sys,time\nfrom pathlib import Path\n"
            "p=subprocess.Popen([sys.executable,sys.argv[1]])\n"
            "Path(sys.argv[2]).write_text(str(p.pid), encoding='ascii')\n"
            "time.sleep(60)\n", encoding="utf-8")
        render = studio.render
        controlled = threading.Event()

        def slow_first(job):
            if not controlled.is_set():
                controlled.set()
                studio.run(job, [sys.executable, parent, child, marker])
            return render(job)

        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        kernel.OpenProcess.restype = w.HANDLE
        kernel.WaitForSingleObject.argtypes = (w.HANDLE, w.DWORD)
        kernel.WaitForSingleObject.restype = w.DWORD
        kernel.CloseHandle.argtypes = (w.HANDLE,)

        with patch.object(studio, "render", slow_first):
            request = studio.Request("demo", seconds=1)
            active = jobs.submit(request, studio.Settings())
            self.addCleanup(active.stop)
            wait_for(lambda: marker.is_file() and marker.stat().st_size > 0)
            pid = int(marker.read_text(encoding="ascii"))
            child_handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            self.assertTrue(child_handle)
            self.addCleanup(kernel.CloseHandle, child_handle)
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 0), 258)  # WAIT_TIMEOUT
            pending = jobs.submit(request, studio.Settings())
            next_job = jobs.submit(request, studio.Settings())
            pending.stop()
            active.stop()
            wait_for(lambda: next_job.snapshot()[0] in {"done", "failed"})
            self.assertEqual(next_job.snapshot()[0], "done")
            for job in (active, pending):
                self.assertEqual(job.snapshot()[0], "cancelled")
                self.assertFalse(job.result.exists())
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 4000), 0)

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_native_gui_demo_export_and_preview_controls(self):
        import tkinter as tk
        import gui

        root = tk.Tk()
        root.withdraw()
        with patch.object(gui, "JobQueue", side_effect=self.queue), patch.object(gui, "DATA", self.base):
            app = gui.StudioApp(root)
            try:
                for key, pitch in (("a", 0), ("b", 3)):
                    weights = self.base / f"{key}.pth"
                    weights.write_bytes(key.encode("ascii"))
                    voice = studio.Voice.register(key, "synthetic fixture", weights, pitch=pitch)
                    studio.write_json(self.base / "voices" / f"{key}.json", asdict(voice))
                app.load_voices()
                self.assertEqual(set(app.voice_box["values"]), {gui.NO_VOICE, "a · a", "b · b"})
                for selection, pitch in (("a · a", 0), ("b · b", 3)):
                    app.voice.set(selection)
                    self.assertEqual(app.voices[app.voice.get()].pitch, pitch)
                app.mode.set(next(label for label, mode in gui.MODES.items() if mode == "demo"))
                app.update_form()
                app.submit_button.invoke()
                job = app.selected()
                wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
                root.update()
                app.update_actions()
                self.assertEqual(job.snapshot()[0], "done", job.snapshot())
                target = self.base / "试听 导出.wav"
                with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(target)):
                    app.actions["export"].invoke()
                self.assertEqual(studio.sha256(job.result), studio.sha256(target))
                shared = self.base / "聊天试听.mp3"
                with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(shared)), \
                     patch.object(gui.messagebox, "showerror") as show_error:
                    app.actions["export"].invoke()
                show_error.assert_not_called()
                self.assertGreater(shared.stat().st_size, 0)
                subprocess.run(["ffmpeg", "-v", "error", "-f", "mp3", "-i", str(shared),
                                "-f", "null", "-"], check=True, capture_output=True)
                with patch.object(app.preview, "play") as play:
                    app.actions["play"].invoke()
                    play.assert_called_once_with()
                self.assertEqual(app.preview.path, job.result)
                app.seek_preview(14.0, False)
                self.assertEqual(app.timecode.cget("text"), "0:14 / 0:30")
                app.actions["stop"].invoke()
                self.assertEqual(app.preview.position, 0)
            finally:
                root.after_cancel(app.tick_callback)
                root.update_idletasks()
                root.destroy()

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_recent_results_remix_uses_final_stems_and_keeps_source(self):
        folder = self.base / "jobs" / "existing"
        for filename, sample in (("result.wav", 1500), ("vocals.wav", 200), ("ai-vocal.wav", 400),
                                 ("timbre.wav", 800), ("instrumental.wav", 600)):
            pcm(folder / filename, seconds=3, sample=sample)
        studio.write_json(folder / "result.json", {"mode": "cover", "seconds": 3, "whole_song": True,
                                                   "voice": "角色音色", "source_title": "已有歌曲"})
        broken = self.base / "jobs" / "incomplete"
        studio.write_json(broken / "result.json", {"mode": "cover", "seconds": 3})
        before = studio.sha256(folder / "result.wav")
        parent, = studio.recent_jobs(self.base)
        self.assertIsNone(parent.request.seconds)
        self.assertEqual(parent.mix_tracks(), (folder / "timbre.wav", folder / "instrumental.wav"))
        with patch.object(studio, "separate", side_effect=AssertionError("must not separate")), \
             patch.object(studio, "convert", side_effect=AssertionError("must not convert")):
            job = self.queue().submit(parent.remix(1.0, 0.0), studio.Settings())
            wait_for(lambda: job.snapshot()[0] in {"done", "failed"})
        self.assertEqual(job.snapshot()[0], "done", job.snapshot())
        self.assertEqual(studio.wav_info(job.result)[0], 3)
        with wave.open(str(job.result), "rb") as handle:
            handle.setpos(44100)
            self.assertAlmostEqual(int.from_bytes(handle.readframes(1)[:2], "little", signed=True), 800, delta=1)
        self.assertEqual(studio.sha256(folder / "result.wav"), before)
        newest, original = studio.recent_jobs(self.base)
        self.assertEqual(newest.folder, job.folder)
        self.assertEqual(original.folder, parent.folder)
        self.assertEqual((newest.source_title, newest.voice_title), ("已有歌曲", "角色音色"))
        self.assertEqual(set(newest.tracks()), {"混音成品", "混音人声", "混音伴奏"})

    @unittest.skipUnless(sys.platform == "win32", "Windows required")
    def test_auto_index_requires_an_unambiguous_match(self):
        from gui import matching_index

        model = self.base / "角色.pth"
        for name in ("another.index", "trained_IVF1_Flat_nprobe_1_角色_v2.index"):
            (self.base / name).touch()
        self.assertEqual(matching_index(model), "")
        index = self.base / "added_IVF1_Flat_nprobe_1_角色_v2.index"
        index.touch()
        self.assertEqual(matching_index(model), str(index))
        (self.base / "角色.index").touch()
        self.assertEqual(matching_index(model), "")

    @unittest.skipUnless(sys.platform == "win32" and shutil.which("ffmpeg"), "Windows + FFmpeg required")
    def test_failed_mp3_export_preserves_existing_file(self):
        from gui import export_audio

        source = self.base / "坏输入.wav"
        target = self.base / "原文件.mp3"
        source.write_bytes(b"invalid input")
        target.write_bytes(b"keep original")
        with self.assertRaisesRegex(ValueError, "MP3 编码失败"):
            export_audio(source, target)
        self.assertEqual(target.read_bytes(), b"keep original")
        self.assertEqual(set(self.base.iterdir()), {source, target})

    @unittest.skipUnless(sys.platform == "win32", "Windows required")
    def test_gui_selects_and_reuses_two_separate_stems(self):
        import tkinter as tk
        import gui

        root = tk.Tk()
        root.withdraw()
        with patch.object(gui, "JobQueue", side_effect=self.queue), patch.object(gui, "DATA", self.base):
            app = gui.StudioApp(root)
            try:
                folder = self.base / "fixture"
                for filename, sample in (("vocals.wav", 300), ("instrumental.wav", 600), ("input.wav", 900)):
                    pcm(folder / filename, sample=sample)
                job = studio.Job(studio.Request("separate", folder / "input.wav", seconds=2),
                                 studio.Settings(), folder, state="done", progress=100)
                app.jobs["fixture"] = ("仅分离人声与伴奏", job)
                app.tree.insert("", "end", iid="fixture")
                app.tree.selection_set("fixture")
                app.update_actions()
                self.assertEqual(app.selected_track(), folder / "vocals.wav")
                app.track.set("分离伴奏")
                exported = self.base / "export.wav"
                with patch.object(gui.filedialog, "asksaveasfilename", return_value=str(exported)):
                    app.actions["export"].invoke()
                self.assertEqual(studio.sha256(exported), studio.sha256(folder / "instrumental.wav"))
                app.actions["reuse"].invoke()
                self.assertEqual(gui.MODES[app.mode.get()], "stems")
                self.assertEqual(app.source.get(), str(folder / "vocals.wav"))
                self.assertEqual(app.backing.get(), str(folder / "instrumental.wav"))
            finally:
                root.after_cancel(app.tick_callback)
                root.update_idletasks()
                root.destroy()


if __name__ == "__main__":
    unittest.main()
