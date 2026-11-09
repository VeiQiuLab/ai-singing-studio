"""Seekable Windows PCM playback. Only a short window of audio stays in memory."""

import ctypes as c
import wave
from collections import deque
from ctypes import wintypes as w

WAVE_MAPPER = 0xFFFFFFFF
WHDR_DONE = 1
TIME_MS, TIME_SAMPLES, TIME_BYTES = 1, 2, 4


class WaveFormat(c.Structure):
    _pack_ = 1
    _fields_ = [("tag", w.WORD), ("channels", w.WORD), ("rate", w.DWORD),
                ("bytes_per_second", w.DWORD), ("block_align", w.WORD),
                ("bits", w.WORD), ("extra", w.WORD)]


class WaveHeader(c.Structure):
    _fields_ = [("data", c.c_void_p), ("length", w.DWORD), ("recorded", w.DWORD),
                ("user", c.c_size_t), ("flags", w.DWORD), ("loops", w.DWORD),
                ("next", c.c_void_p), ("reserved", c.c_size_t)]


class WaveTime(c.Structure):
    _fields_ = [("kind", w.UINT), ("value", w.DWORD), ("reserved", w.DWORD)]


class PreviewPlayer:
    def __init__(self):
        self.api = c.WinDLL("winmm")
        self.api.waveOutOpen.argtypes = [c.POINTER(w.HANDLE), w.UINT, c.POINTER(WaveFormat),
                                         c.c_size_t, c.c_size_t, w.DWORD]
        for name in ("waveOutPrepareHeader", "waveOutWrite", "waveOutUnprepareHeader"):
            getattr(self.api, name).argtypes = [w.HANDLE, c.POINTER(WaveHeader), w.UINT]
        self.api.waveOutGetPosition.argtypes = [w.HANDLE, c.POINTER(WaveTime), w.UINT]
        for name in ("waveOutReset", "waveOutClose"):
            getattr(self.api, name).argtypes = [w.HANDLE]
        self.device = w.HANDLE()
        self.reader = None
        self.buffers = deque()
        self.path = None
        self.duration = self.position = self.offset = 0.0
        self.eof = False

    @property
    def playing(self):
        return bool(self.device)

    @staticmethod
    def check(result):
        if result:
            raise OSError(f"Windows 音频输出错误（{result}）。")

    def load(self, path, position=0.0, play=False):
        self.close()
        if path is not None:
            with wave.open(str(path), "rb") as reader:
                self.duration = reader.getnframes() / reader.getframerate()
            self.path = path
            self.seek(position, play)

    def seek(self, seconds, play=False):
        self.release()
        self.position = min(max(0.0, seconds), self.duration)
        if play and self.position < self.duration:
            self.play()

    def play(self):
        if self.playing or self.path is None or self.position >= self.duration:
            return
        self.reader = wave.open(str(self.path), "rb")
        try:
            rate = self.reader.getframerate()
            width = self.reader.getsampwidth()
            channels = self.reader.getnchannels()
            block = channels * width
            self.format = WaveFormat(1, channels, rate, rate * block, block, width * 8, 0)
            frame = int(self.position * rate)
            self.reader.setpos(frame)
            self.offset = frame / rate
            self.check(self.api.waveOutOpen(c.byref(self.device), WAVE_MAPPER,
                                            c.byref(self.format), 0, 0, 0))
            self.eof = False
            self.update()
        except Exception:
            self.release()
            raise

    def update(self):
        if not self.playing:
            return
        if self.reader is None:
            self.release()
            return
        while self.buffers and self.buffers[0][0].flags & WHDR_DONE:
            header, _ = self.buffers[0]
            self.check(self.api.waveOutUnprepareHeader(self.device, c.byref(header), c.sizeof(header)))
            self.buffers.popleft()
        while len(self.buffers) < 4 and not self.eof:
            samples = self.reader.readframes(max(1, self.format.rate // 4))
            if not samples:
                self.eof = True
                break
            data = c.create_string_buffer(samples)
            header = WaveHeader(c.addressof(data), len(samples))
            self.check(self.api.waveOutPrepareHeader(self.device, c.byref(header), c.sizeof(header)))
            self.buffers.append((header, data))  # Keep memory alive until the device returns it.
            self.check(self.api.waveOutWrite(self.device, c.byref(header), c.sizeof(header)))
        if not self.buffers:
            self.release()
            self.position = self.duration
            return
        clock = WaveTime(TIME_BYTES)  # Drivers may return samples or milliseconds instead.
        self.check(self.api.waveOutGetPosition(self.device, c.byref(clock), c.sizeof(clock)))
        unit = {TIME_BYTES: self.format.bytes_per_second, TIME_SAMPLES: self.format.rate,
                TIME_MS: 1000}.get(clock.kind)
        if unit:
            self.position = min(self.duration, self.offset + clock.value / unit)

    def release(self):
        try:
            if self.playing:
                self.check(self.api.waveOutReset(self.device))
                while self.buffers:
                    header, _ = self.buffers[0]
                    self.check(self.api.waveOutUnprepareHeader(self.device, c.byref(header), c.sizeof(header)))
                    self.buffers.popleft()
                self.check(self.api.waveOutClose(self.device))
                self.device = w.HANDLE()
        finally:
            if self.reader is not None:
                self.reader.close()
                self.reader = None

    def close(self):
        self.release()
        self.path = None
        self.duration = self.position = 0.0
