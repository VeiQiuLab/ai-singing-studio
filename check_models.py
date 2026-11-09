"""Run actual model inference against a locally generated synthetic audio fixture."""

import argparse
import json
import subprocess
import sys
import time
import uuid
import wave
from dataclasses import replace
from pathlib import Path

from studio import DATA, JobQueue, Request, Settings, Voice, require_file, wav_info


def make_fixture(path):
    import numpy as np

    rate = 44100
    t = np.arange(30 * rate) / rate
    notes = np.array([220, 261.626, 293.665, 329.628, 293.665, 261.626, 246.942, 220])
    fundamental = notes[(t * 2).astype(int) % len(notes)] * (1 + 0.014 * np.sin(2 * np.pi * 5.2 * t))
    phase = 2 * np.pi * np.cumsum(fundamental) / rate
    syllable = np.minimum(1, (t % 0.5) * 30) * np.minimum(1, (0.5 - t % 0.5) * 13)
    vocal = sum((np.exp(-((k * fundamental - 720) / 440)**2) +
                 0.28 * np.exp(-((k * fundamental - 1800) / 330)**2)) * np.sin(k * phase)
                for k in range(1, 13)) * 0.085 * syllable
    pad = 0.09 * np.sin(2 * np.pi * 110 * t) + 0.055 * np.sin(2 * np.pi * 164.814 * t)
    drums = 0.09 * np.exp(-17 * (t % 0.5)) * np.sin(2 * np.pi * 72 * t)
    pcm = (np.clip(np.column_stack((vocal + pad + drums, vocal + pad * 0.8 - drums)), -1, 1)
           * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(pcm.tobytes())


def synthetic_voice(folder, rvc_root, with_index):
    import numpy as np
    import torch

    sys.path.insert(0, rvc_root)
    from infer.module.models import SynthesizerTrnMs768NSFsid

    torch.set_num_threads(4)
    torch.manual_seed(267)
    config = [513, 32, 16, 32, 64, 2, 2, 3, 0.0, "1", [3, 7, 11],
              [[1, 3, 5]] * 3, [10, 8, 2, 2], 64, [16, 16, 4, 4], 1, 16, 32000]
    model = SynthesizerTrnMs768NSFsid(*config, is_half=False)
    weights = folder / "synthetic-untrained-v2.pth"
    torch.save({"version": "v2", "f0": 1, "config": config,
                "weight": {key: value for key, value in model.state_dict().items()
                           if not key.startswith("enc_q.")}}, weights)
    index_path = ""
    if with_index:
        import faiss

        index = faiss.IndexFlatL2(768)
        index.add(np.random.default_rng(267).normal(0, 0.12, (64, 768)).astype(np.float32))
        index_path = str(folder / "synthetic.index")
        faiss.write_index(index, index_path)
    return Voice.register("未训练合成测试模型", "确定种子随机生成，无训练数据", weights, index_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "directml"), default="auto")
    parser.add_argument("--separate", action="store_true", help="Only validate UVR separation.")
    parser.add_argument("--index", action="store_true", help="Also exercise a synthetic FAISS index.")
    args = parser.parse_args()
    settings = replace(Settings.load(), uvr_backend=args.backend)
    engine = require_file(Path(settings.rvc_python or settings.audio_python), ".exe").resolve()
    if Path(sys.executable).resolve() != engine:
        return subprocess.call([str(engine), "-B", str(Path(__file__).resolve()), *sys.argv[1:]])
    folder = DATA / "model-check" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    sample = folder / "synthetic-song.wav"
    make_fixture(sample)
    voice = None if args.separate else synthetic_voice(folder, settings.rvc_root, args.index)
    jobs = JobQueue(folder)
    try:
        request = Request("separate" if args.separate else "cover", sample, voice=voice)
        job = jobs.submit(request, settings)
        print(f"合成输入测试，日志目录 {job.folder}", flush=True)
        try:
            while job.snapshot()[0] not in {"done", "failed", "cancelled"}:
                time.sleep(0.2)
        except KeyboardInterrupt:
            job.stop()
            return 130
        state, _, detail = job.snapshot()
        if state != "done":
            raise RuntimeError(f"{detail} 任务目录：{job.folder}")
        if wav_info(job.result) != (30.0, 44100, 2, 2):
            raise RuntimeError("The synthetic fixture did not produce the expected 30-second PCM WAV.")
        print(json.dumps(json.loads((job.folder / "result.json").read_text(encoding="utf-8")),
                         ensure_ascii=False, indent=2))
        return 0
    finally:
        jobs.close()
        jobs.worker.join()


if __name__ == "__main__":
    raise SystemExit(main())
