"""ACE-Step Turbo DiT CPU worker for original music jobs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from studio import DATA, require_file

ACE_REVISION = "ca1e85fe9430179831e6bc6be790c332190a3866"
ACE_WEIGHTS_REVISION = "eb84216a8ece2d56388e401efc28061a0c30a209"
ACE_FILES = {
    "acestep-v15-turbo/model.safetensors": "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7",
    "acestep-v15-turbo/silence_latent.pt": "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b",
    "Qwen3-Embedding-0.6B/model.safetensors": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
    "vae/diffusion_pytorch_model.safetensors": "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0",
}
ACE_CHECKPOINT = "acestep-v15-turbo"
SEED = 267


def generate(root, request, output):
    checkpoints = root / "checkpoints"
    for file in ACE_FILES:
        require_file(checkpoints / file)
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ACESTEP_INIT_LLM="false",
                      ACESTEP_CHECKPOINTS_DIR=str(checkpoints), ACESTEP_VAE_CHECKPOINT="official",
                      HF_HOME=str(DATA / "cache" / "ace-hf"),
                      TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")

    import numpy as np
    import psutil
    import soundfile as sf
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    class LocalDiT(AceStepHandler):
        def _ensure_models_present(self, **kwargs):
            # The pinned upstream precheck expects a 1.7B LM even for DiT-only use.
            # Only the exact locally prepared Turbo, text encoder and VAE are used here.
            if kwargs["config_path"] != ACE_CHECKPOINT:
                return "This worker supports the pinned Turbo DiT only.", False
            return None

    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    start = time.monotonic()
    handler = LocalDiT()
    status, ready = handler.initialize_service(str(root), ACE_CHECKPOINT, device="cpu",
                                               compile_model=False, use_flash_attention=False,
                                               use_mlx_dit=False, quantization=None)
    print(f"ACE_STEP initialization_seconds={time.monotonic() - start:.3f}\n{status}", flush=True)
    if not ready:
        raise RuntimeError(status)
    caption = "Clear lead singing from the start, audible words, vocals in front of the backing. "
    params = GenerationParams(caption=caption + request["caption"], lyrics=request["lyrics"], instrumental=False,
                              vocal_language=request["language"], duration=request["seconds"],
                              seed=SEED, inference_steps=8, thinking=False, use_cot_metas=False,
                              use_cot_caption=False, use_cot_lyrics=False, use_cot_language=False)
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[SEED], audio_format="wav")
    folder = output.parent / "ace"
    folder.mkdir()
    start = time.monotonic()
    result = generate_music(handler, None, params, config, str(folder))
    if not result.success or not result.audios or not result.audios[0]["path"]:
        raise RuntimeError(result.error or result.status_message)
    path = require_file(Path(result.audios[0]["path"]), ".wav")
    audio, rate = sf.read(path)
    if len(audio) == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
        raise ValueError("ACE-Step returned empty or invalid audio.")
    path.replace(output)
    peak = psutil.Process().memory_info().peak_wset / 2**30
    print(f"ACE_STEP generated_seconds={len(audio) / rate:.3f} sample_rate={rate} "
          f"generation_seconds={time.monotonic() - start:.3f} peak_working_set_gib={peak:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), args.output.resolve())


if __name__ == "__main__":
    main()
