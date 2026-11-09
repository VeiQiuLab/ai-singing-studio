"""Run a pinned UVR MDX ONNX model with local files and explicit provider evidence."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from studio import MODEL


def separate(source, model_dir, output, backend):
    import onnxruntime as ort
    import torch
    from audio_separator.separator.architectures.mdx_separator import MDXSeparator

    provider = {"directml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}[backend]
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"ONNX Runtime provider unavailable: {provider}")

    class LocalMDX(MDXSeparator):
        def load_model(self):
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = False
            options.intra_op_num_threads = 4
            options.enable_profiling = True
            options.profile_file_prefix = str(output / "uvr-profile")
            self.session = ort.InferenceSession(self.model_path, sess_options=options, providers=[provider])
            if self.session.get_providers()[0] != provider:
                raise RuntimeError(f"Could not activate {provider}: {self.session.get_providers()}")
            self.session.disable_fallback()

        def model_run(self, spectrum):
            return self.session.run(None, {"input": spectrum.cpu().numpy()})[0]

    torch.set_num_threads(4)
    logger = logging.getLogger("uvr-mdx")
    # Parameters from UVR application_data for MD5 tail hash 55657dd70583b0fedfba5f67df11d711.
    model_data = {"compensate": 1.022, "mdx_dim_f_set": 3072, "mdx_dim_t_set": 8,
                  "mdx_n_fft_scale_set": 6144, "primary_stem": "Instrumental"}
    processor = LocalMDX({
        "logger": logger, "log_level": logging.INFO, "torch_device": torch.device("cpu"),
        "torch_device_cpu": torch.device("cpu"), "onnx_execution_provider": [provider],
        "model_name": MODEL, "model_path": str(model_dir / MODEL), "model_data": model_data,
        "output_dir": str(output), "output_format": "WAV", "sample_rate": 44100,
        "normalization_threshold": 0.9, "amplification_threshold": 0.0, "use_soundfile": True,
    }, {"segment_size": 256, "hop_length": 1024, "overlap": 0.25, "batch_size": 1})
    try:
        processor.separate(str(source), {"Vocals": "vocals", "Instrumental": "instrumental"})
    finally:
        profile = Path(processor.session.end_profiling())
        try:
            events = json.loads(profile.read_text(encoding="utf-8"))
            counts = Counter(item["args"]["provider"] for item in events if item.get("args", {}).get("provider"))
            logger.info("ONNX node executions by provider: %s", dict(counts))
        finally:
            profile.unlink(missing_ok=True)
    if not counts[provider]:
        raise RuntimeError(f"No nodes were executed by {provider}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("backend", choices=("cpu", "directml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    separate(args.source, args.model_dir, args.output, args.backend)


if __name__ == "__main__":
    main()
