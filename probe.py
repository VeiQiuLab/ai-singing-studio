"""Read-only hardware inventory and opt-in inference smoke tests."""

import argparse
import ctypes
import importlib.util
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path


def inventory():
    info = {
        "windows": platform.version(), "python": sys.version.split()[0],
        "processor": platform.processor(), "ffmpeg": shutil.which("ffmpeg"),
        "hip_tools": {tool: shutil.which(tool) for tool in ("rocminfo", "hipInfo", "amd-smi")},
        "modules": {name: importlib.util.find_spec(name) is not None for name in (
            "tkinter", "torch", "torch_directml", "onnx", "onnxruntime", "audio_separator")},
        "gpus_registry": [],
    }
    if sys.platform != "win32":
        return info
    import winreg

    # MEMORYSTATUSEX uses two DWORDs followed by seven unsigned 64-bit fields.
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64), ("remaining", ctypes.c_uint64 * 6)]

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        info["ram_gib"] = round(memory.total_physical / 2**30, 3)
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\Video") as adapters:
            for i in range(winreg.QueryInfoKey(adapters)[0]):
                try:
                    with winreg.OpenKey(adapters, winreg.EnumKey(adapters, i) + r"\0000") as adapter:
                        name = winreg.QueryValueEx(adapter, "DriverDesc")[0]
                        driver = winreg.QueryValueEx(adapter, "DriverVersion")[0]
                        try:
                            memory_bytes = winreg.QueryValueEx(adapter, "HardwareInformation.qwMemorySize")[0]
                        except OSError:
                            memory_bytes = None
                        if isinstance(memory_bytes, bytes):
                            memory_bytes = int.from_bytes(memory_bytes, "little")
                        info["gpus_registry"].append({
                            "name": name, "driver": driver,
                            "vram_gib": round(memory_bytes / 2**30, 3) if memory_bytes else None})
                except OSError:
                    continue
    except OSError as exc:
        info["gpu_inventory_error"] = str(exc)
    return info


def onnx_check(provider):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper as h

    available = ort.get_available_providers()
    if provider not in available:
        return {"status": "unavailable", "available": available, "ort": ort.__version__}
    size = 64
    inputs = [h.make_tensor_value_info(name, TensorProto.FLOAT, [size, size]) for name in ("a", "b")]
    outputs = [h.make_tensor_value_info("y", TensorProto.FLOAT, [size, size])]
    graph = h.make_graph([h.make_node("MatMul", ["a", "b"], ["y"])], "probe", inputs, outputs)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=8)
    with tempfile.TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(tmp) / "probe")
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        if provider == "DmlExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        start = time.monotonic()
        session = ort.InferenceSession(model.SerializeToString(), options, providers=[provider])
        session.disable_fallback()
        values = np.arange(size * size, dtype=np.float32).reshape(size, size) % 7
        result = session.run(None, {"a": values, "b": values})[0]
        events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
        used = sorted({item["args"]["provider"] for item in events
                       if item.get("args", {}).get("provider")})
    correct = np.allclose(result, values @ values)
    return {"status": "pass" if correct and used == [provider] else "fail",
            "executed_on": used, "correct": bool(correct), "available": available,
            "ort": ort.__version__, "seconds": round(time.monotonic() - start, 3)}


def torch_check(backend):
    if importlib.util.find_spec("torch") is None:
        return {"status": "unavailable", "reason": "torch missing"}
    import torch

    torch.set_num_threads(2)
    hip = getattr(torch.version, "hip", None)
    device_name = ""
    if backend == "rocm":
        if not hip or not torch.cuda.is_available():
            return {"status": "unavailable", "torch": torch.__version__, "hip": hip}
        device, device_name = "cuda:0", torch.cuda.get_device_name(0)
    elif backend == "torch-directml":
        if importlib.util.find_spec("torch_directml") is None:
            return {"status": "unavailable", "reason": "torch_directml missing"}
        import torch_directml

        device, device_name = torch_directml.device(), torch_directml.device_name(0)
    else:
        device = "cpu"
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    result = (x.to(device) @ x.to(device)).cpu()
    return {"status": "pass" if torch.allclose(result, x @ x) else "fail",
            "torch": torch.__version__, "device": str(device), "device_name": device_name, "hip": hip}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("inventory", "cpu", "directml", "torch-cpu",
                                            "torch-directml", "rocm"), nargs="?", default="inventory")
    args = parser.parse_args()
    try:
        if args.backend == "inventory":
            result = inventory()
        elif args.backend in {"cpu", "directml"}:
            provider = {"cpu": "CPUExecutionProvider", "directml": "DmlExecutionProvider"}[args.backend]
            result = onnx_check(provider)
        else:
            result = torch_check(args.backend)
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"pass": 0, "unavailable": 2}.get(result.get("status", "pass"), 1)


if __name__ == "__main__":
    raise SystemExit(main())
