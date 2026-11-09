"""Keep Windows inference subprocesses and their children in one job object."""

import ctypes as ct
from ctypes import wintypes as w

kernel = ct.WinDLL("kernel32", use_last_error=True)
kernel.CreateJobObjectW.argtypes = (ct.c_void_p, w.LPCWSTR)
kernel.CreateJobObjectW.restype = w.HANDLE
kernel.SetInformationJobObject.argtypes = (w.HANDLE, ct.c_int, ct.c_void_p, w.DWORD)
kernel.SetInformationJobObject.restype = w.BOOL
kernel.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
kernel.AssignProcessToJobObject.restype = w.BOOL
kernel.CloseHandle.argtypes = (w.HANDLE,)
kernel.CloseHandle.restype = w.BOOL


class BasicLimits(ct.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ct.c_int64), ("PerJobUserTimeLimit", ct.c_int64),
        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ct.c_size_t),
        ("MaximumWorkingSetSize", ct.c_size_t), ("ActiveProcessLimit", w.DWORD),
        ("Affinity", ct.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD),
    ]


class ExtendedLimits(ct.Structure):
    _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ct.c_uint64 * 6),
                ("ProcessMemoryLimit", ct.c_size_t), ("JobMemoryLimit", ct.c_size_t),
                ("PeakProcessMemoryUsed", ct.c_size_t), ("PeakJobMemoryUsed", ct.c_size_t)]


class ProcessTree:
    def __enter__(self):
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ct.WinError(ct.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ct.byref(limits), ct.sizeof(limits)):
            error = ct.WinError(ct.get_last_error())
            kernel.CloseHandle(self.handle)
            raise error
        return self

    def attach(self, process):
        if not kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            error = ct.WinError(ct.get_last_error())
            if process.poll() is None:
                raise error

    def __exit__(self, *_):
        kernel.CloseHandle(self.handle)
