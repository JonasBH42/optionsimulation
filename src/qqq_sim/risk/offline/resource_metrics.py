"""Process memory measurement without a third-party runtime dependency."""
import os


def process_memory(pid=None):
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                      *[(name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]]
        record=Counters();record.cb=ctypes.sizeof(record)
        function=ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        function.restype=wintypes.BOOL
        handle=wintypes.HANDLE(-1)
        external=pid is not None and pid!=os.getpid()
        if external:
            opener=ctypes.windll.kernel32.OpenProcess
            opener.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            opener.restype=wintypes.HANDLE
            handle=opener(0x410,False,pid)
        try:
            if handle and function(handle,ctypes.byref(record),record.cb):
                return {"pid":pid or os.getpid(),"rss_bytes":record.WorkingSetSize,"peak_rss_bytes":record.PeakWorkingSetSize,"measurement":"WINDOWS_PROCESS_MEMORY_COUNTERS"}
        finally:
            if external and handle:
                closer=ctypes.windll.kernel32.CloseHandle
                closer.argtypes=[wintypes.HANDLE]
                closer(handle)
    else:
        if pid is not None and pid!=os.getpid():
            return {"pid":pid,"rss_bytes":None,"peak_rss_bytes":None,"measurement":"EXTERNAL_PROCESS_PEAK_UNAVAILABLE"}
        import resource
        import sys
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"pid":os.getpid(),"rss_bytes":None,"peak_rss_bytes":int(peak*(1 if sys.platform=="darwin" else 1024)),"measurement":"GETRUSAGE_PROCESS_PEAK"}
    return {"pid":pid or os.getpid(),"rss_bytes":None,"peak_rss_bytes":None,"measurement":"UNAVAILABLE"}
