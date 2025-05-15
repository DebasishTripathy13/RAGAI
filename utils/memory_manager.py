import os
import logging

class MemoryManager:
    psutil_available = False
    try:
        import psutil
        psutil_available = True
    except ImportError:
        logging.warning("psutil library not found. Memory monitoring will be disabled.")

    @staticmethod
    def get_memory_usage():
        if not MemoryManager.psutil_available: 
            return {"rss": 0, "vms": 0, "percent": 0, "available_gb": 0, "total_gb": 0}
        
        process = MemoryManager.psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual_mem = MemoryManager.psutil.virtual_memory()
        return {
            "rss": mem_info.rss / (1024 * 1024), 
            "vms": mem_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_gb": virtual_mem.available / (1024**3),
            "total_gb": virtual_mem.total / (1024**3)
        }

    @staticmethod
    def log_memory_usage(context=""):
        if not MemoryManager.psutil_available: return
        mem = MemoryManager.get_memory_usage()
        logging.info(f"Memory {context}: RSS={mem['rss']:.1f}MB, %Proc={mem['percent']:.1f}%, SysAvail={mem['available_gb']:.1f}GB")

    @staticmethod
    def check_memory_pressure():
        if not MemoryManager.psutil_available: return False
        mem = MemoryManager.psutil.virtual_memory()
        if mem.percent > 90: return True  # System memory over 90%
        return False