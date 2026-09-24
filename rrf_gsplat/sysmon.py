"""System resources while experiments run, for the viewer's live tab (viewer.py /api/live/sys).

Every --every seconds appends one line to output/rrf/sysmon.jsonl: total CPU %, the busiest processes, RAM in use
and committed (GB), GPU utilisation and memory (nvidia-smi), disk read / write MB/s. Written to find out what makes
the desktop lag during a run (Ke, 2026-09-24: the display is on the integrated GPU, so it is not the 3060 being
busy). The file is trimmed to its last 24 h when it grows past that.

    python rrf_gsplat/sysmon.py [--every 5]          (runs until killed)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output", "rrf", "sysmon.jsonl")


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        u, m = r.stdout.strip().split(",")[:2]
        return float(u), float(m) / 1024
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=float, default=5.0)
    a = ap.parse_args()
    ncpu = psutil.cpu_count()
    psutil.cpu_percent(None)
    procs = {}
    disk0, t0 = psutil.disk_io_counters(), time.time()
    lines = 0
    while True:
        time.sleep(a.every)
        now = time.time()
        # per-process CPU since the last sample (the first sample of a process reads 0)
        top = []
        for p in psutil.process_iter(["pid", "name"]):
            if p.pid == 0:
                continue                                   # "System Idle Process": idle time, not a process
            try:
                if p.pid not in procs:
                    procs[p.pid] = p
                    p.cpu_percent(None)
                    continue
                c = procs[p.pid].cpu_percent(None) / ncpu
                if c >= 1.0:
                    top.append((round(c, 1), p.info["name"]))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                procs.pop(p.pid, None)
        top.sort(reverse=True)
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        d = psutil.disk_io_counters()
        dt = now - t0
        gu, gm = gpu()
        rec = {"t": round(now, 1), "cpu": psutil.cpu_percent(None), "top": top[:5],
               "ram_gb": round((vm.total - vm.available) / 2**30, 2), "ram_total_gb": round(vm.total / 2**30, 1),
               "commit_gb": round((vm.total - vm.available + sw.used) / 2**30, 2),
               "gpu": gu, "gpu_mem_gb": None if gm is None else round(gm, 2),
               "disk_r_mbs": round((d.read_bytes - disk0.read_bytes) / dt / 2**20, 1),
               "disk_w_mbs": round((d.write_bytes - disk0.write_bytes) / dt / 2**20, 1)}
        disk0, t0 = d, now
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        lines += 1
        if lines % 720 == 0 and os.path.getsize(OUT) > 20 * 2**20:          # keep the last 24 h
            keep = [l for l in open(OUT) if json.loads(l)["t"] > now - 86400]
            with open(OUT + ".tmp", "w") as f:
                f.writelines(keep)
            os.replace(OUT + ".tmp", OUT)


if __name__ == "__main__":
    main()
