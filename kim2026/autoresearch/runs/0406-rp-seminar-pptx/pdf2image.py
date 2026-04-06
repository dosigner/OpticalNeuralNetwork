import os
import subprocess
from pathlib import Path


def pdfinfo_from_path(pdf_path):
    out = subprocess.check_output(["pdfinfo", pdf_path], text=True)
    info = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def convert_from_path(
    pdf_path,
    dpi=200,
    fmt="png",
    thread_count=1,
    output_folder=None,
    paths_only=False,
    output_file="slide",
):
    if fmt.lower() != "png":
      raise ValueError("Only png output is supported by the local shim.")

    output_dir = Path(output_folder or ".").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / output_file
    cmd = [
        "pdftoppm",
        "-r",
        str(dpi),
        "-png",
        pdf_path,
        str(prefix),
    ]
    subprocess.check_call(cmd)
    paths = sorted(str(p) for p in output_dir.glob(f"{output_file}-*.png"))
    if not paths:
        raise RuntimeError(f"No rendered images produced for {pdf_path}")
    if paths_only:
        return paths
    return paths
