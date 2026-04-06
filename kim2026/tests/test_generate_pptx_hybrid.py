from __future__ import annotations

import os
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "autoresearch" / "runs" / "generate_pptx_hybrid.js"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def _slide_texts(deck_path: Path) -> list[str]:
    texts: list[str] = []
    with zipfile.ZipFile(deck_path) as zf:
        slide_names = sorted(
            name
            for name in zf.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        for name in slide_names:
            root = ET.fromstring(zf.read(name))
            texts.append(" ".join(node.text or "" for node in root.findall(".//a:t", NS)))
    return texts


def test_generate_hybrid_pptx_writes_expected_slides(tmp_path: Path) -> None:
    output_path = tmp_path / "hybrid-analysis.pptx"
    env = os.environ.copy()
    env["HYBRID_PPT_OUT"] = str(output_path)

    subprocess.run(
        ["node", str(SCRIPT)],
        check=True,
        cwd=ROOT,
        env=env,
    )

    assert output_path.exists()

    slide_texts = _slide_texts(output_path)
    assert len(slide_texts) == 17

    joined = "\n".join(slide_texts)
    assert "D2NN Beam Cleanup for FSO" in joined
    assert "Research Chronology Map" in joined
    assert "Physics Hardening and Residual Failures" in joined
    assert "Appendix A: Equation Sheet" in joined
