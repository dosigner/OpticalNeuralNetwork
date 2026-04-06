from __future__ import annotations

import os
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "autoresearch"
    / "runs"
    / "0406-distance-sweep-seminar-ko"
    / "build_distance_sweep_seminar_ko.js"
)

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def _slide_texts(deck_path: Path) -> list[str]:
    texts: list[str] = []
    with zipfile.ZipFile(deck_path) as zf:
        slide_names = sorted(
            (
                name
                for name in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),
        )
        for name in slide_names:
            root = ET.fromstring(zf.read(name))
            texts.append(" ".join(node.text or "" for node in root.findall(".//a:t", NS)))
    return texts


def _slide_xml_payload(deck_path: Path) -> str:
    chunks: list[str] = []
    with zipfile.ZipFile(deck_path) as zf:
        slide_names = sorted(
            (
                name
                for name in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ),
            key=lambda value: int(re.search(r"slide(\d+)\.xml", value).group(1)),
        )
        for name in slide_names:
            chunks.append(zf.read(name).decode("utf8", errors="ignore"))
    return "\n".join(chunks)


def test_build_distance_sweep_seminar_deck(tmp_path: Path) -> None:
    output_path = tmp_path / "distance-sweep-seminar-ko.pptx"
    env = os.environ.copy()
    env["DISTANCE_SEMINAR_KO_OUT"] = str(output_path)

    subprocess.run(
        ["node", str(SCRIPT)],
        check=True,
        cwd=ROOT,
        env=env,
    )

    assert output_path.exists()

    slide_texts = _slide_texts(output_path)
    assert len(slide_texts) == 11

    joined = "\n".join(slide_texts)
    assert "Distance Sweep 결과는 무엇을 보여주는가?" in joined
    assert "지표 정의와 해석 범위" in joined
    assert "거리별 핵심 결과" in joined
    assert "한계와 다음 검증" in joined

    assert "500 test samples" not in joined
    assert "WF RMS unchanged (~442 nm)" not in joined
    assert "difference between link outage and recovery" not in joined

    xml_payload = _slide_xml_payload(output_path)
    assert "Calibri" not in xml_payload
    assert "Noto Sans CJK KR" in xml_payload
