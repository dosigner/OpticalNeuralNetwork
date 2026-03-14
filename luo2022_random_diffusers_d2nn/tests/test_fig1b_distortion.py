"""Tests for Fig 1b labeling."""

from __future__ import annotations

from luo2022_d2nn.figures.fig1b_distortion import _build_row_labels


def test_build_row_labels_includes_korean_descriptions_and_diffuser_count():
    labels = _build_row_labels(training_diffusers=20)

    assert labels == [
        "원본 타깃\n(MNIST 숫자)",
        "비학습 기준선\n(디퓨저만 통과)",
        "비학습 기준선\n(디퓨저 + 렌즈)",
        "학습된 D2NN 복원\n(학습 diffuser 수: n=20)",
    ]
