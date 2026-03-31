---
name: d2nn-lab-interpreter
description: Interpret D2NN experiment directories into a canonical registry, task-wise comparison, and Korean-first evidence-based reports. Use when comparing runs under `/root/dj/D2NN`, especially `kim2026/runs`, and when proposing next experiments from recorded metrics.
---

# D2NN Lab Interpreter

Use this skill to turn scattered D2NN run folders into a reproducible interpretation flow.

## Defaults

- Default root: `/root/dj/D2NN`
- Primary use case: beam-cleanup and FD2NN run interpretation under `/root/dj/D2NN/kim2026/runs`
- Default language: Korean
- Default style: 표 우선, 명사형 어미, 근거 우선

## Workflow

1. Read only the references you need:
   - [references/scope-and-task-families.md](references/scope-and-task-families.md)
   - [references/canonical-registry-schema.md](references/canonical-registry-schema.md)
   - [references/root-cause-taxonomy.md](references/root-cause-taxonomy.md)
   - [references/physics-modes.md](references/physics-modes.md)
2. Build a canonical registry when the user wants structured evidence:

```bash
/root/dj/D2NN/miniconda3/envs/d2nn/bin/python /root/dj/D2NN/skills/d2nn-lab-interpreter/scripts/build_registry.py --root /root/dj/D2NN/kim2026/runs --output /tmp/d2nn_registry.json
```

3. Run the analyzer when the user wants Pareto/frontier, elimination, or next-experiment suggestions:

```bash
/root/dj/D2NN/miniconda3/envs/d2nn/bin/python /root/dj/D2NN/skills/d2nn-lab-interpreter/scripts/analyze_registry.py --registry /tmp/d2nn_registry.json --output /tmp/d2nn_analysis.json
```

4. Quote concrete metric values and file paths when making conclusions.
5. Separate evidence from inference. If `config.yaml` or `history.json` is missing, state that explicitly.

## Required Output Template

문장 종결은 명사형으로 유지하고, 표를 먼저 제시합니다.

```md
## 🧭 한 줄 결론
한 줄 핵심 판단

## 📊 Pareto 비교표
| task family | frontier run | 핵심 성능 | 보조 지표 | 판단 |
|---|---|---:|---:|---|

표 아래 2~4줄 요약

## 🚫 탈락 원인 비교표
| run | 탈락 라벨 | 우세 run | 근거 | 해석 |
|---|---|---|---|---|

표 아래 2~4줄 요약

## 🔬 물리 해석
| 구분 | 관찰 | 물리 해석 | 확신 수준 |
|---|---|---|---|

표 아래 2~4줄 요약

## 🧪 우선 실험 표
| 우선순위 | 실험 대상 | 이유 | 기대 효과 | 필요 증거 |
|---|---|---|---|---|

## ⚠️ 해석 한계
증거 부족 항목과 보수적 해석
```

## Invocation

Use `Codex` with:

```text
Use $d2nn-lab-interpreter to analyze /root/dj/D2NN/kim2026/runs and explain what to optimize first.
```

## Notes

- Prefer `evaluation.json`, `test_metrics.json`, `history.json`, `config.yaml` as primary evidence.
- Do not claim a genuine Pareto optimum when complexity or fabrication metadata is missing.
- Treat missing config/history as an interpretation limitation, not as neutral data.
