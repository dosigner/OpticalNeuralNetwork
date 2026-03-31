---
name: d2nn-lab-interpreter
description: Claude wrapper for the shared D2NN lab interpreter skill under `skills/d2nn-lab-interpreter`.
---

# D2NN Lab Interpreter

Use the shared skill at `../../../skills/d2nn-lab-interpreter`.

## Required References

- `../../../skills/d2nn-lab-interpreter/references/scope-and-task-families.md`
- `../../../skills/d2nn-lab-interpreter/references/canonical-registry-schema.md`
- `../../../skills/d2nn-lab-interpreter/references/root-cause-taxonomy.md`
- `../../../skills/d2nn-lab-interpreter/references/physics-modes.md`

## Script Entry Points

- `/root/dj/D2NN/skills/d2nn-lab-interpreter/scripts/build_registry.py`
- `/root/dj/D2NN/skills/d2nn-lab-interpreter/scripts/analyze_registry.py`

## Output Style

- `🧭 한 줄 결론`
- `📊 Pareto 비교표`
- `🚫 탈락 원인 비교표`
- `🔬 물리 해석`
- `🧪 우선 실험 표`
- `⚠️ 해석 한계`

문장 종결은 명사형으로 유지하고, 표를 먼저 제시합니다.

## Invocation

```text
d2nn-lab-interpreter를 사용해서 /root/dj/D2NN/kim2026/runs를 해석하고 한국어 표 중심 보고서를 작성해줘.
```
