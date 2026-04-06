# PPT 요구사항: D2NN Artifact Analysis Presentation

## 디자인
- **흰색 배경** (dark theme 아님)
- **LaTeX 수식 렌더링** (MathJax 또는 KaTeX CDN 사용)
- 그래프/차트 글씨가 잘 보이도록 충분히 크게
- 슬라이드 간 구분 명확하게

## 10 슬라이드 구성

### 1. Title
"D2NN Beam Cleanup for FSO: Data Pipeline Artifacts and Physical Analysis"

### 2. System Overview
- 시스템 다이어그램: TX → 1km → telescope → beam reducer → D2NN → lens → detector
- 유니터리 정리: $\text{CO}(HU_t, HU_v) = \text{CO}(U_t, U_v)$, $\|HU_t - HU_v\|_2 = \|U_t - U_v\|_2$

### 3. The Bug: Bilinear Beam Reducer
- 잘못된 코드: $U_{\text{new}} = \text{interp}(\text{Re}\{U\}) + j \cdot \text{interp}(\text{Im}\{U\})$
- 올바른 코드: $U_{\text{new}} = \text{Lanczos}(U_{\text{complex}})$
- 시각화: 이전 그림 `ppt_fig2_wfe_decomposition.png` 참조

### 4. Vacuum WFE Analysis
- WFE 분해: defocus 7.5% (116nm) vs higher-order 97.9% (420nm)
- 이전 vs 새 데이터: 420nm → 3.4nm (124x 개선)
- 시각화: pie chart + bar chart

### 5. D2NN Was Compensating Artifacts
- D2NN이 vacuum도 16.5% → 99.0%로 개선 (6.0x)
- 진짜 난류 효과는 95.5% → 80.0% = 15.5%p만
- 시각화: `ppt_fig3_effect_decomposition.png` 참조

### 6. Strehl Ratio Failures
- 3번 시도 모두 S > 1: Cauchy-Schwarz $|E(f)| \leq \int |U(\rho)| d\rho$
- D2NN이 amplitude를 바꾸므로 Strehl 정의 불가
- 시각화: `ppt_fig4_strehl_failures.png` 참조

### 7. Old vs New Data Comparison
- PIB@[5,10,25,50]μm 비교
- 시각화: `ppt_fig1_old_vs_new_pib.png` 참조

### 8. Clean Data Baseline
- Vacuum PIB@10μm = 95.5%, Turbulent = 80.0%
- D2NN headroom = 15.5%p
- 시각화: `ppt_fig5_clean_baseline.png` 참조

### 9. Mode Conversion Physics
- $U = \sum a_{mn} \psi_{mn}$ → D2NN → $|b_{00}|^2 \uparrow$, $\sum|b|^2 = \sum|a|^2$
- 시각화: `ppt_fig6_mode_conversion.png` 참조

### 10. Lessons & Next Steps
- 6개 numerical pitfalls
- 실험 계획

## 생성된 그림 위치
- `autoresearch/runs/ppt_fig1_old_vs_new_pib.png`
- `autoresearch/runs/ppt_fig2_wfe_decomposition.png`
- `autoresearch/runs/ppt_fig3_effect_decomposition.png`
- `autoresearch/runs/ppt_fig4_strehl_failures.png`
- `autoresearch/runs/ppt_fig5_clean_baseline.png`
- `autoresearch/runs/ppt_fig6_mode_conversion.png`

**주의**: 그림은 dark theme으로 생성되어 있으므로 흰 배경에 맞게 재생성 필요.

## 수치 데이터 (하드코딩용)
- Old vacuum WFE: 420nm, New: 3.4nm
- Old vacuum PIB@10μm: 16.5%, New: 95.5%
- Old D2NN PIB@10μm: 90.0% (artifact compensation)
- New turbulent PIB@10μm: 80.0%
- True turbulence gap: 15.5%p (95.5% - 80.0%)
- Strehl attempts: S=28, S=1.2, S=0.044(turb)>0.015(vac)
- Defocus: 7.5% of WFE, Higher-order: 97.9%
