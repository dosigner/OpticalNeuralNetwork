const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "DJ, ADD";
pres.title = "GPU 가속 D2NN 기반 대기 난류 보정: 타당성 연구";

// --- Color palette: Ocean Gradient (deep science feel) ---
const C = {
  navy: "0A1628",
  deepBlue: "0F2B46",
  teal: "0D6E8A",
  accent: "14B8A6",
  white: "FFFFFF",
  offWhite: "F0F4F8",
  lightGray: "E2E8F0",
  midGray: "94A3B8",
  darkText: "1E293B",
  red: "DC2626",
  green: "059669",
  amber: "D97706",
};

const FONTS = { header: "Georgia", body: "Calibri" };
const FIG = path.join(__dirname, "figures");
const DIAG = path.join(__dirname, "diagrams");

function makeShadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.12 };
}

// ============================================================
// SLIDE 1: Title
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  // Top accent bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
  s.addText("GPU 가속 D2NN 기반\n대기 난류 보정", {
    x: 0.8, y: 1.0, w: 8.4, h: 2.4,
    fontFace: FONTS.header, fontSize: 40, color: C.white, bold: true,
    lineSpacingMultiple: 1.2,
  });
  s.addText("타당성 연구 및 실험 보고서", {
    x: 0.8, y: 3.3, w: 8.4, h: 0.6,
    fontFace: FONTS.body, fontSize: 22, color: C.accent, italic: true,
  });
  // Bottom info bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.5, w: 10, h: 1.125, fill: { color: C.deepBlue } });
  s.addText([
    { text: "DJ  |  국방과학연구소(ADD) 광네트워크연구실  |  2026-03-23", options: { breakLine: true, color: C.midGray, fontSize: 13 } },
    { text: "PyTorch + CUDA (NVIDIA A100)  |  λ = 1550 nm  |  1024×1024 격자", options: { color: C.midGray, fontSize: 11 } },
  ], { x: 0.8, y: 4.6, w: 8.4, h: 0.7, fontFace: FONTS.body });
}

// ============================================================
// SLIDE 2: Executive Summary
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("Executive Summary", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontFace: FONTS.header, fontSize: 36, color: C.darkText, bold: true, margin: 0,
  });

  // 3 key finding cards
  const cards = [
    { title: "목적", body: "대기 난류로 인한 FSO 빔 왜곡을\n수동 D2NN으로 보정 가능한지 검증", color: C.teal },
    { title: "핵심 발견", body: "5개 위상 전용 회절층 D2NN은\n무작위 대기 난류 보정에 실패", color: C.red },
    { title: "결론", body: "수동 D2NN은 대기 난류 보정에\n근본적으로 부적합", color: C.amber },
  ];
  cards.forEach((c, i) => {
    const cx = 0.5 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.3, w: 2.8, h: 2.4, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.3, w: 2.8, h: 0.06, fill: { color: c.color } });
    s.addText(c.title, { x: cx + 0.2, y: 1.5, w: 2.4, h: 0.5, fontFace: FONTS.header, fontSize: 18, color: c.color, bold: true, margin: 0 });
    s.addText(c.body, { x: cx + 0.2, y: 2.0, w: 2.4, h: 1.5, fontFace: FONTS.body, fontSize: 13, color: C.darkText, margin: 0 });
  });

  // Recommendation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.85, w: 9.0, h: 1.2, fill: { color: C.deepBlue } });
  s.addText("권고: FSO 난류 보정에는 실시간 파면 감지와 동적 보정이 결합된 적응 광학(AO) 시스템이 필수적", {
    x: 0.8, y: 3.95, w: 8.4, h: 0.5,
    fontFace: FONTS.body, fontSize: 14, color: C.white, bold: true, margin: 0,
  });
  s.addText("D2NN은 결정론적 고정 수차(렌즈 수차, 정적 열 왜곡 등)에는 적용 가능하나, 무작위 시변 난류에는 적용 불가", {
    x: 0.8, y: 4.45, w: 8.4, h: 0.5,
    fontFace: FONTS.body, fontSize: 12, color: C.midGray, margin: 0,
  });
}

// ============================================================
// SLIDE 3: FSO 빔 전파 시뮬레이터
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("1. FSO 빔 전파 시뮬레이터", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontFace: FONTS.header, fontSize: 32, color: C.darkText, bold: true, margin: 0,
  });

  // Left: architecture info
  s.addText("GPU 가속 시뮬레이터 (kim2026.fso)", {
    x: 0.5, y: 1.1, w: 4.5, h: 0.4,
    fontFace: FONTS.body, fontSize: 16, color: C.teal, bold: true, margin: 0,
  });
  s.addText([
    { text: "9개 모듈, 2,138 LOC", options: { bullet: true, breakLine: true } },
    { text: "30개 자동화 물리 검증 테스트", options: { bullet: true, breakLine: true } },
    { text: "분할 단계 각 스펙트럼법 (Split-step ASM)", options: { bullet: true, breakLine: true } },
    { text: "FFT + 3단계 서브하모닉 위상 스크린", options: { bullet: true, breakLine: true } },
    { text: "complex128 정밀도 (수치 안정성)", options: { bullet: true } },
  ], { x: 0.7, y: 1.6, w: 4.3, h: 2.5, fontFace: FONTS.body, fontSize: 13, color: C.darkText, margin: 0 });

  // Right: parameter table
  const paramRows = [
    [
      { text: "매개변수", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "값", options: { bold: true, color: C.white, fill: { color: C.teal } } },
    ],
    ["λ", "1550 nm"],
    ["전파 거리", "1 km"],
    ["Cn²", "1×10⁻¹⁴ m⁻²/³"],
    ["빔 발산각", "0.6 mrad"],
    ["수신면 빔 반경", "30 cm"],
    ["r₀ (구면파)", "14.1 cm"],
    ["난류 실현 수", "200"],
  ];
  s.addTable(paramRows, {
    x: 5.2, y: 1.1, w: 4.3,
    fontSize: 11, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [2.0, 2.3],
    autoPage: false,
  });

  // Bottom: verification results
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.0, w: 9.0, h: 1.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("물리 검증 통과:", { x: 0.7, y: 4.05, w: 2, h: 0.35, fontFace: FONTS.body, fontSize: 12, color: C.teal, bold: true, margin: 0 });
  s.addText("FT 왕복 오차 1.9×10⁻¹⁵  |  Parseval 정리 4.0×10⁻¹⁶  |  구조 함수 7/8 합격  |  결맞음 인자 13% 차이", {
    x: 0.7, y: 4.35, w: 8.5, h: 0.6,
    fontFace: FONTS.body, fontSize: 11, color: C.darkText, margin: 0,
  });
}

// ============================================================
// SLIDE 4: FSO 난류 데이터 시각화
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("FSO 난류 데이터 몽타주", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 28, color: C.white, bold: true, margin: 0,
  });
  // Figure
  s.addImage({ path: path.resolve(FIG, "fig1_fso_turbulence.png"), x: 0.3, y: 0.9, w: 9.4, h: 4.2 });
  s.addText("진공 가우시안 빔 vs 개별 난류 실현 — w/r₀ ≈ 2.1, ~4.4개 독립 결맞음 셀", {
    x: 0.5, y: 5.15, w: 9, h: 0.4,
    fontFace: FONTS.body, fontSize: 11, color: C.midGray, italic: true, margin: 0,
  });
}

// ============================================================
// SLIDE 5: D2NN 아키텍처
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("2. D2NN 아키텍처", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontFace: FONTS.header, fontSize: 32, color: C.darkText, bold: true, margin: 0,
  });

  // Left column: model description
  s.addText("BeamCleanupD2NN", {
    x: 0.5, y: 1.1, w: 4.5, h: 0.4,
    fontFace: FONTS.body, fontSize: 18, color: C.teal, bold: true, margin: 0,
  });
  s.addText([
    { text: "5개 위상 전용(phase-only) 회절층", options: { bullet: true, breakLine: true } },
    { text: "위상 마스크: φ ∈ [0, 2π] 래핑", options: { bullet: true, breakLine: true } },
    { text: "에너지 보존: |Uout|² = |Uin|²", options: { bullet: true, breakLine: true } },
    { text: "zeros 초기화 (항등 연산 시작)", options: { bullet: true, breakLine: true } },
    { text: "float64 정밀도 위상 계산", options: { bullet: true } },
  ], { x: 0.7, y: 1.6, w: 4.3, h: 2.0, fontFace: FONTS.body, fontSize: 13, color: C.darkText, margin: 0 });

  // Key equation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.7, w: 4.5, h: 0.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("핵심 제약: 단일 층에서 진폭 변조 불가\n|U · e^(iφ)|² = |U|²", {
    x: 0.7, y: 3.75, w: 4.1, h: 0.7,
    fontFace: "Consolas", fontSize: 12, color: C.red, margin: 0,
  });

  // Right: δ/λ ratio explanation
  s.addText("δ/λ 비율의 물리적 의미", {
    x: 5.3, y: 1.1, w: 4.2, h: 0.4,
    fontFace: FONTS.body, fontSize: 16, color: C.teal, bold: true, margin: 0,
  });
  s.addText("회절 결합 범위:\nN_coupling = λz / (2δ²)", {
    x: 5.3, y: 1.5, w: 4.2, h: 0.7,
    fontFace: "Consolas", fontSize: 12, color: C.darkText, margin: 0,
  });

  const regimeRows = [
    [
      { text: "체제", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "δ/λ", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "결과", options: { bold: true, color: C.white, fill: { color: C.teal } } },
    ],
    ["메타표면", "< 10", "정보 처리 가능"],
    ["물리적", "≫ 10", "항등 연산 축퇴"],
  ];
  s.addTable(regimeRows, {
    x: 5.3, y: 2.4, w: 4.2,
    fontSize: 11, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [1.2, 1.0, 2.0],
    autoPage: false,
  });

  // Scaling table
  const scaleRows = [
    [
      { text: "δ", options: { bold: true, color: C.white, fill: { color: C.deepBlue } } },
      { text: "δ/λ", options: { bold: true, color: C.white, fill: { color: C.deepBlue } } },
      { text: "3px 결합 z", options: { bold: true, color: C.white, fill: { color: C.deepBlue } } },
      { text: "총 깊이", options: { bold: true, color: C.white, fill: { color: C.deepBlue } } },
    ],
    ["10 μm", "6.5", "387 μm", "1.94 mm"],
    ["150 μm", "97", "9 cm", "45 cm"],
    ["2 mm", "1,290", "17 m", "83 m (!)"],
  ];
  s.addTable(scaleRows, {
    x: 5.3, y: 3.4, w: 4.2,
    fontSize: 10, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [0.9, 0.8, 1.2, 1.3],
    autoPage: false,
  });
}

// ============================================================
// SLIDE 6: 층간 간격 스윕 (δ=2mm)
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("3. 층간 간격 스윕 (δ = 2 mm, δ/λ = 1,290)", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontFace: FONTS.header, fontSize: 28, color: C.darkText, bold: true, margin: 0,
  });

  // Results table
  const sweepRows = [
    [
      { text: "층간 간격 z", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "N_coupling", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "훈련 개선", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "검증 개선", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "판정", options: { bold: true, color: C.white, fill: { color: C.teal } } },
    ],
    ["0.01 m", "0.001 px", "~0%", "~0%", { text: "실패", options: { color: C.red, bold: true } }],
    ["10 m", "1.0 px", "~1%", "~1%", { text: "실패", options: { color: C.red, bold: true } }],
    ["50 m", "4.8 px", "~2%", "~2%", { text: "실패", options: { color: C.red, bold: true } }],
  ];
  s.addTable(sweepRows, {
    x: 0.5, y: 1.1, w: 9.0,
    fontSize: 13, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [2.0, 1.8, 1.7, 1.7, 1.8],
    autoPage: false,
  });

  // Figure
  s.addImage({ path: path.resolve(FIG, "fig2_spacing_sweep.png"), x: 0.5, y: 2.5, w: 9.0, h: 2.8 });
  s.addText("50 m 간격에서도 ~5 픽셀 결합만 달성 → 의미있는 공간 정보 처리 불가", {
    x: 0.5, y: 5.2, w: 9, h: 0.35,
    fontFace: FONTS.body, fontSize: 11, color: C.red, italic: true, margin: 0,
  });
}

// ============================================================
// SLIDE 7: 핵심 증거 — 스케일 비교
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("3-1. 핵심 증거: D2NN 학습은 픽셀 결합에 비례", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 26, color: C.accent, bold: true, margin: 0,
  });

  s.addImage({ path: path.resolve(FIG, "fig3_scale_comparison.png"), x: 0.3, y: 0.9, w: 9.4, h: 3.5 });

  // Callout boxes
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.5, w: 4.2, h: 0.9, fill: { color: C.deepBlue } });
  s.addText("δ = 10 μm (메타표면)\n→ 11% 극적 학습, 여전히 상승 중", {
    x: 0.7, y: 4.55, w: 3.8, h: 0.8,
    fontFace: FONTS.body, fontSize: 12, color: C.accent, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 4.5, w: 4.2, h: 0.9, fill: { color: C.deepBlue } });
  s.addText("δ = 2 mm (물리적)\n→ 최대 2%, 거의 평평", {
    x: 5.5, y: 4.55, w: 3.8, h: 0.8,
    fontFace: FONTS.body, fontSize: 12, color: C.midGray, margin: 0,
  });
}

// ============================================================
// SLIDE 8: 메타표면 스케일 결과
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("4. 메타표면 스케일 결과 (δ = 10 μm)", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 28, color: C.darkText, bold: true, margin: 0,
  });

  // Key metrics - large stat callouts
  const stats = [
    { num: "11.0%", label: "훈련 손실 개선", color: C.accent },
    { num: "+6.3%", label: "Strehl 비율 개선", color: C.amber },
    { num: "~2%", label: "결합 효율 이득", color: C.teal },
  ];
  stats.forEach((st, i) => {
    const cx = 0.5 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.05, w: 2.8, h: 1.3, fill: { color: C.white }, shadow: makeShadow() });
    s.addText(st.num, { x: cx, y: 1.1, w: 2.8, h: 0.8, fontFace: FONTS.header, fontSize: 36, color: st.color, bold: true, align: "center", margin: 0 });
    s.addText(st.label, { x: cx, y: 1.85, w: 2.8, h: 0.4, fontFace: FONTS.body, fontSize: 12, color: C.midGray, align: "center", margin: 0 });
  });

  // Figure
  s.addImage({ path: path.resolve(FIG, "fig4_metasurface_performance.png"), x: 0.3, y: 2.5, w: 9.4, h: 2.6 });

  // Caveat
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 5.1, w: 9.0, h: 0.45, fill: { color: "FEF2F2" } });
  s.addText("실용적 의미 없음: 30cm 빔 처리에 30,000×30,000 격자 필요 + Strehl 6.3% 개선은 미달", {
    x: 0.7, y: 5.12, w: 8.6, h: 0.4,
    fontFace: FONTS.body, fontSize: 11, color: C.red, margin: 0,
  });
}

// ============================================================
// SLIDE 9: 학습된 위상 패턴 — 가장 중요한 증거
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("4-1. 학습된 위상 패턴 — \"아무것도 하지 않는 것이 최적\"", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 24, color: C.accent, bold: true, margin: 0,
  });

  s.addImage({ path: path.resolve(FIG, "fig6_learned_phases.png"), x: 0.3, y: 0.85, w: 9.4, h: 3.2 });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 9.0, h: 1.2, fill: { color: C.deepBlue } });
  s.addText([
    { text: "5개 층 모두 φ ≈ 0 rad (거의 평평)으로 수렴", options: { bold: true, breakLine: true, color: C.white, fontSize: 14 } },
    { text: "→ 최적화기가 200 에폭의 역전파를 통해 발견한 결론:", options: { breakLine: true, color: C.midGray, fontSize: 12 } },
    { text: "   \"무작위 난류에 대한 최적의 고정 응답은 아무것도 하지 않는 것\"", options: { color: C.accent, fontSize: 13, italic: true } },
  ], { x: 0.7, y: 4.25, w: 8.6, h: 1.1, fontFace: FONTS.body, margin: 0 });
}

// ============================================================
// SLIDE 10: 조도 비교
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("4-2. 조도 비교: 기준선 vs D2NN vs 진공", {
    x: 0.5, y: 0.15, w: 9, h: 0.5,
    fontFace: FONTS.header, fontSize: 26, color: C.white, bold: true, margin: 0,
  });
  s.addImage({ path: path.resolve(FIG, "fig5_irradiance_comparison.png"), x: 0.2, y: 0.7, w: 9.6, h: 4.3 });
  s.addText("기준선과 D2NN 열이 거의 동일 — 스페클 패턴이 본질적으로 변하지 않음", {
    x: 0.5, y: 5.1, w: 9, h: 0.4,
    fontFace: FONTS.body, fontSize: 11, color: C.midGray, italic: true, margin: 0,
  });
}

// ============================================================
// SLIDE 11: 물리적 스케일 결과
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("5. 물리적 스케일 결과 (δ = 150 μm)", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 28, color: C.darkText, bold: true, margin: 0,
  });

  // Left: config
  const configRows = [
    [
      { text: "매개변수", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "값", options: { bold: true, color: C.white, fill: { color: C.teal } } },
    ],
    ["격자 크기", "4096 (중심 1024 크롭)"],
    ["δ/λ", "97"],
    ["수신 구경", "15 cm"],
    ["층간 간격", "9 cm (~3.1 px)"],
    ["손실 함수", "복소 필드 중첩"],
    ["에폭", "200"],
  ];
  s.addTable(configRows, {
    x: 0.5, y: 1.1, w: 4.5,
    fontSize: 12, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [1.8, 2.7],
    autoPage: false,
  });

  // Right: result with big number
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 1.1, w: 4.0, h: 2.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("0.8%", {
    x: 5.5, y: 1.3, w: 4.0, h: 1.2,
    fontFace: FONTS.header, fontSize: 60, color: C.red, bold: true, align: "center", margin: 0,
  });
  s.addText("훈련 손실 개선\n(수치 노이즈 수준)", {
    x: 5.5, y: 2.4, w: 4.0, h: 0.7,
    fontFace: FONTS.body, fontSize: 14, color: C.midGray, align: "center", margin: 0,
  });
  s.addText("검증 손실: 200 에폭 동안 정체", {
    x: 5.5, y: 3.1, w: 4.0, h: 0.5,
    fontFace: FONTS.body, fontSize: 12, color: C.red, align: "center", margin: 0,
  });

  // Bottom: physical explanation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.2, w: 9.0, h: 1.2, fill: { color: C.deepBlue } });
  s.addText("스케일 제약:", { x: 0.7, y: 4.25, w: 2, h: 0.35, fontFace: FONTS.body, fontSize: 13, color: C.accent, bold: true, margin: 0 });
  s.addText([
    { text: "메타표면: 빔 1mm, 3px 결합 30μm → 빔 대비 3%", options: { breakLine: true } },
    { text: "물리적: 빔 150mm, 3px 결합 0.45mm → 빔 대비 0.3% (10배 작음)", options: {} },
  ], {
    x: 0.7, y: 4.6, w: 8.5, h: 0.7,
    fontFace: FONTS.body, fontSize: 12, color: C.midGray, margin: 0,
  });
}

// ============================================================
// SLIDE 12: 근본 원인 분석 — 3가지 이유
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("6. 근본 원인 분석 — D2NN 실패의 3가지 이유", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 26, color: C.darkText, bold: true, margin: 0,
  });

  const causes = [
    {
      num: "1", title: "고정 마스크 vs 무작위 입력",
      body: "Kolmogorov 난류의 평균 위상 = 0\n→ 최적 고정 보정은 \"아무것도 하지 않는 것\"\n→ 학습된 위상이 평평하게 수렴",
      color: C.red,
    },
    {
      num: "2", title: "위상 전용 조도 불변성",
      body: "|U·e^(iφ)|² = |U|²\n→ 단일 층에서 진폭 변조 불가\n→ 회절 전파를 통한 간접 변환만 가능",
      color: C.amber,
    },
    {
      num: "3", title: "δ/λ 스케일링 법칙",
      body: "결합 z ∝ (δ/λ)²\n→ 물리적 스케일에서 비현실적 깊이\n→ 2mm 픽셀: 5층 D2NN = 83m",
      color: C.teal,
    },
  ];

  causes.forEach((c, i) => {
    const cy = 1.1 + i * 1.45;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: cy, w: 9.0, h: 1.3, fill: { color: C.white }, shadow: makeShadow() });
    // Number circle
    s.addShape(pres.shapes.OVAL, { x: 0.7, y: cy + 0.25, w: 0.7, h: 0.7, fill: { color: c.color } });
    s.addText(c.num, { x: 0.7, y: cy + 0.25, w: 0.7, h: 0.7, fontFace: FONTS.header, fontSize: 24, color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    // Title
    s.addText(c.title, { x: 1.7, y: cy + 0.1, w: 7.5, h: 0.4, fontFace: FONTS.body, fontSize: 16, color: c.color, bold: true, margin: 0 });
    // Body
    s.addText(c.body, { x: 1.7, y: cy + 0.5, w: 7.5, h: 0.7, fontFace: FONTS.body, fontSize: 12, color: C.darkText, margin: 0 });
  });
}

// ============================================================
// SLIDE 13: AO 비교
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("6-1. 적응 광학(AO) vs 수동 D2NN", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 30, color: C.darkText, bold: true, margin: 0,
  });

  const compRows = [
    [
      { text: "특성", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "적응 광학 (AO)", options: { bold: true, color: C.white, fill: { color: C.teal } } },
      { text: "수동 D2NN", options: { bold: true, color: C.white, fill: { color: C.teal } } },
    ],
    ["파면 감지", { text: "실시간 (Shack-Hartmann)", options: { color: C.green } }, { text: "없음", options: { color: C.red, bold: true } }],
    ["보정 요소", "변형 거울 (동적)", "고정 위상 마스크 (정적)"],
    ["적응성", { text: "밀리초 단위 갱신", options: { color: C.green } }, { text: "적응 불가", options: { color: C.red, bold: true } }],
    ["무작위 난류 보정", { text: "가능 (닫힌 루프)", options: { color: C.green, bold: true } }, { text: "불가능", options: { color: C.red, bold: true } }],
    ["전력 소비", "수백 W", { text: "0 W (수동)", options: { color: C.green } }],
    ["복잡도", "높음", "낮음"],
  ];
  s.addTable(compRows, {
    x: 0.5, y: 1.2, w: 9.0,
    fontSize: 13, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [2.5, 3.25, 3.25],
    autoPage: false,
    rowH: [0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
  });

  // Key insight
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.6, w: 9.0, h: 0.8, fill: { color: C.deepBlue } });
  s.addText("AO가 효과적인 이유: 매 순간의 난류 실현에 맞추어 거울 형상을 동적으로 갱신\nD2NN은 AO를 대체할 수 없다 — 이는 물리 법칙이 부과하는 한계", {
    x: 0.7, y: 4.65, w: 8.6, h: 0.7,
    fontFace: FONTS.body, fontSize: 12, color: C.white, margin: 0,
  });
}

// ============================================================
// SLIDE 14: 전체 실험 결과 요약
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("7. 전체 실험 결과 요약", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 30, color: C.darkText, bold: true, margin: 0,
  });

  const sumRows = [
    [
      { text: "#", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "구성", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "δ/λ", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "손실", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "훈련 개선", options: { bold: true, color: C.white, fill: { color: C.navy } } },
      { text: "판정", options: { bold: true, color: C.white, fill: { color: C.navy } } },
    ],
    ["1", "δ=2mm, z=0.01m", "1,290", "조도", "~0%", { text: "실패", options: { color: C.red, bold: true } }],
    ["2", "δ=2mm, z=10m", "1,290", "조도", "~1%", { text: "실패", options: { color: C.red, bold: true } }],
    ["3", "δ=2mm, z=50m", "1,290", "조도", "~2%", { text: "실패", options: { color: C.red, bold: true } }],
    ["4", "δ=10μm, z=387μm", "6.5", "조도", { text: "11.0%", options: { color: C.accent, bold: true } }, { text: "미미", options: { color: C.amber, bold: true } }],
    ["5", "δ=150μm, z=9cm", "97", "복소", "0.8%", { text: "실패", options: { color: C.red, bold: true } }],
  ];
  s.addTable(sumRows, {
    x: 0.5, y: 1.1, w: 9.0,
    fontSize: 12, fontFace: FONTS.body, color: C.darkText,
    border: { pt: 0.5, color: C.lightGray },
    colW: [0.5, 2.5, 1.0, 1.0, 1.5, 2.5],
    autoPage: false,
  });

  // Key observations
  s.addText("핵심 관찰", { x: 0.5, y: 3.5, w: 9, h: 0.4, fontFace: FONTS.body, fontSize: 16, color: C.teal, bold: true, margin: 0 });
  s.addText([
    { text: "#1-3: δ/λ 스케일링 법칙 직접 검증", options: { bullet: true, breakLine: true } },
    { text: "#4: 위상 패턴 → 평평 수렴 = 고정 마스크 한계 실험적 증명", options: { bullet: true, breakLine: true } },
    { text: "#5: 손실 함수 변경으로도 근본 한계 극복 불가", options: { bullet: true, breakLine: true } },
    { text: "유일한 학습 구성(#4)도 Strehl +6.3%로 실용적 의미 없음", options: { bullet: true } },
  ], { x: 0.7, y: 3.9, w: 8.5, h: 1.5, fontFace: FONTS.body, fontSize: 13, color: C.darkText, margin: 0 });
}

// ============================================================
// SLIDE 15: 향후 연구 방향
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offWhite };
  s.addText("8. 향후 연구 방향", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontFace: FONTS.header, fontSize: 30, color: C.darkText, bold: true, margin: 0,
  });

  const futures = [
    { title: "하이브리드 D2NN + WFS", desc: "Shack-Hartmann 파면 감지기의 실시간 피드백 결합" },
    { title: "비선형 D2NN", desc: "포화 흡수체, Kerr 매질로 진폭 변조 달성" },
    { title: "디지털 트윈 + CNN 후처리", desc: "카메라 + CNN으로 디지털 도메인 빔 복원" },
    { title: "결정론적 수차 보정용 D2NN", desc: "렌즈 수차, 정적 열 왜곡 등 고정 수차에 적용" },
    { title: "시간 평균 빔 성형기", desc: "장노출 평균 빔의 에너지 분포 최적화" },
  ];

  // Row 1: 3 cards, Row 2: 2 cards centered
  futures.forEach((f, i) => {
    let cx, cy;
    if (i < 3) {
      cx = 0.5 + i * 3.1;
      cy = 1.1;
    } else {
      cx = 2.05 + (i - 3) * 3.1;
      cy = 3.2;
    }
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 2.8, h: 1.8, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: cy, w: 2.8, h: 0.06, fill: { color: C.teal } });
    s.addText(f.title, { x: cx + 0.15, y: cy + 0.2, w: 2.5, h: 0.5, fontFace: FONTS.body, fontSize: 14, color: C.teal, bold: true, margin: 0 });
    s.addText(f.desc, { x: cx + 0.15, y: cy + 0.7, w: 2.5, h: 0.9, fontFace: FONTS.body, fontSize: 11, color: C.darkText, margin: 0 });
  });
}

// ============================================================
// SLIDE 16: 결론
// ============================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  s.addText("결론", {
    x: 0.8, y: 0.8, w: 8.4, h: 0.8,
    fontFace: FONTS.header, fontSize: 40, color: C.white, bold: true, margin: 0,
  });

  s.addText([
    { text: "수동 D2NN은 대기 난류 보정에 근본적으로 부적합하다", options: { bold: true, fontSize: 18, color: C.accent, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "1. 고정 위상 마스크는 무작위 입력에 대해 항등 연산으로 수렴", options: { breakLine: true, fontSize: 14, color: C.white } },
    { text: "2. 위상 전용 변조는 국소 조도를 변경할 수 없음", options: { breakLine: true, fontSize: 14, color: C.white } },
    { text: "3. 물리적 스케일에서 불충분한 회절 결합", options: { breakLine: true, fontSize: 14, color: C.white } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "FSO 난류 보정 → 적응 광학(AO) 시스템 필수", options: { bold: true, fontSize: 16, color: C.accent } },
  ], { x: 0.8, y: 1.8, w: 8.4, h: 2.8, fontFace: FONTS.body, margin: 0 });

  // Bottom
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.8, w: 10, h: 0.825, fill: { color: C.deepBlue } });
  s.addText("DJ  |  국방과학연구소(ADD) 광네트워크연구실  |  2026-03-23", {
    x: 0.8, y: 4.9, w: 8.4, h: 0.6,
    fontFace: FONTS.body, fontSize: 12, color: C.midGray, margin: 0,
  });
}

// ============================================================
// Write file
// ============================================================
const outPath = path.join(__dirname, "d2nn_turbulence_correction_report.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Created:", outPath);
}).catch(err => {
  console.error("Error:", err);
});
