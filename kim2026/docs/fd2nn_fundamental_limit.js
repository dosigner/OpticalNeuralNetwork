const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

// ── Color palette: Midnight Executive ──
const C = {
  navy: "1E2761",
  ice: "CADCFC",
  white: "FFFFFF",
  dark: "0F1735",
  accent: "4A90D9",
  red: "E74C3C",
  green: "27AE60",
  gray: "8899AA",
  lightBg: "F4F6FA",
  cardBg: "FFFFFF",
  dimText: "6B7B8D",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";

function makeShadow() {
  return { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.12 };
}

function readJson(relPath) {
  return JSON.parse(fs.readFileSync(path.resolve(__dirname, relPath), "utf8"));
}

function fmt3(value) {
  return Number(value).toFixed(3);
}

function fmt1(value) {
  return Number(value).toFixed(1);
}

function addSymbol(slide, symbol, x, y, color) {
  slide.addText(symbol, {
    x, y, w: 0.28, h: 0.28,
    fontFace: FONT_B, fontSize: 18, bold: true, color,
    align: "center", valign: "mid", margin: 0,
  });
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "D2NN Lab";
  pres.title = "Static Fourier-Plane Phase Mask Limits in a 4f FD2NN Sweep";

  const sweepRuns = [
    readJson("../autoresearch/runs/0405-fd2nn-4f-sweep-pitchrescale/f10mm_z1mm/results.json"),
    readJson("../autoresearch/runs/0405-fd2nn-4f-sweep-pitchrescale/f10mm_z3mm/results.json"),
    readJson("../autoresearch/runs/0405-fd2nn-4f-sweep-pitchrescale/f10mm_z5mm/results.json"),
    readJson("../autoresearch/runs/0405-fd2nn-4f-sweep-pitchrescale/f25mm_z1mm/results.json"),
  ].map((run) => {
    const delta = run.focal_pib_10um - run.focal_pib_10um_baseline;
    const dxFourierM = run.dx_fourier_um * 1e-6;
    const zM = run.z_mm * 1e-3;
    const spreadPx = zM * 1.55e-6 / (dxFourierM * dxFourierM);
    return { ...run, delta, spreadPx };
  });
  const bestSweepRun = sweepRuns.reduce((best, run) => (run.delta > best.delta ? run : best), sweepRuns[0]);

  // ══════════════════════════════════════════════
  // SLIDE 1: Title
  // ══════════════════════════════════════════════
  let s1 = pres.addSlide();
  s1.background = { color: C.dark };
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
  s1.addText("Why a Static Fourier-Plane Phase Mask\nStruggles With Random Turbulence", {
    x: 0.8, y: 1.2, w: 8.4, h: 2.0,
    fontFace: FONT_H, fontSize: 36, color: C.white, bold: true, lineSpacingMultiple: 1.2,
  });
  s1.addText("제한된 가정의 수학적 힌트와 현재 4f sweep의 보수적 재해석", {
    x: 0.8, y: 3.4, w: 8.4, h: 0.6,
    fontFace: FONT_B, fontSize: 18, color: C.ice,
  });
  s1.addText("D2NN Lab  |  2026-04-06", {
    x: 0.8, y: 4.6, w: 8.4, h: 0.5,
    fontFace: FONT_B, fontSize: 14, color: C.gray,
  });

  // ══════════════════════════════════════════════
  // SLIDE 2: Turbulence PSD — "Looks Simple"
  // ══════════════════════════════════════════════
  let s2 = pres.addSlide();
  s2.background = { color: C.lightBg };
  s2.addText("Kolmogorov PSD: Helpful, But Not Enough", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 30, color: C.navy, bold: true, margin: 0,
  });

  // Left: formula card
  s2.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.3, w: 4.3, h: 3.5, fill: { color: C.cardBg }, shadow: makeShadow(),
  });
  s2.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.3, w: 0.07, h: 3.5, fill: { color: C.accent } });
  s2.addText("Phase Power Spectral Density", {
    x: 0.9, y: 1.5, w: 3.8, h: 0.4,
    fontFace: FONT_B, fontSize: 14, color: C.accent, bold: true, margin: 0,
  });
  s2.addText([
    { text: "Kolmogorov:", options: { bold: true, breakLine: true, fontSize: 16 } },
    { text: "Φ(κ) = 0.023 r₀⁻⁵ᐟ³ · κ⁻¹¹ᐟ³", options: { breakLine: true, fontSize: 20, fontFace: "Consolas", color: C.navy } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "von Kármán:", options: { bold: true, breakLine: true, fontSize: 16 } },
    { text: "Φ(κ) = 0.023 r₀⁻⁵ᐟ³ (κ²+κ₀²)⁻¹¹ᐟ⁶", options: { fontSize: 18, fontFace: "Consolas", color: C.navy } },
  ], { x: 0.9, y: 2.1, w: 3.8, h: 2.4, fontFace: FONT_B, color: "333333", valign: "top", margin: 0 });

  // Right: the trap
  s2.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.3, w: 4.3, h: 3.5, fill: { color: C.cardBg }, shadow: makeShadow(),
  });
  s2.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.3, w: 0.07, h: 3.5, fill: { color: C.red } });
  s2.addText("The Trap", {
    x: 5.5, y: 1.5, w: 3.8, h: 0.4,
    fontFace: FONT_B, fontSize: 14, color: C.red, bold: true, margin: 0,
  });
  s2.addText([
    { text: '"Log-log에서 직선이니까\n Fourier mask로 쉽게 보정할 수 있지 않을까?"', options: { italic: true, breakLine: true, fontSize: 15, color: "555555" } },
    { text: "", options: { breakLine: true, fontSize: 12 } },
    { text: "PSD는 2차 통계량.\nrealization-level 복원에는 추가 prior가 필요하다.", options: { bold: true, fontSize: 16, color: C.red } },
  ], { x: 5.5, y: 2.1, w: 3.8, h: 2.4, fontFace: FONT_B, valign: "top", margin: 0 });

  s2.addText("PSD = ⟨|ψ̂(κ)|²⟩  ←  phase-sensitive information은 직접 남지 않는다", {
    x: 0.6, y: 5.0, w: 9, h: 0.4,
    fontFace: "Consolas", fontSize: 14, color: C.dimText,
  });

  // ══════════════════════════════════════════════
  // SLIDE 3: PSD vs Realization
  // ══════════════════════════════════════════════
  let s3 = pres.addSlide();
  s3.background = { color: C.lightBg };
  s3.addText("PSD Does Not Specify A Realization", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 30, color: C.navy, bold: true, margin: 0,
  });

  // Key equation
  s3.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.2, w: 8.8, h: 1.2, fill: { color: C.navy },
  });
  s3.addText([
    { text: "ψ̂(κ) = ", options: { color: C.ice, fontSize: 22 } },
    { text: "|ψ̂(κ)|", options: { color: C.green, fontSize: 22, bold: true } },
    { text: " · exp( j · ", options: { color: C.ice, fontSize: 22 } },
    { text: "θ(κ)", options: { color: C.red, fontSize: 22, bold: true } },
    { text: " )          ", options: { color: C.ice, fontSize: 22 } },
    { text: "toy prior: θ(κ) ~ Uniform[0, 2π]", options: { color: C.red, fontSize: 16 } },
  ], { x: 0.8, y: 1.3, w: 8.4, h: 1.0, fontFace: "Consolas", valign: "middle" });

  // Example: dice analogy
  s3.addText("비유: 주사위", {
    x: 0.6, y: 2.8, w: 4.2, h: 0.5,
    fontFace: FONT_H, fontSize: 20, color: C.navy, bold: true, margin: 0,
  });

  // Card: what PSD tells
  s3.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.4, w: 4.2, h: 1.8, fill: { color: C.cardBg }, shadow: makeShadow() });
  addSymbol(s3, "✓", 0.83, 3.56, C.green);
  s3.addText("PSD가 알려주는 것", {
    x: 1.25, y: 3.5, w: 3.3, h: 0.4,
    fontFace: FONT_B, fontSize: 14, color: C.green, bold: true, margin: 0,
  });
  s3.addText('"각 면이 나올 확률 = 1/6"\n→ 주파수별 에너지 분포 (통계)', {
    x: 0.8, y: 4.0, w: 3.8, h: 1.0,
    fontFace: FONT_B, fontSize: 14, color: "444444", margin: 0,
  });

  // Card: what we need
  s3.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.4, w: 4.2, h: 1.8, fill: { color: C.cardBg }, shadow: makeShadow() });
  addSymbol(s3, "✗", 5.43, 3.56, C.red);
  s3.addText("보정에 필요한 것", {
    x: 5.85, y: 3.5, w: 3.3, h: 0.4,
    fontFace: FONT_B, fontSize: 14, color: C.red, bold: true, margin: 0,
  });
  s3.addText('"이번에 나온 면 자체"\n→ 이번 realization의 θ(κ) + 추가 sensing/prior', {
    x: 5.4, y: 4.0, w: 3.8, h: 1.0,
    fontFace: FONT_B, fontSize: 14, color: "444444", margin: 0,
  });

  // ══════════════════════════════════════════════
  // SLIDE 4: Static Mask — What It Does
  // ══════════════════════════════════════════════
  let s4 = pres.addSlide();
  s4.background = { color: C.lightBg };
  s4.addText("Single Static Fourier Multiplier의 한계", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 30, color: C.navy, bold: true, margin: 0,
  });

  // What mask does
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 8.8, h: 1.0, fill: { color: C.navy } });
  s4.addText([
    { text: "Mask: ", options: { color: C.ice, fontSize: 20 } },
    { text: "H(κ) = exp( j · Δφ(κ) )", options: { color: C.green, fontSize: 20, bold: true } },
    { text: "   ← 매 입력마다 동일한 고정값", options: { color: C.gray, fontSize: 15 } },
  ], { x: 0.8, y: 1.3, w: 8.4, h: 0.8, fontFace: "Consolas", valign: "middle" });

  // What we need
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 2.4, w: 8.8, h: 1.0, fill: { color: "3D1420" } });
  s4.addText([
    { text: "Need: ", options: { color: C.ice, fontSize: 20 } },
    { text: "H_ideal(κ) = exp( -j · θ_turb(κ) )", options: { color: C.red, fontSize: 20, bold: true } },
    { text: "   ← single realization마다 달라짐", options: { color: C.red, fontSize: 15 } },
  ], { x: 0.8, y: 2.5, w: 8.4, h: 0.8, fontFace: "Consolas", valign: "middle" });

  // Example
  s4.addText("toy example: 3개 realization에서 주파수 κ₀의 위상", {
    x: 0.6, y: 3.8, w: 9, h: 0.4,
    fontFace: FONT_B, fontSize: 16, color: C.navy, bold: true, margin: 0,
  });

  const tableData = [
    [
      { text: "", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "θ_turb(κ₀)", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "Mask Δφ(κ₀)", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "보정 후 잔차", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "결과", options: { fill: { color: C.navy }, color: C.white, bold: true } },
    ],
    [
      { text: "Realization 1" }, { text: "47°" }, { text: "90°" },
      { text: "47° - 90° = -43°" }, { text: "✗ 악화", options: { color: C.red, bold: true } },
    ],
    [
      { text: "Realization 2" }, { text: "203°" }, { text: "90°" },
      { text: "203° - 90° = 113°" }, { text: "✗ 악화", options: { color: C.red, bold: true } },
    ],
    [
      { text: "Realization 3" }, { text: "85°" }, { text: "90°" },
      { text: "85° - 90° = -5°" }, { text: "✓ 운 좋게 맞음", options: { color: C.green, bold: true } },
    ],
  ];
  s4.addTable(tableData, {
    x: 0.6, y: 4.3, w: 8.8, h: 1.2,
    fontSize: 13, fontFace: FONT_B,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [1.5, 1.4, 1.5, 2.4, 2.0],
  });

  // ══════════════════════════════════════════════
  // SLIDE 5: Optimal Static Mask = Do Nothing
  // ══════════════════════════════════════════════
  let s5 = pres.addSlide();
  s5.background = { color: C.lightBg };
  s5.addText("제약 없는 Wiener Filter가 주는 힌트", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });

  // Wiener filter derivation
  s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 8.8, h: 3.8, fill: { color: C.cardBg }, shadow: makeShadow() });
  s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 0.07, h: 3.8, fill: { color: C.accent } });

  s5.addText("Toy derivation: single complex scalar Wiener filter", {
    x: 0.9, y: 1.35, w: 8, h: 0.4,
    fontFace: FONT_B, fontSize: 16, color: C.accent, bold: true, margin: 0,
  });
  s5.addText([
    { text: "Step 1.  ", options: { bold: true, color: C.navy } },
    { text: "목표: 단일 주파수에서 앙상블 평균 MSE 최소화", options: { breakLine: true } },
    { text: "         minimize  E[ |H(κ)·û_turb(κ) - û_vac(κ)|² ]", options: { fontFace: "Consolas", fontSize: 14, breakLine: true, color: C.navy } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Step 2.  ", options: { bold: true, color: C.navy } },
    { text: "제약 없는 복소 스칼라 최적해:", options: { breakLine: true } },
    { text: "         H_opt(κ) = ⟨û_vac · û_turb*⟩ / ⟨|û_turb|²⟩", options: { fontFace: "Consolas", fontSize: 14, breakLine: true, color: C.navy } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Step 3.  ", options: { bold: true, color: C.navy } },
    { text: "단순화 가정:", options: { breakLine: true } },
    { text: "         û_turb(κ) = û_vac(κ) · exp(jθ),   |A| and θ independent", options: { fontFace: "Consolas", fontSize: 12, breakLine: true, color: C.navy } },
    { text: "         ⟨û_vac · û_turb*⟩ = ⟨|A|² · exp(-j·θ)⟩", options: { fontFace: "Consolas", fontSize: 12, breakLine: true, color: C.navy } },
    { text: "                           ≈ ⟨|A|²⟩ · ⟨exp(-j·θ)⟩", options: { fontFace: "Consolas", fontSize: 12, breakLine: true, color: C.navy } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Step 4.  ", options: { bold: true, color: C.red } },
    { text: "toy prior: θ ~ Uniform[0, 2π]  →  ", options: {} },
    { text: "⟨exp(-jθ)⟩ = 0", options: { bold: true, color: C.red, fontSize: 18 } },
  ], { x: 0.9, y: 1.85, w: 8.2, h: 3.0, fontFace: FONT_B, fontSize: 15, color: "333333", valign: "top", margin: 0 });

  // Conclusion box
  s5.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.05, w: 8.8, h: 0.55, fill: { color: "C97A00" } });
  s5.addText("Under this toy model, unconstrained H_opt(κ) can collapse toward 0.  This is not a direct proof for phase-only multilayer FD2NN.", {
    x: 0.8, y: 5.05, w: 8.4, h: 0.55,
    fontFace: FONT_B, fontSize: 13, color: C.white, bold: true, valign: "middle",
  });

  // ══════════════════════════════════════════════
  // SLIDE 6: Experimental Verification
  // ══════════════════════════════════════════════
  let s6 = pres.addSlide();
  s6.background = { color: C.lightBg };
  s6.addText("현재 4f FD2NN Sweep: 제한된 경험적 증거", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });

  s6.addText("Architecture:  Input → Lens(f) → [Mask₁ →ASM(z)→ ... → Mask₅] → Lens(f) → Focal Lens → PIB", {
    x: 0.6, y: 1.1, w: 8.8, h: 0.35,
    fontFace: "Consolas", fontSize: 12, color: C.dimText,
  });

  const resultTable = [
    [
      { text: "Config", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "dx_f [μm]", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "fPIB@10μm", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "Baseline", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "Δ", options: { fill: { color: C.navy }, color: C.white, bold: true } },
      { text: "판정", options: { fill: { color: C.navy }, color: C.white, bold: true } },
    ],
    ...sweepRuns.map((run) => [
      { text: `f=${fmt1(run.f_mm)}mm, z=${fmt1(run.z_mm)}mm`, options: run.delta > 0 ? { bold: true } : {} },
      fmt3(run.dx_fourier_um),
      { text: fmt3(run.focal_pib_10um), options: run.delta > 0 ? { bold: true } : {} },
      fmt3(run.focal_pib_10um_baseline),
      { text: `${run.delta >= 0 ? "+" : ""}${fmt3(run.delta)}`, options: { color: run.delta >= 0 ? C.green : C.red, bold: true } },
      run.delta > 0.02 ? "✓ 개선" : (run.delta > 0 ? "△ 소폭 개선" : "✗ 악화"),
    ]),
  ];
  s6.addTable(resultTable, {
    x: 0.6, y: 1.6, w: 8.8, h: 2.0,
    fontSize: 14, fontFace: FONT_B,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [2.2, 1.0, 1.4, 1.2, 1.2, 1.8],
    rowH: [0.35, 0.33, 0.33, 0.33, 0.35],
  });

  // Interpretation cards
  s6.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.9, w: 4.2, h: 1.5, fill: { color: C.cardBg }, shadow: makeShadow() });
  addSymbol(s6, "✗", 0.83, 4.06, C.red);
  s6.addText("small-f branch (10mm)", {
    x: 1.2, y: 4.0, w: 3.4, h: 0.35, fontFace: FONT_B, fontSize: 14, color: C.red, bold: true, margin: 0,
  });
  s6.addText("작은 dx_f로 층간 Fresnel spread는 커진다.\n하지만 현재 sweep에서는 tested z 전 범위에서\nbaseline PIB를 넘지 못했다.", {
    x: 0.8, y: 4.45, w: 3.8, h: 0.85, fontFace: FONT_B, fontSize: 13, color: "555555", margin: 0,
  });

  s6.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.9, w: 4.2, h: 1.5, fill: { color: C.cardBg }, shadow: makeShadow() });
  addSymbol(s6, "✓", 5.43, 4.06, C.green);
  s6.addText("large-f branch (25mm)", {
    x: 5.8, y: 4.0, w: 3.4, h: 0.35, fontFace: FONT_B, fontSize: 14, color: C.green, bold: true, margin: 0,
  });
  s6.addText("큰 dx_f로 층간 spread는 약해진다.\n현재 tested point에서는 +0.010 수준의\n소폭 개선만 관찰됐다.", {
    x: 5.4, y: 4.45, w: 3.8, h: 0.85, fontFace: FONT_B, fontSize: 13, color: "555555", margin: 0,
  });
  s6.addText("Note: table values are from final results.json in the 0405 4f sweep run family.  This slide is evidence, not an impossibility theorem.", {
    x: 0.6, y: 5.28, w: 8.8, h: 0.18,
    fontFace: FONT_B, fontSize: 9.5, color: C.dimText, margin: 0,
  });

  // ══════════════════════════════════════════════
  // SLIDE 7: The Dilemma
  // ══════════════════════════════════════════════
  let s7 = pres.addSlide();
  s7.background = { color: C.lightBg };
  s7.addText("Reading The f-z Sweep More Carefully", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });

  // Left arrow: small f
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.3, w: 4.2, h: 2.5, fill: { color: C.cardBg }, shadow: makeShadow() });
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.3, w: 0.07, h: 2.5, fill: { color: C.red } });
  s7.addText("f = 10 mm", {
    x: 0.9, y: 1.45, w: 3.7, h: 0.35, fontFace: FONT_B, fontSize: 16, color: C.red, bold: true, margin: 0,
  });
  s7.addText([
    { text: `✓ 1mm step gives ~${fmt1(sweepRuns[0].spreadPx)} px Fresnel spread`, options: { breakLine: true, color: C.green } },
    { text: "✓ smaller dx_f means stronger layer-to-layer coupling", options: { breakLine: true, color: C.green } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "✗ this sweep still degraded PIB at z=1,3,5mm", options: { breakLine: true, color: C.red } },
    { text: "✗ cause may mix diffraction, sampling, and optimization", options: { color: C.red } },
  ], { x: 0.9, y: 1.95, w: 3.7, h: 1.6, fontFace: FONT_B, fontSize: 14, margin: 0 });

  // Right: large f
  s7.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.3, w: 4.2, h: 2.5, fill: { color: C.cardBg }, shadow: makeShadow() });
  s7.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.3, w: 0.07, h: 2.5, fill: { color: C.green } });
  s7.addText("f = 25 mm", {
    x: 5.5, y: 1.45, w: 3.7, h: 0.35, fontFace: FONT_B, fontSize: 16, color: C.green, bold: true, margin: 0,
  });
  s7.addText([
    { text: `✓ 1mm step gives ~${fmt1(sweepRuns[3].spreadPx)} px Fresnel spread`, options: { breakLine: true, color: C.green } },
    { text: "✓ larger dx_f keeps propagation closer to near-field", options: { breakLine: true, color: C.green } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "✗ inter-layer mixing is weaker in the tested geometry", options: { breakLine: true, color: C.red } },
    { text: "✗ observed gain stayed small in this run family", options: { color: C.red } },
  ], { x: 5.5, y: 1.95, w: 3.7, h: 1.6, fontFace: FONT_B, fontSize: 14, margin: 0 });

  // Key formula
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.1, w: 8.8, h: 1.2, fill: { color: C.navy } });
  s7.addText([
    { text: "Approx. Fresnel spread (pixels) ≈ z · λ / dx_fourier²", options: { fontSize: 17, fontFace: "Consolas", color: C.ice, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "dx_fourier = λf/(N·dx)  →  f↑ 하면 dx_f↑  →  spread↓  →  coupling 약화", options: { fontSize: 14, fontFace: "Consolas", color: C.gray } },
  ], { x: 0.8, y: 4.2, w: 8.4, h: 1.0, valign: "middle" });

  // ══════════════════════════════════════════════
  // SLIDE 8: Why Spatial D2NN Works Better
  // ══════════════════════════════════════════════
  let s8 = pres.addSlide();
  s8.background = { color: C.lightBg };
  s8.addText("Spatial Architectures May Help Differently", {
    x: 0.6, y: 0.3, w: 9, h: 0.7,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });

  // Comparison
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 4.2, h: 3.0, fill: { color: C.cardBg }, shadow: makeShadow() });
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.2, w: 0.07, h: 3.0, fill: { color: C.red } });
  s8.addText("Current 4f FD2NN Sweep", {
    x: 0.9, y: 1.35, w: 3.7, h: 0.35, fontFace: FONT_B, fontSize: 16, color: C.red, bold: true, margin: 0,
  });
  s8.addText([
    { text: "Fourier-plane phase-only stack", options: { breakLine: true, bold: true } },
    { text: "= realization mismatch에 민감", options: { breakLine: true } },
    { text: "= this sweep did not show a large gain", options: { breakLine: true, color: C.red, bold: true } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: `best in this run family: Δ PIB = +${fmt3(bestSweepRun.delta)}`, options: { breakLine: true } },
    { text: "(same baseline family, final test numbers)", options: { color: C.dimText } },
  ], { x: 0.9, y: 1.85, w: 3.7, h: 2.2, fontFace: FONT_B, fontSize: 15, color: "333333", margin: 0 });

  s8.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 4.2, h: 3.0, fill: { color: C.cardBg }, shadow: makeShadow() });
  s8.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.2, w: 0.07, h: 3.0, fill: { color: C.green } });
  s8.addText("Separate Spatial-D2NN Reports", {
    x: 5.5, y: 1.35, w: 3.7, h: 0.35, fontFace: FONT_B, fontSize: 16, color: C.green, bold: true, margin: 0,
  });
  s8.addText([
    { text: "output-plane energy redistribution", options: { breakLine: true, bold: true } },
    { text: "= mode-conversion-like behavior can help PIB", options: { breakLine: true } },
    { text: "= but this deck does not present an apples-to-apples comparison", options: { breakLine: true, color: C.green, bold: true } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "use as a hypothesis, not as proof of superiority", options: { breakLine: true } },
    { text: "(same metric / same baseline / same propagation needed)", options: { color: C.green } },
  ], { x: 5.5, y: 1.85, w: 3.7, h: 2.2, fontFace: FONT_B, fontSize: 15, color: "333333", margin: 0 });

  // Key insight
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.3, w: 8.8, h: 0.9, fill: { color: "FFF8E1" }, shadow: makeShadow() });
  addSymbol(s8, "!", 0.84, 4.46, "F39C12");
  s8.addText("Mechanism may differ from phase correction, but this still needs a controlled apples-to-apples experiment.", {
    x: 1.3, y: 4.3, w: 7.9, h: 0.9,
    fontFace: FONT_B, fontSize: 16, color: C.navy, bold: true, valign: "middle", margin: 0,
  });

  // ══════════════════════════════════════════════
  // SLIDE 9: Conclusion
  // ══════════════════════════════════════════════
  let s9 = pres.addSlide();
  s9.background = { color: C.dark };
  s9.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });

  s9.addText("Conclusion: What This Deck Can Safely Claim", {
    x: 0.8, y: 0.5, w: 8.4, h: 0.7,
    fontFace: FONT_H, fontSize: 30, color: C.white, bold: true,
  });

  s9.addText([
    { text: "1. ", options: { bold: true, color: C.accent } },
    { text: "PSD만으로 realization-level 보정을 결정할 수는 없다", options: { bold: true, breakLine: true } },
    { text: "   PSD는 2차 통계량이다. inversion에는 추가 prior / sensing / operator analysis가 필요하다.", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "2. ", options: { bold: true, color: C.accent } },
    { text: "toy Wiener derivation은 single unconstrained filter의 힌트일 뿐이다", options: { bold: true, breakLine: true } },
    { text: "   phase-only multilayer FD2NN의 직접 proof로 쓰면 안 된다.", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "3. ", options: { bold: true, color: C.accent } },
    { text: "현재 0405 4f sweep에서는 큰 개선이 관찰되지 않았다", options: { bold: true, breakLine: true } },
    { text: `   best final-test gain in this family: Δ PIB = +${fmt3(bestSweepRun.delta)}.  이건 empirical evidence이지 impossibility theorem은 아니다.`, options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "4. ", options: { bold: true, color: C.accent } },
    { text: "stronger claims need stricter controls", options: { bold: true, breakLine: true } },
    { text: "   alias-safe propagation, consistent metrics, and apples-to-apples spatial-vs-Fourier comparison are still required.", options: { fontSize: 14, color: C.gray } },
  ], {
    x: 0.8, y: 1.5, w: 8.4, h: 3.8,
    fontFace: FONT_B, fontSize: 17, color: C.white, valign: "top",
  });

  const outPath = "/root/dj/D2NN/kim2026/docs/fd2nn_fundamental_limit.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("Saved to " + outPath);
}

main().catch(e => { console.error(e); process.exit(1); });
