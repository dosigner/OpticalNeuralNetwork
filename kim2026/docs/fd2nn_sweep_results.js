const pptxgen = require("pptxgenjs");

const C = {
  navy: "1E2761", ice: "CADCFC", white: "FFFFFF", dark: "0F1735",
  accent: "4A90D9", red: "E74C3C", green: "27AE60", gray: "8899AA",
  lightBg: "F4F6FA", cardBg: "FFFFFF", dimText: "6B7B8D",
};
const FONT_H = "Georgia";
const FONT_B = "Calibri";
const FIG = "/root/dj/D2NN/kim2026/docs/fd2nn_sweep_figures";

function makeShadow() {
  return { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.12 };
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "D2NN Lab";
  pres.title = "FD2NN 4f Multi-layer Sweep Results";

  // ── Slide 1: Title ──
  let s1 = pres.addSlide();
  s1.background = { color: C.dark };
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
  s1.addText("FD2NN 4f Multi-layer Sweep\n실험 결과 보고", {
    x: 0.8, y: 1.0, w: 8.4, h: 2.0,
    fontFace: FONT_H, fontSize: 36, color: C.white, bold: true, lineSpacingMultiple: 1.3,
  });
  s1.addText([
    { text: "Architecture: ", options: { color: C.gray, fontSize: 14 } },
    { text: "Input → Lens(f) → [Mask₁→ASM(z)→...→Mask₅] → Lens(f) → Focal Lens(6.5mm) → PIB", options: { color: C.ice, fontSize: 14, fontFace: "Consolas" } },
  ], { x: 0.8, y: 3.3, w: 8.4, h: 0.5 });
  s1.addText([
    { text: "Sweep: ", options: { color: C.gray } },
    { text: "f ∈ {10, 25}mm × z ∈ {1, 3, 5}mm = 6 configs", options: { color: C.ice } },
    { text: "  |  Loss: focal_raw_received_power", options: { color: C.ice } },
  ], { x: 0.8, y: 3.9, w: 8.4, h: 0.4, fontFace: FONT_B, fontSize: 14 });
  s1.addText("2026-04-06  |  D2NN Lab", {
    x: 0.8, y: 4.8, w: 8.4, h: 0.4, fontFace: FONT_B, fontSize: 13, color: C.gray,
  });

  // ── Slide 2: Overview — PIB Bar Chart ──
  let s2 = pres.addSlide();
  s2.background = { color: C.lightBg };
  s2.addText("Focal PIB @10μm — 전체 결과", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s2.addImage({ path: `${FIG}/fig1_pib_bar.png`, x: 0.3, y: 0.9, w: 9.4, h: 4.5 });

  // ── Slide 3: Heatmap ──
  let s3 = pres.addSlide();
  s3.background = { color: C.lightBg };
  s3.addText("f × z Sweep Heatmap", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s3.addImage({ path: `${FIG}/fig2_heatmap.png`, x: 1.2, y: 0.9, w: 7.6, h: 4.3 });

  // ── Slide 4: Results Table ──
  let s4 = pres.addSlide();
  s4.background = { color: C.lightBg };
  s4.addText("정량적 결과 요약", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });

  const hdrOpts = { fill: { color: C.navy }, color: C.white, bold: true, fontSize: 12 };
  const tableData = [
    [
      { text: "Config", options: hdrOpts },
      { text: "f (mm)", options: hdrOpts },
      { text: "z (mm)", options: hdrOpts },
      { text: "dx_f (μm)", options: hdrOpts },
      { text: "z/z_R", options: hdrOpts },
      { text: "fPIB@10μm", options: hdrOpts },
      { text: "Δ PIB", options: hdrOpts },
      { text: "CO", options: hdrOpts },
      { text: "TP", options: hdrOpts },
    ],
    ["f10mm_z1mm", "10", "1", "7.6", "0.67", "0.377", { text: "-0.198", options: { color: C.red, bold: true } }, "0.204", "1.000"],
    ["f10mm_z3mm", "10", "3", "7.6", "2.0", "0.290", { text: "-0.285", options: { color: C.red, bold: true } }, "0.190", "1.000"],
    ["f10mm_z5mm", "10", "5", "7.6", "3.3", "0.251", { text: "-0.324", options: { color: C.red, bold: true } }, "0.174", "1.000"],
    [
      { text: "f25mm_z1mm", options: { bold: true } }, "25", "1", "18.9", "0.028",
      { text: "0.586", options: { bold: true } },
      { text: "+0.010", options: { color: C.green, bold: true } },
      { text: "0.249", options: { bold: true } }, "1.000"
    ],
    ["f25mm_z3mm", "25", "3", "18.9", "0.083", "0.430", { text: "-0.145", options: { color: C.red, bold: true } }, "0.220", "1.000"],
    ["f25mm_z5mm", "25", "5", "18.9", "0.139", "0.419", { text: "-0.156", options: { color: C.red, bold: true } }, "0.218", "1.000"],
  ];
  s4.addTable(tableData, {
    x: 0.3, y: 1.0, w: 9.4, h: 2.2,
    fontSize: 12, fontFace: FONT_B,
    border: { pt: 0.5, color: "CCCCCC" },
    colW: [1.6, 0.7, 0.7, 0.9, 0.7, 1.1, 1.0, 0.8, 0.8],
  });

  s4.addText("Baseline (no D2NN): fPIB@10μm = 0.575, CO = 0.268", {
    x: 0.5, y: 3.4, w: 9, h: 0.35,
    fontFace: FONT_B, fontSize: 13, color: C.dimText,
  });

  // Key finding cards
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.9, w: 4.2, h: 1.4, fill: { color: C.cardBg }, shadow: makeShadow() });
  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.9, w: 0.07, h: 1.4, fill: { color: C.red } });
  s4.addText([
    { text: "f=10mm: 모두 baseline 미달", options: { bold: true, color: C.red, breakLine: true, fontSize: 14 } },
    { text: "z/z_R > 0.67 → defocus 심각\n학습이 PIB 0.3%→37.7%로 복구했지만\nbaseline(57.5%)까지 못 도달", options: { fontSize: 12, color: "555555" } },
  ], { x: 0.8, y: 4.0, w: 3.7, h: 1.2, fontFace: FONT_B, margin: 0, valign: "top" });

  s4.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 3.9, w: 4.2, h: 1.4, fill: { color: C.cardBg }, shadow: makeShadow() });
  s4.addShape(pres.shapes.RECTANGLE, { x: 5.3, y: 3.9, w: 0.07, h: 1.4, fill: { color: C.green } });
  s4.addText([
    { text: "f=25mm: baseline과 동등", options: { bold: true, color: C.green, breakLine: true, fontSize: 14 } },
    { text: "z/z_R < 0.1 → defocus 없음\nbaseline에서 시작하지만\n회절 mixing 부족 → Δ=+0.01", options: { fontSize: 12, color: "555555" } },
  ], { x: 5.6, y: 4.0, w: 3.7, h: 1.2, fontFace: FONT_B, margin: 0, valign: "top" });

  // ── Slide 5: Training Curves ──
  let s5 = pres.addSlide();
  s5.background = { color: C.lightBg };
  s5.addText("Training Loss Curves", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s5.addImage({ path: `${FIG}/fig3_loss_curves.png`, x: 0.2, y: 0.85, w: 9.6, h: 4.0 });
  s5.addText([
    { text: "f=10mm: ", options: { bold: true } },
    { text: "loss > 1.0 (exp(-1.0) ≈ 37% RP). z↑일수록 수렴이 느리고 final loss 높음.", options: { breakLine: true } },
    { text: "f=25mm: ", options: { bold: true } },
    { text: "loss < 1.0 달성 (exp(-0.63) ≈ 53% RP). 더 효율적인 에너지 집중.", options: {} },
  ], { x: 0.5, y: 4.95, w: 9, h: 0.6, fontFace: FONT_B, fontSize: 12, color: "444444" });

  // ── Slide 6: Throughput + CO ──
  let s6 = pres.addSlide();
  s6.background = { color: C.lightBg };
  s6.addText("Throughput & Complex Overlap", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s6.addImage({ path: `${FIG}/fig4_tp_co.png`, x: 0.2, y: 0.85, w: 9.6, h: 4.3 });
  s6.addText("Throughput ≈ 1.0 (모든 config에서 에너지 보존). CO는 모두 baseline(0.268) 미달 — static mask의 근본적 한계.", {
    x: 0.5, y: 5.1, w: 9, h: 0.4, fontFace: FONT_B, fontSize: 12, color: "444444",
  });

  // ── Slide 7: Physics — z/z_R regime ──
  let s7 = pres.addSlide();
  s7.background = { color: C.lightBg };
  s7.addText("물리적 해석: Diffraction Regime", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s7.addImage({ path: `${FIG}/fig5_regime.png`, x: 0.8, y: 0.85, w: 8.4, h: 4.5 });

  // ── Slide 8: Phase Masks ──
  let s8 = pres.addSlide();
  s8.background = { color: C.lightBg };
  s8.addText("학습된 Phase Masks (Layer 1 & 5)", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontFace: FONT_H, fontSize: 28, color: C.navy, bold: true, margin: 0,
  });
  s8.addImage({ path: `${FIG}/fig6_phase_masks.png`, x: 0.2, y: 0.85, w: 9.6, h: 4.1 });
  s8.addText("f=10mm masks: 강한 패턴 (학습이 defocus 보상 시도). f=25mm masks: 약한 패턴 (거의 flat → 기존 빔 유지).", {
    x: 0.5, y: 5.0, w: 9, h: 0.4, fontFace: FONT_B, fontSize: 12, color: "444444",
  });

  // ── Slide 9: Conclusion ──
  let s9 = pres.addSlide();
  s9.background = { color: C.dark };
  s9.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
  s9.addText("결론 및 시사점", {
    x: 0.8, y: 0.4, w: 8.4, h: 0.6,
    fontFace: FONT_H, fontSize: 30, color: C.white, bold: true,
  });
  s9.addText([
    { text: "1. f-z 딜레마 확인", options: { bold: true, color: C.accent, breakLine: true, fontSize: 18 } },
    { text: "   f↓: 회절 mixing 강하지만 defocus로 baseline 미달 (최대 -0.324)", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "   f↑: 안정적이지만 회절 mixing 약해 실질 효과 없음 (+0.010)", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "2. Spatial D2NN 대비 열위", options: { bold: true, color: C.accent, breakLine: true, fontSize: 18 } },
    { text: "   Spatial D2NN: PIB +0.240 (mode conversion)  vs  FD2NN: PIB +0.010 (Fourier 위상 보정 한계)", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "3. 근본 원인: Static mask로 랜덤 위상 보정 불가", options: { bold: true, color: C.accent, breakLine: true, fontSize: 18 } },
    { text: "   Wiener filter 최적해 H_opt → 0. PSD가 단순해도 개별 realization의 θ(κ)는 매번 랜덤.", options: { breakLine: true, fontSize: 14, color: C.gray } },
    { text: "", options: { breakLine: true, fontSize: 10 } },
    { text: "4. 향후 방향", options: { bold: true, color: C.accent, breakLine: true, fontSize: 18 } },
    { text: "   Adaptive mask (CNN 기반 per-input mask 생성) 또는 Spatial D2NN에 집중.", options: { fontSize: 14, color: C.gray } },
  ], { x: 0.8, y: 1.2, w: 8.4, h: 4.2, fontFace: FONT_B, color: C.white, valign: "top" });

  const outPath = "/root/dj/D2NN/kim2026/docs/fd2nn_sweep_results.pptx";
  await pres.writeFile({ fileName: outPath });
  console.log("Saved to " + outPath);
}

main().catch(e => { console.error(e); process.exit(1); });
