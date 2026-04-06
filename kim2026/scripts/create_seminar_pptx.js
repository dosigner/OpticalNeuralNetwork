const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "D2NN Lab";
pres.title = "D2NN Focal-Plane Beam Cleanup - Clean Data Results";

// Paths
const BASE = "/root/dj/D2NN/kim2026/autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14/focal_pib_only";
const FIG1 = path.join(BASE, "06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png");
const FIG2 = path.join(BASE, "07_fig2_pib_bar_chart.png");
const FIG3 = path.join(BASE, "08_fig3_d2nn_output_plane_irradiance_phase_residual.png");
const FIG4 = path.join(BASE, "09_fig4_wavefront_rms_distribution.png");

// Colors
const C = {
  darkBg: "0A1628",
  lightBg: "F0F4F8",
  cardBg: "FFFFFF",
  primary: "065A82",
  secondary: "1C7293",
  accent: "E74C3C",
  text: "1E293B",
  textLight: "64748B",
  textMid: "94A3B8",
  white: "FFFFFF",
  vacuumBlue: "2563EB",
  turbGray: "6B7280",
  d2nnRed: "DC2626",
  green: "059669",
  amber: "D97706",
};

// Helper: fresh shadow
const mkShadow = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.08 });

// ─────────────────────────────────────────────────────────────────
// Slide 1: Title
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.darkBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 1.3, w: 1.6, h: 0.35, fill: { color: C.secondary } });
  s.addText("SEMINAR", { x: 0.8, y: 1.3, w: 1.6, h: 0.35, fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle", charSpacing: 4 });
  s.addText("D2NN Focal-Plane\nBeam Cleanup", {
    x: 0.8, y: 1.9, w: 8, h: 1.6,
    fontSize: 40, fontFace: "Georgia", color: C.white, bold: true, lineSpacingMultiple: 1.1, margin: 0,
  });
  s.addText("Clean \uB370\uC774\uD130 \uC815\uC815 \uD6C4 \uACB0\uACFC \uBD84\uC11D \uBC0F \uBB3C\uB9AC\uC801 \uD574\uC11D", {
    x: 0.8, y: 3.55, w: 8, h: 0.5,
    fontSize: 18, fontFace: "Calibri", color: C.secondary, margin: 0,
  });
  s.addShape(pres.shapes.LINE, { x: 0.8, y: 4.2, w: 2.5, h: 0, line: { color: C.secondary, width: 2 } });
  s.addText("2026-04-02  |  Cn\u00B2 = 5\u00D710\u207B\u00B9\u2074  |  \u03BB = 1.55\u03BCm  |  5-layer D2NN", {
    x: 0.8, y: 4.45, w: 8, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.textMid, margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.425, w: 10, h: 0.2, fill: { color: C.primary } });
}

// ─────────────────────────────────────────────────────────────────
// Slide 2: Problem Setup
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("Problem: \uB300\uAE30 \uB09C\uB958\uC640 \uBE54 \uC9D1\uC18D \uD488\uC9C8", {
    x: 0.6, y: 0.3, w: 9, h: 0.6, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // Left column
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.15, w: 4.6, h: 3.6, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("\uB300\uAE30 \uB09C\uB958\uC758 \uC601\uD5A5", {
    x: 0.85, y: 1.3, w: 4.1, h: 0.4, fontSize: 16, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s.addText([
    { text: "1 km \uC804\uD30C \uD6C4 \uC218\uC2E0 \uB808\uC774\uC800 \uBE54\uC740 \uB300\uAE30 \uB09C\uB958\uB85C \uD30C\uBA74 \uC654\uACE1", options: { bullet: true, breakLine: true, fontSize: 13.5 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "\uCD08\uC810\uBA74\uC5D0\uC11C \uC5D0\uB108\uC9C0 \uBD84\uC0B0 \u2192 PIB \uC800\uD558", options: { bullet: true, breakLine: true, fontSize: 13.5 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "\uAE30\uC874 AO: \uC2E4\uC2DC\uAC04 \uC13C\uC11C + \uBCC0\uD615 \uBBF8\uB7EC \uD544\uC694 \u2192 \uACE0\uBE44\uC6A9", options: { bullet: true, breakLine: true, fontSize: 13.5 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "D2NN: \uD559\uC2B5\uB41C \uACE0\uC815 \uC704\uC0C1 \uB9C8\uC2A4\uD06C\uB85C \uC218\uB3D9 \uBCF4\uC815", options: { bullet: true, fontSize: 13.5 } },
  ], { x: 0.85, y: 1.8, w: 4.1, h: 2.8, fontFace: "Calibri", color: C.text, valign: "top", margin: 0 });

  // Right column
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 1.15, w: 3.9, h: 3.6, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("\uC2E4\uD5D8 \uC870\uAC74", {
    x: 5.75, y: 1.3, w: 3.4, h: 0.4, fontSize: 16, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  const params = [
    ["\uD30C\uC7A5 (\u03BB)", "1.55 \u03BCm"],
    ["\uC804\uD30C \uAC70\uB9AC", "1 km"],
    ["Cn\u00B2", "5\u00D710\u207B\u00B9\u2074"],
    ["\uB9DD\uC6D0\uACBD \uAD6C\uACBD", "15 cm"],
    ["D2NN \uACA9\uC790", "1024\u00D71024, dx=2\u03BCm"],
    ["D2NN \uB808\uC774\uC5B4", "5 layers, 10mm"],
    ["\uCD08\uC810 \uB80C\uC988", "f = 4.5 mm"],
    ["Loss", "Focal PIB @10\u03BCm"],
  ];
  const tableRows = params.map(([k, v]) => [
    { text: k, options: { fontSize: 12.5, fontFace: "Calibri", color: C.textLight, bold: true } },
    { text: v, options: { fontSize: 12.5, fontFace: "Calibri", color: C.text } },
  ]);
  s.addTable(tableRows, {
    x: 5.75, y: 1.8, w: 3.4, colW: [1.5, 1.9],
    border: { pt: 0.5, color: "E2E8F0" },
    rowH: 0.33,
  });
}

// ─────────────────────────────────────────────────────────────────
// Slide 3: Bug Discovery & Fix
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("Bug Fix: \uB370\uC774\uD130 \uC624\uC5FC \uBB38\uC81C \uBC1C\uACAC \uBC0F \uD574\uACB0", {
    x: 0.6, y: 0.3, w: 9, h: 0.6, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // BEFORE card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.15, w: 4.3, h: 2.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.15, w: 0.07, h: 2.0, fill: { color: C.accent } });
  s.addText("BEFORE (Padded \u2014 \uC624\uC5FC \uB370\uC774\uD130)", {
    x: 0.9, y: 1.25, w: 3.8, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText([
    { text: "Turbulent PIB@10\u03BCm: 21.4%", options: { breakLine: true, fontSize: 13.5 } },
    { text: "D2NN PIB@10\u03BCm: 45.1% (2.1\u00D7)", options: { breakLine: true, fontSize: 13.5 } },
    { text: "WF RMS: 345.8 \u2192 417.8 nm (\uC545\uD654!)", options: { breakLine: true, fontSize: 13.5, color: C.accent, bold: true } },
    { text: "CO: 0.294 \u2192 0.110 (\uC545\uD654!)", options: { fontSize: 13.5, color: C.accent, bold: true } },
  ], { x: 0.9, y: 1.7, w: 3.8, h: 1.3, fontFace: "Calibri", color: C.text, margin: 0 });

  // AFTER card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.15, w: 4.3, h: 2.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.15, w: 0.07, h: 2.0, fill: { color: C.green } });
  s.addText("AFTER (Clean \u2014 \uC815\uC815 \uB370\uC774\uD130)", {
    x: 5.5, y: 1.25, w: 3.8, h: 0.35, fontSize: 14, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  s.addText([
    { text: "Turbulent PIB@10\u03BCm: 80.2%", options: { breakLine: true, fontSize: 13.5 } },
    { text: "D2NN PIB@10\u03BCm: 90.1% (1.12\u00D7)", options: { breakLine: true, fontSize: 13.5 } },
    { text: "WF RMS: 368.9 \u2192 358.5 nm (\uAC1C\uC120!)", options: { breakLine: true, fontSize: 13.5, color: C.green, bold: true } },
    { text: "CO: 0.293 \u2192 0.336 (\uAC1C\uC120!)", options: { fontSize: 13.5, color: C.green, bold: true } },
  ], { x: 5.5, y: 1.7, w: 3.8, h: 1.3, fontFace: "Calibri", color: C.text, margin: 0 });

  // Root cause
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.4, w: 8.9, h: 1.7, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("\uBC84\uADF8 \uC6D0\uC778", {
    x: 0.85, y: 3.5, w: 8.4, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s.addText([
    { text: "Padded \uB370\uC774\uD130: pad_factor \uCC98\uB9AC\uC5D0\uC11C \uD559\uC2B5/\uD3C9\uAC00 \uB370\uC774\uD130 \uC624\uC5FC", options: { bullet: true, breakLine: true, fontSize: 13 } },
    { text: "\uD3C9\uAC00 \uC2DC \uD559\uC2B5 \uB370\uC774\uD130\uB85C \uD14C\uC2A4\uD2B8 \u2192 PIB \uBE44\uC815\uC0C1 \uCE21\uC815 (21.4%)", options: { bullet: true, breakLine: true, fontSize: 13 } },
    { text: "\uD575\uC2EC: \uC774\uC804\uC758 'D2NN\uC774 CO/WF\uB97C \uC545\uD654' \uACB0\uB860\uC740 \uBC84\uADF8 \uB54C\uBB38", options: { bullet: true, fontSize: 13, color: C.accent, bold: true } },
  ], { x: 0.85, y: 3.9, w: 8.4, h: 1.0, fontFace: "Calibri", color: C.text, margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 4: Architecture
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("5-Layer D2NN + Focal PIB Loss", {
    x: 0.6, y: 0.3, w: 9, h: 0.6, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // Pipeline
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.1, w: 5.5, h: 4.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("Optical Pipeline", {
    x: 0.85, y: 1.2, w: 5, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  const pipeSteps = [
    { label: "Turbulent Input", sub: "u_turb (1024\u00D71024)", color: C.turbGray },
    { label: "Receiver Aperture", sub: "\u00D8 2mm circular", color: C.textLight },
    { label: "5-Layer D2NN", sub: "Learned phase masks", color: C.primary },
    { label: "Output Plane", sub: "CO, WF RMS", color: C.secondary },
    { label: "f = 4.5mm Lens", sub: "Fourier transform", color: C.textLight },
    { label: "Focal Plane", sub: "PIB (loss target)", color: C.accent },
  ];
  pipeSteps.forEach((step, i) => {
    const yPos = 1.7 + i * 0.55;
    s.addShape(pres.shapes.RECTANGLE, { x: 1.1, y: yPos, w: 4.6, h: 0.42, fill: { color: step.color }, shadow: mkShadow() });
    s.addText(step.label, { x: 1.2, y: yPos, w: 2.2, h: 0.42, fontSize: 12.5, fontFace: "Calibri", color: C.white, bold: true, valign: "middle", margin: 0 });
    s.addText(step.sub, { x: 3.4, y: yPos, w: 2.2, h: 0.42, fontSize: 11.5, fontFace: "Calibri", color: C.white, valign: "middle", align: "right", margin: [0,0.15,0,0] });
    if (i < pipeSteps.length - 1) {
      s.addText("\u25BC", { x: 3.1, y: yPos + 0.38, w: 0.5, h: 0.2, fontSize: 10, color: C.textLight, align: "center", margin: 0 });
    }
  });

  // Design Choices
  s.addShape(pres.shapes.RECTANGLE, { x: 6.4, y: 1.1, w: 3.2, h: 4.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("Design Choices", {
    x: 6.6, y: 1.2, w: 2.8, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s.addText([
    { text: "Loss Function", options: { bold: true, breakLine: true, fontSize: 13, color: C.primary } },
    { text: "Focal PIB @10\u03BCm \uCD5C\uC801\uD654\n\uCD08\uC810\uBA74 \uC5D0\uB108\uC9C0 \uC9D1\uC911 \uC9C1\uC811 \uD559\uC2B5", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Metric Planes", options: { bold: true, breakLine: true, fontSize: 13, color: C.primary } },
    { text: "Output: CO, WF RMS\nFocal: PIB, Strehl", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Training", options: { bold: true, breakLine: true, fontSize: 13, color: C.primary } },
    { text: "30 epochs, cosine LR\nBatch=8, 5000 pairs", options: { breakLine: true, fontSize: 12 } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "Throughput", options: { bold: true, breakLine: true, fontSize: 13, color: C.primary } },
    { text: "50.6% (non-unitary)", options: { fontSize: 12 } },
  ], { x: 6.6, y: 1.65, w: 2.8, h: 3.3, fontFace: "Calibri", color: C.text, valign: "top", margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 5: Focal Plane Results (fig1)
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("\uCD08\uC810\uBA74 \uBE54 \uD504\uB85C\uD30C\uC77C: Vacuum vs Turbulent vs D2NN", {
    x: 0.5, y: 0.15, w: 9, h: 0.5, fontSize: 22, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });
  // Larger figure
  s.addImage({ path: FIG1, x: 1.2, y: 0.7, w: 7.6, h: 4.7,
    sizing: { type: "contain", w: 7.6, h: 4.7 } });
  s.addText("Row 1: Focal irradiance | Row 2: Log irradiance (4-decade) | Row 3: 1D cross-section | Red circle: PIB@10\u03BCm bucket", {
    x: 0.5, y: 5.35, w: 9, h: 0.2, fontSize: 10, fontFace: "Calibri", color: C.textLight, margin: 0,
  });
}

// ─────────────────────────────────────────────────────────────────
// Slide 6: PIB Multi-Radius (fig2 + analysis) — FIXED overlap
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("PIB Analysis: \uBC18\uACBD\uBCC4 \uC5D0\uB108\uC9C0 \uC9D1\uC911\uB3C4", {
    x: 0.6, y: 0.2, w: 9, h: 0.55, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // fig2
  s.addImage({ path: FIG2, x: 0.3, y: 0.85, w: 5.8, h: 3.1,
    sizing: { type: "contain", w: 5.8, h: 3.1 } });

  // Right: Analysis card — fixed layout with stacked rows
  s.addShape(pres.shapes.RECTANGLE, { x: 6.3, y: 0.85, w: 3.4, h: 3.1, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText("D2NN \uAC1C\uC120 \uD6A8\uACFC", {
    x: 6.5, y: 0.95, w: 3.0, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });

  // Row 1: PIB@5um
  s.addText("PIB@5\u03BCm", { x: 6.5, y: 1.4, w: 1.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.textLight, margin: 0 });
  s.addText("1.28\u00D7", { x: 8.0, y: 1.35, w: 1.5, h: 0.4, fontSize: 26, fontFace: "Georgia", color: C.d2nnRed, bold: true, align: "right", margin: 0 });

  // Row 2: PIB@10um
  s.addText("PIB@10\u03BCm", { x: 6.5, y: 1.9, w: 1.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.textLight, margin: 0 });
  s.addText("1.12\u00D7", { x: 8.0, y: 1.85, w: 1.5, h: 0.4, fontSize: 26, fontFace: "Georgia", color: C.d2nnRed, bold: true, align: "right", margin: 0 });

  // Row 3: Recovery rate
  s.addText("Vacuum \uD68C\uBCF5\uB960", { x: 6.5, y: 2.4, w: 1.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.textLight, margin: 0 });
  s.addText("94%", { x: 8.0, y: 2.35, w: 1.5, h: 0.4, fontSize: 26, fontFace: "Georgia", color: C.green, bold: true, align: "right", margin: 0 });

  // Row 4: Throughput
  s.addText("Throughput", { x: 6.5, y: 2.9, w: 1.5, h: 0.3, fontSize: 12, fontFace: "Calibri", color: C.textLight, margin: 0 });
  s.addText("50.6%", { x: 8.0, y: 2.85, w: 1.5, h: 0.4, fontSize: 26, fontFace: "Georgia", color: C.amber, bold: true, align: "right", margin: 0 });

  // Bottom: interpretation — simplified
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.15, w: 8.9, h: 1.1, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText([
    { text: "@5\u03BCm (1.28\u00D7): D2NN\uC774 \uC0B0\uB780\uAD11\uC744 \uC911\uC2EC \uCF54\uC5B4\uB85C \uC7AC\uBC30\uCE58", options: { bullet: true, breakLine: true, fontSize: 13 } },
    { text: "@10\u03BCm (1.12\u00D7): \uC8FC \uBC84\uD0B7 \uBC18\uACBD, vacuum 95.4% \uB300\uBE44 90.1% \uD68C\uBCF5", options: { bullet: true, breakLine: true, fontSize: 13 } },
    { text: "@50\u03BCm (0.99\u00D7): \uC5D0\uB108\uC9C0\uAC00 \uC548\uCABD\uC73C\uB85C \uC7AC\uBC30\uCE58\uB41C \uACB0\uACFC (throughput 50.6%)", options: { bullet: true, fontSize: 13 } },
  ], { x: 0.85, y: 4.2, w: 8.4, h: 1.0, fontFace: "Calibri", color: C.text, margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 7: WF RMS Analysis — SIMPLIFIED
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("WF RMS: Piston/Tip-Tilt \uC81C\uAC70 \uD6C4 \uBE44\uAD50", {
    x: 0.6, y: 0.2, w: 9, h: 0.55, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // Table — larger, centered
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.95, w: 9, h: 2.3, fill: { color: C.cardBg }, shadow: mkShadow() });
  const hdrOpts = { fontSize: 13, fontFace: "Calibri", color: C.white, bold: true, fill: { color: C.primary }, align: "center", valign: "middle" };
  const cellOpts = { fontSize: 13, fontFace: "Calibri", color: C.text, align: "center", valign: "middle" };
  const greenCell = (txt) => ({ text: txt, options: { ...cellOpts, color: C.green, bold: true } });

  s.addTable([
    [
      { text: "Analysis Method", options: hdrOpts },
      { text: "Vacuum", options: hdrOpts },
      { text: "Turbulent", options: hdrOpts },
      { text: "D2NN", options: hdrOpts },
      { text: "Improvement", options: hdrOpts },
    ],
    [
      { text: "Raw (align phase)", options: { ...cellOpts, bold: true } },
      { text: "0.0 nm", options: cellOpts },
      { text: "368.9 \u00B1 62.7 nm", options: cellOpts },
      { text: "358.5 \u00B1 70.7 nm", options: cellOpts },
      greenCell("\u039410.4 nm"),
    ],
    [
      { text: "Piston removed", options: { ...cellOpts, bold: true } },
      { text: "0.0 nm", options: cellOpts },
      { text: "367.2 \u00B1 63.5 nm", options: cellOpts },
      { text: "356.5 \u00B1 71.3 nm", options: cellOpts },
      greenCell("\u039410.6 nm"),
    ],
    [
      { text: "Piston + Tip/Tilt", options: { ...cellOpts, bold: true } },
      { text: "0.0 nm", options: cellOpts },
      { text: "328.1 \u00B1 101.0 nm", options: cellOpts },
      { text: "304.8 \u00B1 109.4 nm", options: { ...cellOpts, color: C.d2nnRed, bold: true } },
      greenCell("\u039423.2 nm"),
    ],
  ], {
    x: 0.8, y: 1.05, w: 8.6, colW: [1.6, 1.2, 1.7, 1.7, 1.3],
    border: { pt: 0.5, color: "CBD5E1" },
    rowH: [0.4, 0.42, 0.42, 0.42],
  });

  // fig4 — histogram (wider)
  s.addImage({ path: FIG4, x: 0.6, y: 3.45, w: 5.0, h: 2.0,
    sizing: { type: "contain", w: 5.0, h: 2.0 } });

  // Key insight card — right of histogram
  s.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: 3.45, w: 3.8, h: 2.0, fill: { color: "EFF6FF" }, shadow: mkShadow() });
  s.addText("Key Insight", {
    x: 6.0, y: 3.55, w: 3.4, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s.addText([
    { text: "TT \uC81C\uAC70 \uD6C4 D2NN \uAC1C\uC120\uB7C9:", options: { breakLine: true, fontSize: 13 } },
    { text: "10.4nm \u2192 23.2nm (2.2\u00D7)", options: { breakLine: true, fontSize: 16, bold: true, color: C.d2nnRed } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "D2NN\uC740 tip/tilt\uBCF4\uB2E4 \uACE0\uCC28 \uBAA8\uB4DC\n(focus, astigmatism, coma)\n\uBCF4\uC815\uC5D0 2\uBC30 \uB354 \uD6A8\uACFC\uC801", options: { fontSize: 13 } },
  ], { x: 6.0, y: 3.95, w: 3.4, h: 1.3, fontFace: "Calibri", color: C.text, margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 8: Output Plane Phase — FULL-WIDTH FIGURE + bottom strip
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("\uCD9C\uB825\uBA74: Irradiance / Phase / Residual / 1D Profile", {
    x: 0.5, y: 0.12, w: 9, h: 0.45, fontSize: 20, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // fig3 — nearly full-width, tall
  s.addImage({ path: FIG3, x: 0.5, y: 0.6, w: 9.0, h: 4.15,
    sizing: { type: "contain", w: 9.0, h: 4.15 } });

  // Thin annotation strip at bottom
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.85, w: 9.0, h: 0.7, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addText([
    { text: "Residual Phase: ", options: { bold: true, color: C.primary, fontSize: 12.5 } },
    { text: "Turbulent\uC740 \uB300\uADDC\uBAA8 \uC800\uCC28 \uD328\uD134 | D2NN\uC740 \uC794\uCC28\uAC00 \uACE0\uC8FC\uD30C\uC801 \u2192 \uC800\uCC28 \uBAA8\uB4DC \uBCF4\uC815 \uD655\uC778   |   ", options: { fontSize: 12, color: C.text } },
    { text: "CO: 0.293 \u2192 0.336 (+14.7%)", options: { bold: true, fontSize: 12.5, color: C.green } },
  ], { x: 0.7, y: 4.9, w: 8.6, h: 0.55, fontFace: "Calibri", valign: "middle", margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 9: Physical Interpretation — REDUCED DENSITY
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });
  s.addText("D2NN\uC758 \uBB3C\uB9AC\uC801 \uC5ED\uD560", {
    x: 0.6, y: 0.2, w: 9, h: 0.55, fontSize: 26, fontFace: "Georgia", color: C.text, bold: true, margin: 0,
  });

  // Card 1: Fundamental limit
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.0, w: 4.3, h: 2.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.0, w: 0.07, h: 2.0, fill: { color: C.accent } });
  s.addText("Unitary Invariance \uD55C\uACC4", {
    x: 0.9, y: 1.1, w: 3.8, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.accent, bold: true, margin: 0,
  });
  s.addText([
    { text: "\uACE0\uC815 \uC704\uC0C1 \uB9C8\uC2A4\uD06C\uB294 \uB79C\uB364 \uB09C\uB958\uB97C\n'\uC81C\uAC70'\uD560 \uC218 \uC5C6\uC74C (unitary theorem)", options: { breakLine: true, fontSize: 13.5 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\uC2E4\uC81C D2NN\uC740 \uBE44\uC720\uB2C8\uD130\uB9AC\n(throughput = 50.6%)\n\u2192 \uC120\uD0DD\uC801 \uC5D0\uB108\uC9C0 \uC190\uC2E4\uB85C CO \uAC1C\uC120 \uAC00\uB2A5", options: { fontSize: 13.5, bold: true } },
  ], { x: 0.9, y: 1.5, w: 3.8, h: 1.4, fontFace: "Calibri", color: C.text, margin: 0 });

  // Card 2: Energy redistribution
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 2.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 0.07, h: 2.0, fill: { color: C.green } });
  s.addText("\uC5D0\uB108\uC9C0 \uC7AC\uBD84\uBC30 \uBA54\uCEE4\uB2C8\uC998", {
    x: 5.5, y: 1.1, w: 3.8, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
  });
  s.addText([
    { text: "\uC0B0\uB780\uAD11\uC744 \uC911\uC2EC \uCF54\uC5B4\uB85C \uC7AC\uBC30\uCE58\nPIB@5\u03BCm: 52% \u2192 67% (+28%)\nPIB@10\u03BCm: 80% \u2192 90% (+12%)", options: { breakLine: true, fontSize: 13.5 } },
    { text: "", options: { breakLine: true, fontSize: 6 } },
    { text: "\uB300\uAC00: \uC804\uCCB4 \uD30C\uC6CC\uC758 49% \uC190\uC2E4\n\uADF8\uB7EC\uB098 \uB0A8\uC740 \uD30C\uC6CC\uC758 \uC9D1\uC911\uB3C4 \uAC1C\uC120", options: { fontSize: 13.5, bold: true } },
  ], { x: 5.5, y: 1.5, w: 3.8, h: 1.4, fontFace: "Calibri", color: C.text, margin: 0 });

  // Card 3: Mode-selective (full width)
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.25, w: 8.9, h: 2.0, fill: { color: C.cardBg }, shadow: mkShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.25, w: 0.07, h: 2.0, fill: { color: C.primary } });
  s.addText("\uBAA8\uB4DC \uC120\uD0DD\uC801 \uC9D1\uC18D: \uC5B4\uB5A4 \uBE54\uC744 \uC5B4\uB5BB\uAC8C?", {
    x: 0.9, y: 3.35, w: 8.4, h: 0.35, fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });

  // Two sub-columns inside card 3
  s.addText([
    { text: "\uC800\uCC28 \uBAA8\uB4DC (Tip/Tilt)", options: { bold: true, breakLine: true, fontSize: 14, color: C.secondary } },
    { text: "\uCD08\uC810\uBA74\uC5D0\uC11C \uC2A4\uD31F \uC2DC\uD504\uD2B8\uB9CC \uC720\uBC1C\nPIB \uC601\uD5A5 \uC801\uC74C\nD2NN \uBCF4\uC815 \uD544\uC694\uC131 \uB0AE\uC74C", options: { fontSize: 13 } },
  ], { x: 0.9, y: 3.8, w: 4.0, h: 1.2, fontFace: "Calibri", color: C.text, margin: 0 });

  s.addText([
    { text: "\uACE0\uCC28 \uBAA8\uB4DC (Focus, Coma, Astig...)", options: { bold: true, breakLine: true, fontSize: 14, color: C.d2nnRed } },
    { text: "\uC5D0\uB108\uC9C0\uB97C \uCF54\uC5B4 \uBC16\uC73C\uB85C \uC0B0\uB780\nPIB \uC2EC\uAC01\uD558\uAC8C \uC800\uD558\nD2NN\uC774 2\uBC30 \uB354 \uD6A8\uACFC\uC801\uC73C\uB85C \uBCF4\uC815", options: { fontSize: 13, bold: true } },
  ], { x: 5.2, y: 3.8, w: 4.0, h: 1.2, fontFace: "Calibri", color: C.text, margin: 0 });
}

// ─────────────────────────────────────────────────────────────────
// Slide 10: Conclusions — IMPROVED CONTRAST
// ─────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.darkBg };
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.secondary } });

  s.addText("Conclusions", {
    x: 0.8, y: 0.35, w: 8, h: 0.6, fontSize: 32, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
  });
  s.addShape(pres.shapes.LINE, { x: 0.8, y: 1.0, w: 2, h: 0, line: { color: C.secondary, width: 2 } });

  const conclusions = [
    {
      num: "1",
      title: "\uB370\uC774\uD130 \uC815\uC815 \uD6C4 \uBAA8\uB4E0 \uBA54\uD2B8\uB9AD \uAC1C\uC120",
      body: "PIB@10\u03BCm: 80.2% \u2192 90.1% (1.12\u00D7)  |  WF RMS: 369 \u2192 359 nm  |  CO: 0.293 \u2192 0.336",
      accent: C.green,
    },
    {
      num: "2",
      title: "D2NN\uC740 \uACE0\uCC28 \uBAA8\uB4DC \uBCF4\uC815\uC5D0 \uD2B9\uD654",
      body: "Piston+TT \uC81C\uAC70 \uD6C4 \uAC1C\uC120\uB7C9 2.2\u00D7 \uC99D\uAC00 (10.4nm \u2192 23.2nm)  |  Focus, astigmatism, coma \uBCF4\uC815",
      accent: C.secondary,
    },
    {
      num: "3",
      title: "Throughput 50% \uB300\uAC00\uB85C \uC9D1\uC911\uB3C4 \uCD5C\uC801\uD654",
      body: "\uC5D0\uB108\uC9C0 \uC0AC\uC6A9 \uD6A8\uC728: 50.6% \u00D7 90.1% = 45.6% vs Turbulent 80.2%  |  \uC808\uB300\uB7C9 \uAC10\uC18C\uC9C0\uB9CC \uC9D1\uC911\uB3C4 \uC99D\uAC00",
      accent: C.accent,
    },
  ];

  conclusions.forEach((c, i) => {
    const yPos = 1.3 + i * 1.2;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: yPos, w: 8.4, h: 1.0, fill: { color: "152238" } });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: yPos, w: 0.07, h: 1.0, fill: { color: c.accent } });
    s.addShape(pres.shapes.OVAL, { x: 1.15, y: yPos + 0.2, w: 0.55, h: 0.55, fill: { color: c.accent } });
    s.addText(c.num, { x: 1.15, y: yPos + 0.2, w: 0.55, h: 0.55, fontSize: 18, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle", margin: 0 });
    s.addText(c.title, { x: 1.95, y: yPos + 0.08, w: 7, h: 0.35, fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0 });
    // Brighter body text for better contrast
    s.addText(c.body, { x: 1.95, y: yPos + 0.48, w: 7, h: 0.45, fontSize: 12.5, fontFace: "Calibri", color: C.textMid, margin: 0 });
  });

  // Next steps — brighter
  s.addText("Next Steps", {
    x: 0.8, y: 4.85, w: 2, h: 0.3, fontSize: 14, fontFace: "Calibri", color: C.secondary, bold: true, margin: 0,
  });
  s.addText("\uB2E4\uC591\uD55C loss \uC804\uB7B5 \uBE44\uAD50 (CO/IO, Strehl)  |  \uB354 \uAC15\uD55C \uB09C\uB958 \uC870\uAC74  |  \uB2E4\uC911 \uD30C\uC7A5 \uD655\uC7A5", {
    x: 0.8, y: 5.1, w: 8.4, h: 0.3, fontSize: 13, fontFace: "Calibri", color: "B0BEC5", margin: 0,
  });
}

// Save
const outPath = path.join(BASE, "seminar_d2nn_focal_cleanup.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved:", outPath);
}).catch(err => {
  console.error("Error:", err);
});
