const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

// === Color Palette: Ocean Gradient (optics-themed) ===
const C = {
  navy:    "0D1B2A",   // deep navy for title slides
  dark:    "1B2838",   // dark backgrounds
  blue:    "065A82",   // primary blue
  teal:    "1C7293",   // secondary
  accent:  "21B6A8",   // accent green-teal
  light:   "E8F4F8",   // light bg
  white:   "FFFFFF",
  offwhite:"F7FAFB",
  text:    "1E293B",   // dark text
  muted:   "64748B",   // muted text
  red:     "DC2626",   // loss/bad
  green:   "059669",   // gain/good
  orange:  "D97706",   // warning
  purple:  "7C3AED",   // neutral
  gray:    "94A3B8",
  gridline:"E2E8F0",
};

const RUNS = __dirname;
const R0403C = path.join(RUNS, "0403-combined-6strat-pitchrescale-cn2-5e14");
const R0403F = path.join(RUNS, "0403-focal-pib-vacuum-target-pitchrescale-3strat-cn2-5e14");

function imgData(p) {
  if (!fs.existsSync(p)) { console.warn("Missing:", p); return null; }
  const ext = path.extname(p).slice(1).toLowerCase();
  const mime = ext === "jpg" ? "jpeg" : ext;
  return `image/${mime};base64,` + fs.readFileSync(p).toString("base64");
}

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "D2NN Lab";
pres.title = "D2NN Loss Function Sweep — 0403 Experiments";

// =========================================================
// SLIDE 1: Title
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  // Left accent bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.12, h: 5.625, fill: { color: C.accent } });

  s.addText("Static D2NN\nLoss Function Sweep", {
    x: 0.8, y: 0.8, w: 8, h: 2.2,
    fontSize: 42, fontFace: "Arial Black", color: C.white, bold: true, lineSpacingMultiple: 0.9,
  });
  s.addText("Practical Limits & Optimal Loss Strategy", {
    x: 0.8, y: 2.9, w: 8, h: 0.6,
    fontSize: 20, fontFace: "Calibri", color: C.accent, italic: true,
  });
  // Separator line
  s.addShape(pres.shapes.LINE, { x: 0.8, y: 3.6, w: 3, h: 0, line: { color: C.accent, width: 3 } });
  s.addText("D2NN Lab  |  2026-04-03", {
    x: 0.8, y: 3.9, w: 6, h: 0.5,
    fontSize: 14, fontFace: "Calibri", color: C.muted,
  });
  s.addText([
    { text: "FSO Beam Cleanup  ", options: { color: C.gray } },
    { text: "Cn\u00B2=5\u00D710\u207B\u00B9\u2074, L=1km, \u03BB=1.55\u03BCm", options: { color: C.muted } },
  ], { x: 0.8, y: 4.4, w: 8, h: 0.4, fontSize: 12, fontFace: "Calibri" });
}

// =========================================================
// SLIDE 2: Experiment Overview
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.light };

  s.addText("Experiment Overview", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: C.text, margin: 0,
  });
  s.addText("10 loss strategies across 3 sweeps, 2 datasets", {
    x: 0.5, y: 0.95, w: 9, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.muted,
  });

  // Table
  const hdrOpts = { fill: { color: C.blue }, color: C.white, bold: true, fontSize: 13, fontFace: "Calibri", align: "left", valign: "middle" };
  const cellOpts = { fill: { color: C.white }, color: C.text, fontSize: 13, fontFace: "Calibri", align: "left", valign: "middle" };
  s.addTable([
    [
      { text: "Sweep", options: hdrOpts },
      { text: "Data", options: hdrOpts },
      { text: "Strategies", options: hdrOpts },
      { text: "Status", options: hdrOpts },
    ],
    [
      { text: "0401", options: cellOpts },
      { text: "dn100um (Lanczos)", options: cellOpts },
      { text: "PIB, Strehl, IO, CO+PIB", options: cellOpts },
      { text: "Complete", options: { ...cellOpts, color: C.green, bold: true } },
    ],
    [
      { text: "0402", options: cellOpts },
      { text: "pitch_rescale", options: cellOpts },
      { text: "Abs Bucket (F), TP+PIB (B), Multiplane (C)", options: cellOpts },
      { text: "Complete", options: { ...cellOpts, color: C.green, bold: true } },
    ],
    [
      { text: "0403", options: cellOpts },
      { text: "pitch_rescale", options: cellOpts },
      { text: "CO+hardTP, AbsBkt+CO, PIB+hardTP", options: cellOpts },
      { text: "Complete", options: { ...cellOpts, color: C.green, bold: true } },
    ],
  ], {
    x: 0.5, y: 1.5, w: 9,
    border: { pt: 0.5, color: C.gridline },
    colW: [1.0, 2.2, 4.0, 1.2],
    rowH: [0.45, 0.45, 0.45, 0.45],
  });

  // Stat callouts
  const stats = [
    { n: "10", label: "Loss Strategies", color: C.blue },
    { n: "5000", label: "Training Pairs", color: C.orange },
    { n: "30", label: "Epochs Each", color: C.teal },
    { n: "~3h", label: "Per Strategy (A100)", color: C.purple },
  ];
  stats.forEach((st, i) => {
    const xOff = 0.5 + i * 2.3;
    s.addShape(pres.shapes.RECTANGLE, {
      x: xOff, y: 3.7, w: 2.0, h: 1.5,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 4, offset: 2, angle: 135, opacity: 0.08 },
    });
    s.addText(st.n, {
      x: xOff, y: 3.85, w: 2.0, h: 0.75,
      fontSize: 36, fontFace: "Arial Black", color: st.color, align: "center", valign: "middle",
    });
    s.addText(st.label, {
      x: xOff, y: 4.55, w: 2.0, h: 0.5,
      fontSize: 12, fontFace: "Calibri", color: C.muted, align: "center", valign: "top",
    });
  });

  // Architecture spec
  s.addText("5-layer D2NN  |  spacing=10mm  |  f=4.5mm  |  N=1024  |  dx=2\u03BCm  |  pad=2", {
    x: 0.5, y: 5.2, w: 9, h: 0.3,
    fontSize: 11, fontFace: "Consolas", color: C.muted, align: "center",
  });
}

// =========================================================
// SLIDE 3: Key Finding — Normalized PIB Lies
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Normalized PIB Lies", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: C.red, margin: 0,
  });
  s.addText("High PIB does NOT mean high received power", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.muted,
  });

  const hdr = { fill: { color: C.blue }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri", align: "center", valign: "middle" };
  const cc = (txt, color) => ({ text: txt, options: { fill: { color: C.white }, color: color || C.text, fontSize: 12, fontFace: "Calibri", align: "center", valign: "middle", bold: color ? true : false } });

  s.addTable([
    [
      { text: "Strategy", options: { ...hdr, align: "left" } },
      { text: "PIB@10\u03BCm", options: hdr },
      { text: "Throughput", options: hdr },
      { text: "Abs Power\nvs Turb", options: hdr },
      { text: "Verdict", options: hdr },
    ],
    [
      { text: "PIB Only (0401)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("90.1%", C.green),
      cc("50.7%", C.red),
      cc("58.7%", C.red),
      cc("LOSS", C.red),
    ],
    [
      { text: "Strehl Only (0401)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("81.8%"),
      cc("57.5%", C.red),
      cc("60.4%", C.red),
      cc("LOSS", C.red),
    ],
    [
      { text: "Abs Bucket F (0402)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("81.4%"),
      cc("98.6%", C.green),
      cc("105.8%", C.green),
      cc("GAIN", C.green),
    ],
    [
      { text: "TP+PIB B (0402)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("81.6%"),
      cc("98.4%", C.green),
      cc("105.8%", C.green),
      cc("GAIN", C.green),
    ],
    [
      { text: "Multiplane C (0402)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("84.2%"),
      cc("39.5%", C.red),
      cc("44.2%", C.red),
      cc("LOSS", C.red),
    ],
    [
      { text: "CO+hardTP (0403)", options: { fill: { color: C.white }, color: C.text, fontSize: 12, fontFace: "Calibri", align: "left", valign: "middle" } },
      cc("74.0%"),
      cc("99.3%", C.green),
      cc("97.0%", C.orange),
      cc("~NEUTRAL", C.orange),
    ],
  ], {
    x: 0.5, y: 1.45, w: 9,
    border: { pt: 0.5, color: C.gridline },
    colW: [2.2, 1.4, 1.5, 1.6, 1.3],
    rowH: [0.45, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
  });

  // Key insight box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.5, w: 9, h: 0.9,
    fill: { color: "FFF8E1" },
    line: { color: C.orange, width: 1.5 },
  });
  s.addText([
    { text: "Key Insight: ", options: { bold: true, color: C.red } },
    { text: "PIB Only achieved 90% PIB but absolute received power was only 58.7% of turbulent baseline. ", options: { color: C.text } },
    { text: "Throughput is the real metric.", options: { bold: true, color: C.red } },
  ], {
    x: 0.7, y: 4.55, w: 8.6, h: 0.8,
    fontSize: 13, fontFace: "Calibri", valign: "middle",
  });
}

// =========================================================
// SLIDE 4: Why Does Throughput Drop?
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addText("Why Does Throughput Drop?", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: C.white, margin: 0,
  });
  s.addText("Spatial Filtering Mechanism", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.accent,
  });

  const steps = [
    { n: "1", title: "Phase Mask Pattern", desc: "D2NN learns high-frequency phase gratings to redirect energy" },
    { n: "2", title: "Angular Vignetting", desc: "Diffracted light at large angles exceeds finite aperture (dx=2\u03BCm \u226B \u03BB/2=775nm)" },
    { n: "3", title: "Aperture Spatial Filter", desc: "Energy scattered beyond computational domain is lost at each layer" },
    { n: "4", title: "Energy Lost", desc: "Turbulent beams lose MORE energy (50%) than vacuum (20-30%)" },
  ];

  steps.forEach((st, i) => {
    const yOff = 1.5 + i * 0.95;
    // Number circle
    s.addShape(pres.shapes.OVAL, {
      x: 0.8, y: yOff, w: 0.55, h: 0.55,
      fill: { color: C.accent },
    });
    s.addText(st.n, {
      x: 0.8, y: yOff, w: 0.55, h: 0.55,
      fontSize: 18, fontFace: "Arial Black", color: C.white, align: "center", valign: "middle",
    });
    s.addText(st.title, {
      x: 1.55, y: yOff, w: 7, h: 0.3,
      fontSize: 18, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(st.desc, {
      x: 1.55, y: yOff + 0.3, w: 7, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.muted, margin: 0,
    });
  });

  // Bottom callout
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.0, w: 5.5, h: 0.45,
    fill: { color: C.accent },
  });
  s.addText("D2NN acts as aperture-based spatial filter \u2014 diffracts energy beyond collection aperture", {
    x: 0.7, y: 5.0, w: 5.3, h: 0.45,
    fontSize: 11, fontFace: "Calibri", color: C.white, bold: true, valign: "middle",
  });
}

// =========================================================
// SLIDE 5: Received Power Gap Analysis
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Received Power Gap Decomposition", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial Black", color: C.text, margin: 0,
  });

  // Left: gap table
  const hdr = { fill: { color: C.blue }, color: C.white, bold: true, fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle" };
  const cv = (t, c) => ({ text: t, options: { fill: { color: C.white }, color: c || C.text, fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle", bold: c ? true : false } });

  s.addTable([
    [
      { text: "", options: hdr },
      { text: "Total Power", options: hdr },
      { text: "Bucket@10\u03BCm", options: hdr },
      { text: "PIB", options: hdr },
    ],
    [ cv("Vacuum"), cv("290.18"), cv("277.17"), cv("95.5%") ],
    [ cv("Turbulent"), cv("278.67"), cv("217.95"), cv("78.1%") ],
    [
      cv("Gap", C.red),
      cv("11.51 (4.0%)", C.orange),
      cv("59.22 (21.4%)", C.red),
      cv("17.4%p", C.red),
    ],
  ], {
    x: 0.5, y: 1.2, w: 5.5,
    border: { pt: 0.5, color: C.gridline },
    colW: [1.2, 1.5, 1.5, 1.0],
    rowH: [0.4, 0.38, 0.38, 0.42],
  });

  // Right: physics equations
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.3, y: 1.2, w: 3.4, h: 2.8,
    fill: { color: C.white },
    shadow: { type: "outer", color: "000000", blur: 4, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addText("Physical Parameters", {
    x: 6.5, y: 1.25, w: 3.0, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.blue, bold: true, margin: 0,
  });
  s.addText([
    { text: "WFE = 442.9 nm (2.87\u03BB)\n", options: { breakLine: true } },
    { text: "Mar\u00E9chal limit = \u03BB/14 = 111nm\n", options: { breakLine: true } },
    { text: "Strehl = exp(-\u03C3\u00B2) = 0.04\n", options: { breakLine: true } },
    { text: "\n", options: { breakLine: true, fontSize: 4 } },
    { text: "Gap origin:\n", options: { bold: true, breakLine: true } },
    { text: "\u2022 WFE spreading: 81.8%\n", options: { breakLine: true } },
    { text: "\u2022 Throughput loss: 18.2%", options: {} },
  ], {
    x: 6.5, y: 1.6, w: 3.0, h: 2.3,
    fontSize: 11, fontFace: "Consolas", color: C.text, margin: 0,
  });

  // Bottom: Bucket Power formula
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.5, w: 9, h: 0.7,
    fill: { color: C.dark },
  });
  s.addText("Bucket Power  \u2248  TP  \u00D7  exp(-\u03C3\u00B2_WFE)  \u00D7  P_diff.limited", {
    x: 0.5, y: 3.5, w: 9, h: 0.7,
    fontSize: 18, fontFace: "Consolas", color: C.accent, align: "center", valign: "middle",
  });

  // Static D2NN limit box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.5, w: 9, h: 0.8,
    fill: { color: "FFF0F0" },
    line: { color: C.red, width: 1.5 },
  });
  s.addText([
    { text: "Structural Limit: ", options: { bold: true, color: C.red } },
    { text: "Each sample has random WFE from different turbulence realization. Fixed masks cannot correct random per-sample error \u2192 fundamental constraint.", options: { color: C.text } },
  ], {
    x: 0.7, y: 4.55, w: 8.6, h: 0.7,
    fontSize: 12, fontFace: "Calibri", valign: "middle",
  });
}

// =========================================================
// SLIDE 6: 0403 Combined Sweep Results
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("0403 Combined 6-Strategy Results", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 32, fontFace: "Arial Black", color: C.text, margin: 0,
  });
  s.addText("Focal-plane PIB@10\u03BCm with pitch_rescale data, Cn\u00B2=5\u00D710\u207B\u00B9\u2074", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.muted,
  });

  // Results chart — bar chart
  const strategies = ["Vacuum", "Turbulent", "AbsBkt+CO", "PIB+HardTP", "TP-PIB\nw=0.5", "TP-PIB\nw=2.0", "Raw RP"];
  const pib10 = [95.98, 77.11, 80.52, 80.42, 81.32, 80.41, 80.83];
  const throughput = [97.68, 97.00, 98.30, 98.92, 96.94, 98.93, 98.59];

  s.addChart(pres.charts.BAR, [
    { name: "PIB@10\u03BCm (%)", labels: strategies, values: pib10 },
  ], {
    x: 0.3, y: 1.4, w: 4.8, h: 3.0, barDir: "col",
    showTitle: true, title: "Focal PIB@10\u03BCm (%)",
    titleColor: C.text, titleFontSize: 12,
    chartColors: [C.blue],
    chartArea: { fill: { color: C.white }, roundedCorners: true },
    catAxisLabelColor: C.muted, catAxisLabelFontSize: 9,
    valAxisLabelColor: C.muted,
    valGridLine: { color: C.gridline, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.text, dataLabelFontSize: 9,
    valAxisMinVal: 60, valAxisMaxVal: 100,
  });

  s.addChart(pres.charts.BAR, [
    { name: "Throughput (%)", labels: strategies, values: throughput },
  ], {
    x: 5.2, y: 1.4, w: 4.6, h: 3.0, barDir: "col",
    showTitle: true, title: "Throughput (%)",
    titleColor: C.text, titleFontSize: 12,
    chartColors: [C.accent],
    chartArea: { fill: { color: C.white }, roundedCorners: true },
    catAxisLabelColor: C.muted, catAxisLabelFontSize: 9,
    valAxisLabelColor: C.muted,
    valGridLine: { color: C.gridline, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.text, dataLabelFontSize: 9,
    valAxisMinVal: 90, valAxisMaxVal: 100,
  });

  // Key takeaway
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.7, w: 9, h: 0.7,
    fill: { color: "E8F5E9" },
    line: { color: C.green, width: 1 },
  });
  s.addText([
    { text: "Best strategy: ", options: { bold: true, color: C.green } },
    { text: "TP-PIB w=0.5 achieves highest PIB (81.3%) while maintaining 96.9% throughput. All 0403 strategies preserve throughput >96%.", options: { color: C.text } },
  ], {
    x: 0.7, y: 4.75, w: 8.6, h: 0.6,
    fontSize: 12, fontFace: "Calibri", valign: "middle",
  });
}

// =========================================================
// SLIDE 7: Absolute Received Power Histogram (image)
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Absolute Received Power Distribution", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });
  s.addText("Turbulent (gray) vs D2NN (color) \u2014 500 test samples, bucket @10\u03BCm", {
    x: 0.5, y: 0.75, w: 9, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.muted,
  });

  // Cross-strategy comparison image
  const crossImg = imgData(path.join(R0403C, "19_cross_strategy_received_power.png"));
  if (crossImg) {
    s.addImage({ data: crossImg, x: 0.3, y: 1.1, w: 9.4, h: 3.2, sizing: { type: "contain", w: 9.4, h: 3.2 } });
  }

  // Bottom stats
  const stats0403 = [
    { label: "AbsBkt+CO", val: "+4.0%", color: C.green },
    { label: "PIB+HardTP", val: "+3.3%", color: C.green },
    { label: "TP-PIB w=0.5", val: "+5.2%", color: C.green },
    { label: "TP-PIB w=2.0", val: "+3.3%", color: C.green },
    { label: "Raw RP", val: "+3.8%", color: C.green },
  ];
  stats0403.forEach((st, i) => {
    const xOff = 0.3 + i * 1.9;
    s.addShape(pres.shapes.RECTANGLE, {
      x: xOff, y: 4.5, w: 1.7, h: 0.9,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 3, offset: 1, angle: 135, opacity: 0.06 },
    });
    s.addText(st.val, {
      x: xOff, y: 4.52, w: 1.7, h: 0.5,
      fontSize: 20, fontFace: "Arial Black", color: st.color, align: "center", valign: "middle",
    });
    s.addText(st.label, {
      x: xOff, y: 5.0, w: 1.7, h: 0.35,
      fontSize: 10, fontFace: "Calibri", color: C.muted, align: "center",
    });
  });
}

// =========================================================
// SLIDE 8: Encircled Energy
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Encircled Energy Analysis", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });

  const eeImg = imgData(path.join(R0403C, "20_encircled_energy_per_strategy.png"));
  if (eeImg) {
    s.addImage({ data: eeImg, x: 0.3, y: 0.85, w: 9.4, h: 4.2, sizing: { type: "contain", w: 9.4, h: 4.2 } });
  }

  s.addText("PIB computed at 4 radii: 5\u03BCm, 10\u03BCm, 25\u03BCm, 50\u03BCm | All strategies converge at r=50\u03BCm", {
    x: 0.5, y: 5.1, w: 9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center",
  });
}

// =========================================================
// SLIDE 9: CO Loss Limitation (0403 co_hard_tp)
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("CO Loss: High Throughput, Low PIB", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });
  s.addText("CO+hardTP (0403) \u2014 Complex Overlap + Quadratic TP Penalty (\u03BB=10)", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.muted,
  });

  // Left: stats
  const coStats = [
    { label: "Throughput", value: "99.3%", desc: "Best across all strategies", color: C.green },
    { label: "PIB@10\u03BCm", value: "74.0%", desc: "Below turbulent baseline (76.1%)", color: C.red },
    { label: "Abs Power", value: "97.0%", desc: "3% loss vs turbulent", color: C.orange },
    { label: "Strehl", value: "0.49", desc: "Far below diffraction limit", color: C.muted },
  ];

  coStats.forEach((st, i) => {
    const yOff = 1.5 + i * 0.85;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: yOff, w: 4.5, h: 0.7,
      fill: { color: C.white },
      shadow: { type: "outer", color: "000000", blur: 3, offset: 1, angle: 135, opacity: 0.06 },
    });
    s.addText(st.value, {
      x: 0.7, y: yOff, w: 1.2, h: 0.7,
      fontSize: 24, fontFace: "Arial Black", color: st.color, valign: "middle",
    });
    s.addText(st.label, {
      x: 2.0, y: yOff + 0.05, w: 2.8, h: 0.3,
      fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, valign: "middle", margin: 0,
    });
    s.addText(st.desc, {
      x: 2.0, y: yOff + 0.35, w: 2.8, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.muted, valign: "middle", margin: 0,
    });
  });

  // Right: phase mask image
  const phaseImg = imgData(path.join(R0403F, "co_hard_tp/15_phase_masks_5layers.png"));
  if (phaseImg) {
    s.addImage({ data: phaseImg, x: 5.3, y: 1.5, w: 4.4, h: 2.2, sizing: { type: "contain", w: 4.4, h: 2.2 } });
  }
  s.addText("Phase masks \u2014 nearly flat (minimal spatial filtering)", {
    x: 5.3, y: 3.7, w: 4.4, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center",
  });

  // Explanation
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.6, w: 9, h: 0.8,
    fill: { color: "FFF8E1" },
    line: { color: C.orange, width: 1 },
  });
  s.addText([
    { text: "Analysis: ", options: { bold: true, color: C.orange } },
    { text: "CO optimizes entire field matching, not bucket concentration. No gradient drives energy into the bucket. Highest TP but worst PIB \u2014 confirms CO alone is insufficient for FSO beam cleanup.", options: { color: C.text } },
  ], {
    x: 0.7, y: 4.65, w: 8.6, h: 0.7,
    fontSize: 12, fontFace: "Calibri", valign: "middle",
  });
}

// =========================================================
// SLIDE 10: Deep Fade / Scintillation Analysis
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Deep Fade & Scintillation Analysis", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });

  // Left: Scintillation table
  s.addText("Scintillation Statistics (dB, vacuum mean = 0 dB)", {
    x: 0.5, y: 0.85, w: 4.5, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.blue, bold: true,
  });

  const h = { fill: { color: C.blue }, color: C.white, bold: true, fontSize: 10, fontFace: "Calibri", align: "center", valign: "middle" };
  const c = (t, clr) => ({ text: t, options: { fill: { color: C.white }, color: clr || C.text, fontSize: 10, fontFace: "Calibri", align: "center", valign: "middle", bold: clr ? true : false } });

  s.addTable([
    [{ text: "", options: h }, { text: "Mean", options: h }, { text: "Std", options: h }, { text: "Min", options: h }, { text: "1%ile", options: h }, { text: "5%ile", options: h }],
    [c("Vacuum"), c("0.00"), c("0.00"), c("0.00"), c("0.00"), c("0.00")],
    [c("Turbulent"), c("-1.12"), c("0.86"), c("-7.73", C.red), c("-3.72"), c("-2.45")],
    [c("D2NN (F)"), c("-0.85"), c("0.74"), c("-6.36", C.green), c("-3.25"), c("-2.00")],
  ], {
    x: 0.5, y: 1.15, w: 4.8,
    border: { pt: 0.5, color: C.gridline },
    colW: [0.9, 0.75, 0.65, 0.8, 0.8, 0.8],
    rowH: [0.3, 0.28, 0.28, 0.28],
  });

  // Right: Deep fade cases
  s.addText("Deep Fade Worst Cases", {
    x: 5.5, y: 0.85, w: 4, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.blue, bold: true,
  });

  s.addTable([
    [{ text: "Rank", options: h }, { text: "Turb", options: h }, { text: "D2NN", options: h }, { text: "Gain", options: h }],
    [c("1"), c("-7.73 dB"), c("-6.36 dB"), c("+1.37 dB", C.green)],
    [c("2"), c("-6.38 dB"), c("-5.20 dB"), c("+1.18 dB", C.green)],
    [c("3"), c("-6.28 dB"), c("-5.05 dB"), c("+1.24 dB", C.green)],
    [c("8"), c("-3.68 dB"), c("-2.66 dB"), c("+1.02 dB", C.green)],
  ], {
    x: 5.5, y: 1.15, w: 4.0,
    border: { pt: 0.5, color: C.gridline },
    colW: [0.6, 1.1, 1.1, 1.1],
    rowH: [0.3, 0.28, 0.28, 0.28, 0.28],
  });

  // Fade Probability
  s.addText("Fade Probability", {
    x: 5.5, y: 2.85, w: 4, h: 0.3,
    fontSize: 12, fontFace: "Calibri", color: C.blue, bold: true,
  });
  s.addTable([
    [{ text: "Threshold", options: h }, { text: "Turbulent", options: h }, { text: "D2NN (F)", options: h }],
    [c("< -3 dB"), c("2.8%"), c("1.8%", C.green)],
    [c("< -5 dB"), c("0.6%"), c("0.6%")],
    [c("< -7 dB"), c("0.2%"), c("0.0%", C.green)],
  ], {
    x: 5.5, y: 3.15, w: 4.0,
    border: { pt: 0.5, color: C.gridline },
    colW: [1.2, 1.4, 1.4],
    rowH: [0.3, 0.28, 0.28, 0.28],
  });

  // Bottom insight
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.7, w: 9, h: 0.6,
    fill: { color: "E8F5E9" },
    line: { color: C.green, width: 1 },
  });
  s.addText([
    { text: "D2NN is more effective in deep fades (+1.37 dB worst case) than average (+0.24 dB). ", options: { color: C.text } },
    { text: "Static D2NN acts as a \"floor raiser\" for worst-case link budget.", options: { bold: true, color: C.green } },
  ], {
    x: 0.7, y: 4.73, w: 8.6, h: 0.55,
    fontSize: 12, fontFace: "Calibri", valign: "middle",
  });
}

// =========================================================
// SLIDE 11: Zernike Mode Analysis
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Zernike Mode Analysis", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });
  s.addText("Wavefront correction capability by loss strategy", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 13, fontFace: "Calibri", color: C.muted,
  });

  // Zernike table
  const zh = { fill: { color: C.blue }, color: C.white, bold: true, fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle" };
  const zc = (t, clr) => ({ text: t, options: { fill: { color: C.white }, color: clr || C.text, fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle", bold: clr ? true : false } });

  s.addTable([
    [{ text: "Strategy", options: { ...zh, align: "left" } }, { text: "Higher-order \u0394", options: zh }, { text: "Tip/Tilt \u0394", options: zh }, { text: "Throughput", options: zh }],
    [
      { text: "F/B (Abs Bucket)", options: { fill: { color: C.white }, color: C.text, fontSize: 13, fontFace: "Calibri", align: "left", valign: "middle" } },
      zc("\u00B11 nm"), zc("\u00B10.2 nm"), zc("98%", C.green),
    ],
    [
      { text: "Multiplane C", options: { fill: { color: C.white }, color: C.text, fontSize: 13, fontFace: "Calibri", align: "left", valign: "middle" } },
      zc("-28 nm", C.green), zc("+1.3 nm", C.orange), zc("40%", C.red),
    ],
    [
      { text: "PIB Only", options: { fill: { color: C.white }, color: C.text, fontSize: 13, fontFace: "Calibri", align: "left", valign: "middle" } },
      zc("-28 nm", C.green), zc("+16 nm", C.red), zc("51%", C.red),
    ],
  ], {
    x: 0.5, y: 1.5, w: 9,
    border: { pt: 0.5, color: C.gridline },
    colW: [2.5, 2.0, 2.0, 1.5],
    rowH: [0.45, 0.42, 0.42, 0.42],
  });

  // Tradeoff diagram
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 3.2, w: 9, h: 0.5,
    fill: { color: C.dark },
  });
  s.addText("PIB  \u2194  Throughput : Structural Tradeoff", {
    x: 0.5, y: 3.2, w: 9, h: 0.5,
    fontSize: 18, fontFace: "Arial Black", color: C.accent, align: "center", valign: "middle",
  });

  // Explanation
  s.addText([
    { text: "\u2022 When TP is preserved (F/B), Zernike delta \u2248 0 \u2014 no wavefront correction\n", options: { breakLine: true } },
    { text: "\u2022 When TP drops (PIB Only), higher-order corrected -28 nm but tip/tilt added +16 nm\n", options: { breakLine: true } },
    { text: "\u2022 Static mask trades one aberration type for another via mode conversion\n", options: { breakLine: true } },
    { text: "\u2022 This confirms: Fixed masks cannot correct random wavefront errors", options: {} },
  ], {
    x: 0.7, y: 3.85, w: 8.6, h: 1.5,
    fontSize: 12, fontFace: "Calibri", color: C.text,
    paraSpaceAfter: 4,
  });
}

// =========================================================
// SLIDE 12: Phase Mask Comparison
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Phase Mask Comparison", {
    x: 0.5, y: 0.2, w: 9, h: 0.6,
    fontSize: 30, fontFace: "Arial Black", color: C.text, margin: 0,
  });

  // Top row: AbsBkt+CO
  const abImg = imgData(path.join(R0403C, "abs_bucket_plus_co/15_phase_masks_5layers.png"));
  if (abImg) {
    s.addImage({ data: abImg, x: 0.3, y: 0.9, w: 9.4, h: 1.8, sizing: { type: "contain", w: 9.4, h: 1.8 } });
  }
  s.addText("AbsBkt+CO \u2014 nearly flat masks (TP=98.3%)", {
    x: 0.5, y: 2.65, w: 9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center",
  });

  // Bottom row: PIB+HardTP
  const pibImg = imgData(path.join(R0403C, "pib_hard_tp/15_phase_masks_5layers.png"));
  if (pibImg) {
    s.addImage({ data: pibImg, x: 0.3, y: 3.0, w: 9.4, h: 1.8, sizing: { type: "contain", w: 9.4, h: 1.8 } });
  }
  s.addText("PIB+HardTP \u2014 also flat masks due to strong TP penalty (TP=98.9%)", {
    x: 0.5, y: 4.75, w: 9, h: 0.3,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center",
  });

  s.addText("All 0403 strategies learn conservative masks \u2014 TP penalty prevents aggressive spatial filtering", {
    x: 0.5, y: 5.15, w: 9, h: 0.35,
    fontSize: 12, fontFace: "Calibri", color: C.blue, bold: true, align: "center",
  });
}

// =========================================================
// SLIDE 13: Absolute Bucket Loss Function
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addText("Optimal Loss Function", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: C.white, margin: 0,
  });
  s.addText("Absolute Bucket Loss \u2014 direct fiber-coupling optimization", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.accent,
  });

  // Formula box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.0, y: 1.6, w: 8, h: 1.2,
    fill: { color: "0D2137" },
    line: { color: C.accent, width: 2 },
  });
  s.addText("L_abs  =  1  -  \u03A3_{r\u226410\u03BCm} |E_pred^focal|\u00B2  /  \u03A3_{r\u226410\u03BCm} |E_vac^focal|\u00B2", {
    x: 1.0, y: 1.6, w: 8, h: 1.2,
    fontSize: 22, fontFace: "Consolas", color: C.accent, align: "center", valign: "middle",
  });

  // Why it works
  const reasons = [
    { title: "Throughput-aware", desc: "Normalizing by vacuum bucket power captures both TP and PIB" },
    { title: "Gradient-aligned", desc: "Every photon directed into bucket reduces loss directly" },
    { title: "Physics-faithful", desc: "Bucket Power \u221D TP \u00D7 Strehl \u00D7 P_diff \u2192 loss aligns with all three" },
  ];

  reasons.forEach((r, i) => {
    const yOff = 3.1 + i * 0.75;
    s.addShape(pres.shapes.OVAL, {
      x: 1.0, y: yOff + 0.05, w: 0.4, h: 0.4,
      fill: { color: C.accent },
    });
    s.addText(String(i + 1), {
      x: 1.0, y: yOff + 0.05, w: 0.4, h: 0.4,
      fontSize: 14, fontFace: "Arial Black", color: C.white, align: "center", valign: "middle",
    });
    s.addText(r.title, {
      x: 1.6, y: yOff, w: 3, h: 0.3,
      fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, margin: 0,
    });
    s.addText(r.desc, {
      x: 1.6, y: yOff + 0.3, w: 7, h: 0.3,
      fontSize: 12, fontFace: "Calibri", color: C.gray, margin: 0,
    });
  });
}

// =========================================================
// SLIDE 14: Conclusions & Next Steps
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.offwhite };

  s.addText("Conclusions", {
    x: 0.5, y: 0.3, w: 9, h: 0.6,
    fontSize: 36, fontFace: "Arial Black", color: C.text, margin: 0,
  });

  const conclusions = [
    { num: "1", text: "Throughput is the real performance metric. Normalized PIB is misleading." },
    { num: "2", text: "Static D2NN absolute power improvement: +5.8% (+0.24 dB) \u2014 Abs Bucket (F) / TP+PIB (B)." },
    { num: "3", text: "PIB \u2194 Throughput structural tradeoff. Energy concentration via scatter \u2192 throughput loss unavoidable." },
    { num: "4", text: "Vacuum PIB (94.2%) is TP-preserved ceiling. WFE=442.9nm (random per-sample) \u2192 fixed masks can't correct." },
    { num: "5", text: "Deep fade improvement: +1.37 dB worst case. Static D2NN is a \"floor raiser\"." },
    { num: "6", text: "Absolute bucket loss is the optimal objective. Bucket Power \u221D TP \u00D7 Strehl \u00D7 P_diff." },
  ];

  conclusions.forEach((c, i) => {
    const yOff = 1.0 + i * 0.65;
    s.addShape(pres.shapes.OVAL, {
      x: 0.5, y: yOff + 0.08, w: 0.4, h: 0.4,
      fill: { color: C.blue },
    });
    s.addText(c.num, {
      x: 0.5, y: yOff + 0.08, w: 0.4, h: 0.4,
      fontSize: 16, fontFace: "Arial Black", color: C.white, align: "center", valign: "middle",
    });
    s.addText(c.text, {
      x: 1.1, y: yOff, w: 8.3, h: 0.55,
      fontSize: 14, fontFace: "Calibri", color: C.text, valign: "middle", margin: 0,
    });
  });

  // Next Steps
  s.addShape(pres.shapes.LINE, { x: 0.5, y: 5.0, w: 9, h: 0, line: { color: C.gridline, width: 1 } });
  s.addText([
    { text: "Next: ", options: { bold: true, color: C.blue } },
    { text: "Stronger turbulence (Cn\u00B2=10\u207B\u00B9\u00B3) | Switchable static masks | Hybrid AO+D2NN | Dynamic D2NN (Zernike-based)", options: { color: C.muted } },
  ], {
    x: 0.5, y: 5.1, w: 9, h: 0.4,
    fontSize: 12, fontFace: "Calibri",
  });
}

// =========================================================
// SLIDE 15: Physics Verification (placeholder — will be filled after Codex review)
// =========================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };

  s.addText("Physics Verification", {
    x: 0.5, y: 0.3, w: 9, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: C.white, margin: 0,
  });
  s.addText("Expert review of equations and physical claims (Codex/GPT-5.4 analysis)", {
    x: 0.5, y: 0.95, w: 9, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.accent,
  });

  // Verification items — Codex/GPT-5.4 reviewed
  const items = [
    { eq: "Mar\u00E9chal Strehl = exp(-\u03C3\u00B2)", status: "CORRECT", statusColor: C.green, desc: "Exact for Gaussian phase stats, not just small-aberration" },
    { eq: "L_abs = 1 - \u03A3|E_pred|\u00B2 / \u03A3|E_vac|\u00B2", status: "CORRECT", statusColor: C.green, desc: "Implicitly couples TP+PIB gradients; ideal for fiber" },
    { eq: "Bucket \u221D TP \u00D7 exp(-\u03C3\u00B2) \u00D7 P_diff", status: "APPROX", statusColor: C.orange, desc: "Assumes separable TP/WFE; breaks in strong scintillation" },
    { eq: "Spatial filtering mechanism", status: "CORRECTED", statusColor: C.orange, desc: "Angular vignetting, NOT evanescent (dx=2\u03BCm\u226B\u03BB/2)" },
    { eq: "PIB \u2194 TP structural tradeoff", status: "CORRECT", statusColor: C.green, desc: "Real for static masks; not information-theoretic limit" },
    { eq: "Unitary CO preservation |\u0394CO|<0.01", status: "CORRECT", statusColor: C.green, desc: "Breaks with finite aperture + multi-layer losses" },
    { eq: "Zernike tip/tilt addition", status: "CORRECT", statusColor: C.green, desc: "Fixed mask adds fixed Zernike; modes affected differently" },
    { eq: "Deep fade floor effect", status: "CORRECT", statusColor: C.green, desc: "Spatial filtering establishes quality floor; expected" },
  ];

  items.forEach((it, i) => {
    const yOff = 1.4 + i * 0.47;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: yOff, w: 9, h: 0.40,
      fill: { color: "0D2137" },
    });
    s.addText(it.eq, {
      x: 0.7, y: yOff, w: 3.8, h: 0.40,
      fontSize: 11, fontFace: "Consolas", color: C.accent, valign: "middle", margin: 0,
    });
    s.addText(it.desc, {
      x: 4.6, y: yOff, w: 3.5, h: 0.40,
      fontSize: 9, fontFace: "Calibri", color: C.gray, valign: "middle", margin: 0,
    });
    s.addText(it.status, {
      x: 8.2, y: yOff, w: 1.3, h: 0.40,
      fontSize: 11, fontFace: "Calibri", color: it.statusColor, valign: "middle", align: "center", margin: 0, bold: true,
    });
  });

  s.addText("Reviewed by Codex (GPT-5.4)  |  6/8 fully correct, 1 approximately correct, 1 terminology corrected", {
    x: 0.5, y: 5.15, w: 9, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.muted, align: "center",
  });
}

// Save
const outPath = path.join(RUNS, "0403-d2nn-loss-sweep-comprehensive.pptx");
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPTX saved:", outPath);
}).catch(err => {
  console.error("Error:", err);
});
