const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "D2NN Lab";
pres.title = "D2NN Loss Function Sweep Report";

// Color palette: Ocean/Deep Blue theme for optics
const C = {
  bg_dark: "0B1D33",
  bg_mid: "132D4F",
  bg_light: "F4F6F9",
  accent: "1B98E0",
  accent2: "E67E22",
  accent3: "27AE60",
  accent4: "9B59B6",
  red: "E74C3C",
  text_light: "FFFFFF",
  text_dark: "1A1A2E",
  text_muted: "7B8794",
  card: "FFFFFF",
  table_header: "1B4F72",
  table_alt: "EBF5FB",
};

const mkShadow = () => ({ type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

const IMG = {
  histogram: "/root/dj/D2NN/kim2026/autoresearch/runs/0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/16_abs_power_histogram_per_strategy.png",
  phase_pib: "/root/dj/D2NN/kim2026/autoresearch/runs/0401-focal-pib-sweep-clean-4loss-cn2-5e14/focal_pib_only/15_phase_masks_5layers.png",
  phase_abs: "/root/dj/D2NN/kim2026/autoresearch/runs/0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/focal_absolute_bucket/15_phase_masks_5layers.png",
};

// ═══════════════════════════════════════════════
// SLIDE 1: Title
// ═══════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.bg_dark };
// Left accent bar
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.625, fill: { color: C.accent } });
s1.addText("D2NN Loss Function Sweep", {
  x: 0.8, y: 1.2, w: 8.5, h: 1.2, fontSize: 40, fontFace: "Arial Black",
  color: C.text_light, bold: true, margin: 0,
});
s1.addText([
  { text: "Static D2NN", options: { color: C.accent, bold: true } },
  { text: "  Practical Limits & Optimal Loss Strategy", options: { color: "CADCFC" } },
], { x: 0.8, y: 2.5, w: 8.5, h: 0.6, fontSize: 20, fontFace: "Arial", margin: 0 });
s1.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 3.3, w: 2.5, h: 0.04, fill: { color: C.accent } });
s1.addText("2026-04-03  |  FSO Beam Cleanup Research", {
  x: 0.8, y: 3.6, w: 8, h: 0.5, fontSize: 14, fontFace: "Arial", color: C.text_muted, margin: 0,
});

// ═══════════════════════════════════════════════
// SLIDE 2: Experiment Overview
// ═══════════════════════════════════════════════
let s2 = pres.addSlide();
s2.background = { color: C.bg_light };
s2.addText("Experiment Overview", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 32, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});
s2.addText("10 loss strategies across 3 sweeps, 2 datasets", {
  x: 0.6, y: 0.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: C.text_muted, margin: 0,
});

const sweepTable = [
  [
    { text: "Sweep", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Data", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Strategies", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Status", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
  ],
  ["0401", "dn100um (Lanczos)", "PIB, Strehl, IO, CO+PIB", "Complete"],
  [
    { text: "0402", options: { fill: { color: C.table_alt } } },
    { text: "pitch_rescale", options: { fill: { color: C.table_alt } } },
    { text: "Abs Bucket (F), TP+PIB (B), Multiplane (C)", options: { fill: { color: C.table_alt } } },
    { text: "Complete", options: { fill: { color: C.table_alt } } },
  ],
  ["0403", "pitch_rescale", "CO+hardTP, AbsBkt+CO, PIB+hardTP", "In progress"],
];
s2.addTable(sweepTable, {
  x: 0.6, y: 1.5, w: 8.8, colW: [1.0, 2.0, 4.0, 1.8],
  fontSize: 12, fontFace: "Arial", border: { pt: 0.5, color: "D5D8DC" },
  rowH: [0.4, 0.4, 0.4, 0.4],
});

// Key numbers
const nums = [
  { val: "10", label: "Loss Strategies", color: C.accent },
  { val: "5000", label: "Training Pairs", color: C.accent2 },
  { val: "30", label: "Epochs Each", color: C.accent3 },
  { val: "~6 min", label: "Per Epoch (A100)", color: C.accent4 },
];
nums.forEach((n, i) => {
  const cx = 0.6 + i * 2.3;
  s2.addShape(pres.shapes.RECTANGLE, { x: cx, y: 3.6, w: 2.0, h: 1.6, fill: { color: C.card }, shadow: mkShadow() });
  s2.addText(n.val, { x: cx, y: 3.7, w: 2.0, h: 0.8, fontSize: 32, fontFace: "Arial Black", color: n.color, align: "center", valign: "middle", margin: 0 });
  s2.addText(n.label, { x: cx, y: 4.5, w: 2.0, h: 0.5, fontSize: 11, fontFace: "Arial", color: C.text_muted, align: "center", valign: "top", margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 3: Normalized PIB Lies
// ═══════════════════════════════════════════════
let s3 = pres.addSlide();
s3.background = { color: C.bg_light };
s3.addText("Normalized PIB Lies", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 32, fontFace: "Arial Black", color: C.red, margin: 0,
});
s3.addText("High PIB does NOT mean high received power", {
  x: 0.6, y: 0.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: C.text_muted, margin: 0,
});

const pibTable = [
  [
    { text: "Strategy", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "PIB@10\u03BCm", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Throughput", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Abs Power vs Turb", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Verdict", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
  ],
  [
    "PIB Only (0401)", { text: "90.1%", options: { bold: true, color: C.accent3 } }, { text: "50.7%", options: { color: C.red } }, { text: "58.7%", options: { color: C.red, bold: true } }, { text: "LOSS", options: { bold: true, color: C.red } },
  ],
  [
    { text: "Abs Bucket F (0402)", options: { fill: { color: C.table_alt } } },
    { text: "81.4%", options: { fill: { color: C.table_alt } } },
    { text: "98.6%", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
    { text: "105.8%", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
    { text: "GAIN", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
  ],
  [
    "TP+PIB B (0402)", "81.6%", { text: "98.4%", options: { bold: true, color: C.accent3 } }, { text: "105.8%", options: { bold: true, color: C.accent3 } }, { text: "GAIN", options: { bold: true, color: C.accent3 } },
  ],
  [
    { text: "Multiplane C (0402)", options: { fill: { color: C.table_alt } } },
    { text: "84.2%", options: { fill: { color: C.table_alt } } },
    { text: "39.5%", options: { color: C.red, fill: { color: C.table_alt } } },
    { text: "44.2%", options: { color: C.red, bold: true, fill: { color: C.table_alt } } },
    { text: "LOSS", options: { bold: true, color: C.red, fill: { color: C.table_alt } } },
  ],
  [
    "CO+hardTP (0403)", "74.0%", { text: "99.3%", options: { bold: true, color: C.accent3 } }, { text: "97.0%", options: { color: C.accent2 } }, { text: "~NEUTRAL", options: { color: C.accent2 } },
  ],
];
s3.addTable(pibTable, {
  x: 0.4, y: 1.5, w: 9.2, colW: [2.2, 1.3, 1.5, 2.2, 2.0],
  fontSize: 11, fontFace: "Arial", border: { pt: 0.5, color: "D5D8DC" },
  rowH: [0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
});

s3.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 4.1, w: 8.8, h: 1.2, fill: { color: "FFF3E0" }, shadow: mkShadow() });
s3.addText([
  { text: "Key Insight: ", options: { bold: true, color: C.accent2 } },
  { text: "PIB Only achieved 90% PIB but absolute received power was only 58.7% of turbulent baseline. ", options: { color: C.text_dark } },
  { text: "Throughput is the real metric.", options: { bold: true, color: C.red } },
], { x: 0.9, y: 4.2, w: 8.2, h: 1.0, fontSize: 13, fontFace: "Arial", margin: 0 });

// ═══════════════════════════════════════════════
// SLIDE 4: Why Throughput Drops
// ═══════════════════════════════════════════════
let s4 = pres.addSlide();
s4.background = { color: C.bg_dark };
s4.addText("Why Does Throughput Drop?", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 32, fontFace: "Arial Black", color: C.text_light, margin: 0,
});

const steps = [
  { num: "1", title: "Phase Mask Pattern", desc: "D2NN learns high-frequency phase gratings to redirect energy" },
  { num: "2", title: "Evanescent Waves", desc: "High spatial frequencies exceed propagation cutoff (k\u00B2x + k\u00B2y > k\u00B2)" },
  { num: "3", title: "Angular Spectrum Filter", desc: "Band-limited propagation zeros out evanescent components" },
  { num: "4", title: "Energy Lost", desc: "Turbulent beams lose MORE energy (50%) than vacuum (20-30%)" },
];
steps.forEach((st, i) => {
  const cy = 1.3 + i * 1.0;
  s4.addShape(pres.shapes.OVAL, { x: 0.7, y: cy, w: 0.55, h: 0.55, fill: { color: C.accent } });
  s4.addText(st.num, { x: 0.7, y: cy, w: 0.55, h: 0.55, fontSize: 20, fontFace: "Arial Black", color: C.text_light, align: "center", valign: "middle", margin: 0 });
  s4.addText(st.title, { x: 1.5, y: cy - 0.05, w: 7.5, h: 0.3, fontSize: 16, fontFace: "Arial", color: C.text_light, bold: true, margin: 0 });
  s4.addText(st.desc, { x: 1.5, y: cy + 0.25, w: 7.5, h: 0.3, fontSize: 12, fontFace: "Arial", color: "AABBCC", margin: 0 });
});

s4.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 5.0, w: 4.2, h: 0.45, fill: { color: C.accent }, shadow: mkShadow() });
s4.addText("D2NN acts as a spatial filter \u2014 scatters unwanted energy away", {
  x: 0.8, y: 5.0, w: 4.0, h: 0.45, fontSize: 11, fontFace: "Arial", color: C.text_light, bold: true, valign: "middle", margin: 0,
});

// ═══════════════════════════════════════════════
// SLIDE 5: Absolute Power Histogram
// ═══════════════════════════════════════════════
let s5 = pres.addSlide();
s5.background = { color: C.bg_light };
s5.addText("Absolute Received Power (0402)", {
  x: 0.6, y: 0.2, w: 9, h: 0.6, fontSize: 30, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});
s5.addText("Turbulent (gray) vs D2NN (color) \u2014 500 test samples, bucket @10\u03BCm", {
  x: 0.6, y: 0.75, w: 9, h: 0.35, fontSize: 13, fontFace: "Arial", color: C.text_muted, margin: 0,
});
s5.addImage({ path: IMG.histogram, x: 0.3, y: 1.2, w: 9.4, h: 2.8 });

// Bottom callouts
const callouts = [
  { label: "Abs Bucket (F)", val: "+5.8%", color: C.accent2 },
  { label: "TP+PIB (B)", val: "+5.8%", color: C.accent3 },
  { label: "Multiplane (C)", val: "-55.8%", color: C.accent4 },
];
callouts.forEach((c, i) => {
  const cx = 0.6 + i * 3.2;
  s5.addShape(pres.shapes.RECTANGLE, { x: cx, y: 4.2, w: 2.8, h: 1.1, fill: { color: C.card }, shadow: mkShadow() });
  s5.addShape(pres.shapes.RECTANGLE, { x: cx, y: 4.2, w: 0.08, h: 1.1, fill: { color: c.color } });
  s5.addText(c.label, { x: cx + 0.2, y: 4.25, w: 2.4, h: 0.4, fontSize: 12, fontFace: "Arial", color: C.text_muted, margin: 0 });
  s5.addText(c.val + " vs turbulent", { x: cx + 0.2, y: 4.6, w: 2.4, h: 0.5, fontSize: 20, fontFace: "Arial Black", color: c.color, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 6: Throughput vs PIB Trade-off
// ═══════════════════════════════════════════════
let s6 = pres.addSlide();
s6.background = { color: C.bg_light };
s6.addText("Throughput vs PIB: Structural Trade-off", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 30, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});

// PIB vs TP comparison table (replaces scatter chart which doesn't render in LibreOffice)
const tradeoffTable = [
  [
    { text: "Strategy", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "PIB@10\u03BCm", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Throughput", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Abs Power", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
  ],
  [
    "PIB Only", { text: "90.1%", options: { bold: true, color: C.accent3 } }, { text: "50.7%", options: { color: C.red, bold: true } }, { text: "58.7%", options: { color: C.red } },
  ],
  [
    { text: "Abs Bucket (F)", options: { fill: { color: C.table_alt } } },
    { text: "81.4%", options: { fill: { color: C.table_alt } } },
    { text: "98.6%", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
    { text: "105.8%", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
  ],
  [
    "TP+PIB (B)", "81.6%", { text: "98.4%", options: { bold: true, color: C.accent3 } }, { text: "105.8%", options: { bold: true, color: C.accent3 } },
  ],
  [
    { text: "Multiplane (C)", options: { fill: { color: C.table_alt } } },
    { text: "84.2%", options: { fill: { color: C.table_alt } } },
    { text: "39.5%", options: { color: C.red, bold: true, fill: { color: C.table_alt } } },
    { text: "44.2%", options: { color: C.red, fill: { color: C.table_alt } } },
  ],
  [
    "CO+hardTP", "74.0%", { text: "99.3%", options: { bold: true, color: C.accent3 } }, { text: "97.0%", options: { color: C.accent2 } },
  ],
];
s6.addText("PIB@10\u03BCm vs Throughput", {
  x: 0.5, y: 1.0, w: 5, h: 0.4, fontSize: 16, fontFace: "Arial", color: C.text_dark, bold: true, margin: 0,
});
s6.addTable(tradeoffTable, {
  x: 0.5, y: 1.5, w: 5, colW: [1.6, 1.1, 1.2, 1.1],
  fontSize: 12, fontFace: "Arial", border: { pt: 0.5, color: "D5D8DC" },
  rowH: [0.38, 0.38, 0.38, 0.38, 0.38, 0.38],
});

// Right side: key insight cards
const insights = [
  { title: "Enforce TP", desc: "Scatter blocked \u2192 PIB limited to ~81%\n(+5.8% absolute power)", color: C.accent3 },
  { title: "Release TP", desc: "PIB reaches 90% but absolute\npower drops to 59% of baseline", color: C.red },
  { title: "Fundamental Limit", desc: "Static phase masks cannot correct\nrandom turbulence without energy loss", color: C.accent },
];
insights.forEach((ins, i) => {
  const cy = 1.3 + i * 1.35;
  s6.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: cy, w: 3.8, h: 1.15, fill: { color: C.card }, shadow: mkShadow() });
  s6.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: cy, w: 0.08, h: 1.15, fill: { color: ins.color } });
  s6.addText(ins.title, { x: 6.1, y: cy + 0.05, w: 3.3, h: 0.35, fontSize: 14, fontFace: "Arial", bold: true, color: ins.color, margin: 0 });
  s6.addText(ins.desc, { x: 6.1, y: cy + 0.4, w: 3.3, h: 0.65, fontSize: 11, fontFace: "Arial", color: C.text_muted, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 7: Phase Mask Comparison
// ═══════════════════════════════════════════════
let s7 = pres.addSlide();
s7.background = { color: C.bg_light };
s7.addText("Phase Mask Patterns", {
  x: 0.6, y: 0.2, w: 9, h: 0.6, fontSize: 30, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});

s7.addText("PIB Only (0401) \u2014 High-frequency grating, aggressive scatter", {
  x: 0.6, y: 0.85, w: 9, h: 0.3, fontSize: 12, fontFace: "Arial", color: C.red, bold: true, margin: 0,
});
s7.addImage({ path: IMG.phase_pib, x: 0.3, y: 1.2, w: 9.4, h: 1.8 });

s7.addText("Abs Bucket (0402) \u2014 Nearly flat, throughput preserved", {
  x: 0.6, y: 3.15, w: 9, h: 0.3, fontSize: 12, fontFace: "Arial", color: C.accent3, bold: true, margin: 0,
});
s7.addImage({ path: IMG.phase_abs, x: 0.3, y: 3.5, w: 9.4, h: 1.8 });

// ═══════════════════════════════════════════════
// SLIDE 8: Zernike Analysis
// ═══════════════════════════════════════════════
let s8 = pres.addSlide();
s8.background = { color: C.bg_light };
s8.addText("Zernike Mode Analysis", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 30, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});
s8.addText("Can D2NN correct wavefront aberrations while maintaining throughput?", {
  x: 0.6, y: 0.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: C.text_muted, margin: 0,
});

const zernikeTable = [
  [
    { text: "Strategy", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Higher-order \u0394", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Tip/Tilt \u0394", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Throughput", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
    { text: "Assessment", options: { bold: true, color: "FFFFFF", fill: { color: C.table_header } } },
  ],
  ["F/B (Abs Bucket)", { text: "\u00B11 nm", options: { color: C.text_muted } }, "\u00B10.2 nm", { text: "98%", options: { bold: true, color: C.accent3 } }, "No WF correction"],
  [
    { text: "Multiplane C", options: { fill: { color: C.table_alt } } },
    { text: "-28 nm", options: { bold: true, color: C.accent3, fill: { color: C.table_alt } } },
    { text: "+1.3 nm", options: { fill: { color: C.table_alt } } },
    { text: "40%", options: { color: C.red, fill: { color: C.table_alt } } },
    { text: "Good WF, bad TP", options: { fill: { color: C.table_alt } } },
  ],
  ["PIB Only", "-28 nm", { text: "+16 nm", options: { color: C.red } }, { text: "51%", options: { color: C.red } }, "WF + scatter"],
];
s8.addTable(zernikeTable, {
  x: 0.5, y: 1.5, w: 9.0, colW: [2.0, 1.8, 1.5, 1.5, 2.2],
  fontSize: 12, fontFace: "Arial", border: { pt: 0.5, color: "D5D8DC" },
  rowH: [0.4, 0.4, 0.4, 0.4],
});

s8.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 3.5, w: 8.8, h: 1.8, fill: { color: "FFF3E0" }, shadow: mkShadow() });
s8.addText([
  { text: "Dilemma: ", options: { bold: true, color: C.accent2, fontSize: 16 } },
  { text: "\n\n", options: { breakLine: true, fontSize: 6 } },
  { text: "Wavefront correction requires the D2NN to create complex phase patterns.\n", options: { breakLine: true, color: C.text_dark, fontSize: 13 } },
  { text: "These patterns generate high spatial frequencies that get clipped by propagation.\n", options: { breakLine: true, color: C.text_dark, fontSize: 13 } },
  { text: "Result: correcting wavefront = losing energy. A structural limit of static optics.", options: { bold: true, color: C.red, fontSize: 13 } },
], { x: 0.9, y: 3.6, w: 8.2, h: 1.6, fontFace: "Arial", margin: 0 });

// ═══════════════════════════════════════════════
// SLIDE 9: CO Loss Alone Fails
// ═══════════════════════════════════════════════
let s9 = pres.addSlide();
s9.background = { color: C.bg_dark };
s9.addText("CO Loss Alone Fails (0403)", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 30, fontFace: "Arial Black", color: C.text_light, margin: 0,
});
s9.addText("co_hard_tp: CO + Quadratic TP penalty (\u03BB=10)", {
  x: 0.6, y: 0.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Arial", color: "AABBCC", margin: 0,
});

// Big stat cards
const coStats = [
  { val: "99.3%", label: "Throughput", sub: "Best of all strategies", color: C.accent3 },
  { val: "74.0%", label: "PIB@10\u03BCm", sub: "Below baseline (76.1%)", color: C.red },
  { val: "97.0%", label: "Abs Power vs Turb", sub: "3% loss in received power", color: C.accent2 },
];
coStats.forEach((st, i) => {
  const cx = 0.6 + i * 3.1;
  s9.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.6, w: 2.8, h: 2.0, fill: { color: C.bg_mid }, shadow: mkShadow() });
  s9.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.6, w: 2.8, h: 0.06, fill: { color: st.color } });
  s9.addText(st.val, { x: cx, y: 1.8, w: 2.8, h: 0.9, fontSize: 36, fontFace: "Arial Black", color: st.color, align: "center", valign: "middle", margin: 0 });
  s9.addText(st.label, { x: cx, y: 2.7, w: 2.8, h: 0.4, fontSize: 14, fontFace: "Arial", color: C.text_light, align: "center", margin: 0 });
  s9.addText(st.sub, { x: cx, y: 3.05, w: 2.8, h: 0.4, fontSize: 11, fontFace: "Arial", color: C.text_muted, align: "center", margin: 0 });
});

s9.addText([
  { text: "Why: ", options: { bold: true, color: C.accent } },
  { text: "CO optimizes full-field matching (amplitude + phase), not bucket concentration. ", options: { color: "CADCFC" } },
  { text: "No gradient signal to focus energy into the 10\u03BCm bucket.", options: { color: C.text_light, bold: true } },
], { x: 0.6, y: 4.0, w: 8.8, h: 0.8, fontSize: 13, fontFace: "Arial", margin: 0 });

s9.addText("Remaining hope: pib_hard_tp combines PIB concentration + TP enforcement", {
  x: 0.6, y: 4.9, w: 8.8, h: 0.4, fontSize: 13, fontFace: "Arial", color: C.accent3, italic: true, margin: 0,
});

// ═══════════════════════════════════════════════
// SLIDE 10: Conclusion
// ═══════════════════════════════════════════════
let s10 = pres.addSlide();
s10.background = { color: C.bg_light };
s10.addText("Conclusions", {
  x: 0.6, y: 0.3, w: 9, h: 0.7, fontSize: 32, fontFace: "Arial Black", color: C.text_dark, margin: 0,
});

const conclusions = [
  { icon: "1", title: "Throughput is the real metric", desc: "Normalized PIB can be misleading. Only strategies with TP > 95% improved absolute received power.", color: C.accent },
  { icon: "2", title: "Static D2NN limit: +5.8%", desc: "Best absolute power improvement over turbulent baseline (Abs Bucket / TP+PIB strategies).", color: C.accent2 },
  { icon: "3", title: "PIB + TP are structurally opposed", desc: "Concentrating energy requires scattering \u2192 throughput loss. Static masks can't escape this trade-off.", color: C.red },
  { icon: "4", title: "Vacuum PIB (94%) is unreachable", desc: "With throughput maintained, maximum PIB is ~82%. Closing the gap requires fundamentally different approaches.", color: C.accent4 },
];
conclusions.forEach((c, i) => {
  const cy = 1.2 + i * 1.05;
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: cy, w: 8.8, h: 0.9, fill: { color: C.card }, shadow: mkShadow() });
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: cy, w: 0.08, h: 0.9, fill: { color: c.color } });
  s10.addShape(pres.shapes.OVAL, { x: 0.9, y: cy + 0.15, w: 0.55, h: 0.55, fill: { color: c.color } });
  s10.addText(c.icon, { x: 0.9, y: cy + 0.15, w: 0.55, h: 0.55, fontSize: 18, fontFace: "Arial Black", color: C.text_light, align: "center", valign: "middle", margin: 0 });
  s10.addText(c.title, { x: 1.7, y: cy + 0.05, w: 7.5, h: 0.35, fontSize: 15, fontFace: "Arial", bold: true, color: c.color, margin: 0 });
  s10.addText(c.desc, { x: 1.7, y: cy + 0.42, w: 7.5, h: 0.4, fontSize: 12, fontFace: "Arial", color: C.text_muted, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 11: Next Steps
// ═══════════════════════════════════════════════
let s11 = pres.addSlide();
s11.background = { color: C.bg_dark };
s11.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.15, h: 5.625, fill: { color: C.accent } });
s11.addText("Next Steps", {
  x: 0.8, y: 0.3, w: 9, h: 0.7, fontSize: 32, fontFace: "Arial Black", color: C.text_light, margin: 0,
});

// Left column: pending experiments
s11.addText("Pending (0403)", {
  x: 0.8, y: 1.2, w: 4, h: 0.4, fontSize: 16, fontFace: "Arial", color: C.accent, bold: true, margin: 0,
});
s11.addText([
  { text: "abs_bucket_plus_co", options: { bold: true, color: C.text_light, breakLine: true } },
  { text: "  Focal abs power + CO wavefront matching\n", options: { color: "AABBCC", breakLine: true } },
  { text: "\n", options: { fontSize: 4, breakLine: true } },
  { text: "pib_hard_tp", options: { bold: true, color: C.text_light, breakLine: true } },
  { text: "  PIB maximization + quadratic TP (\u03BB=10)\n", options: { color: "AABBCC", breakLine: true } },
  { text: "  Most promising: PIB Only's 90% + TP enforcement", options: { color: C.accent3 } },
], { x: 0.8, y: 1.7, w: 4.2, h: 2.5, fontSize: 13, fontFace: "Arial", margin: 0 });

// Right column: future directions
s11.addText("Future Directions", {
  x: 5.5, y: 1.2, w: 4.2, h: 0.4, fontSize: 16, fontFace: "Arial", color: C.accent2, bold: true, margin: 0,
});
const futures = [
  { title: "Dynamic D2NN", desc: "Sample-adaptive phase masks" },
  { title: "Hybrid AO + D2NN", desc: "AO for low-order, D2NN for high-order" },
  { title: "More Layers / Wider Aperture", desc: "Increase degrees of freedom" },
  { title: "TV Weight Sweep", desc: "Optimize smoothness vs correction" },
];
futures.forEach((f, i) => {
  const cy = 1.75 + i * 0.7;
  s11.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: cy, w: 4.0, h: 0.55, fill: { color: C.bg_mid } });
  s11.addText(f.title, { x: 5.7, y: cy + 0.02, w: 3.6, h: 0.25, fontSize: 13, fontFace: "Arial", bold: true, color: C.text_light, margin: 0 });
  s11.addText(f.desc, { x: 5.7, y: cy + 0.27, w: 3.6, h: 0.25, fontSize: 11, fontFace: "Arial", color: C.text_muted, margin: 0 });
});

// Bottom tagline
s11.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 4.8, w: 8.5, h: 0.04, fill: { color: C.accent } });
s11.addText("Breaking the static limit requires moving beyond fixed phase masks", {
  x: 0.8, y: 4.95, w: 8.5, h: 0.4, fontSize: 14, fontFace: "Arial", color: "CADCFC", italic: true, margin: 0,
});

// Save
const outPath = "/root/dj/D2NN/kim2026/autoresearch/runs/0403-d2nn-loss-sweep-report.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("Saved: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
