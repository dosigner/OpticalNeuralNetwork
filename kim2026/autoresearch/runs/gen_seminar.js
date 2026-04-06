const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "D2NN Lab";
pres.title = "Static D2NN Seminar";

// Deep teal/charcoal palette — optics/scientific feel
const C = {
  bg_dark: "0D1B2A",
  bg_mid: "1B2A4A",
  bg_light: "F0F4F8",
  accent: "00B4D8",    // teal
  accent2: "E07A2F",   // warm orange for contrast
  accent3: "48BB78",   // green for positive
  red: "E53E3E",
  text_w: "FFFFFF",
  text_d: "1A202C",
  text_m: "718096",
  card: "FFFFFF",
  th: "1B4F72",
  ta: "EBF5FB",
};

const IMG = {
  ee: "/root/dj/D2NN/kim2026/autoresearch/runs/0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/20_encircled_energy_per_strategy.png",
  hist: "/root/dj/D2NN/kim2026/autoresearch/runs/0402-focal-new-losses-pitchrescale-3strat-cn2-5e14/16_abs_power_histogram_per_strategy.png",
  dist6: "/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm/21_distance_sweep_summary_6panel.png",
  pm100: "/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm/L100m/focal_raw_received_power/15_phase_masks_5layers.png",
  pm3k: "/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm/L3000m/focal_raw_received_power/15_phase_masks_5layers.png",
  cross: "/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm/22_cross_distance_3km_model/focal_L1000m_3km_vs_local.png",
};

// Question badge helper
function addQ(slide, num) {
  slide.addShape(pres.shapes.OVAL, { x: 0.4, y: 0.3, w: 0.6, h: 0.6, fill: { color: C.accent } });
  slide.addText("Q" + num, { x: 0.4, y: 0.3, w: 0.6, h: 0.6, fontSize: 20, fontFace: "Arial Black", color: C.text_w, align: "center", valign: "middle", margin: 0 });
}

// ═══════════════════════════════════════════════════════════
// SLIDE 1: Title
// ═══════════════════════════════════════════════════════════
let s1 = pres.addSlide();
s1.background = { color: C.bg_dark };
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
s1.addText("Static D2NN\uC740 \uB300\uAE30 \uB09C\uB958\uB97C\n\uBCF4\uC815\uD560 \uC218 \uC788\uB294\uAC00?", {
  x: 0.8, y: 1.0, w: 8.5, h: 2.2, fontSize: 42, fontFace: "Arial Black",
  color: C.text_w, bold: true, margin: 0, lineSpacingMultiple: 1.2,
});
s1.addText("Loss Function Sweep & Distance Analysis", {
  x: 0.8, y: 3.3, w: 8.5, h: 0.5, fontSize: 18, fontFace: "Calibri",
  color: C.accent, margin: 0,
});
s1.addText("2026-04-06  |  D2NN Lab", {
  x: 0.8, y: 4.1, w: 4, h: 0.4, fontSize: 14, fontFace: "Calibri",
  color: C.text_m, margin: 0,
});
// Bottom specs bar
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 4.8, w: 10, h: 0.82, fill: { color: C.bg_mid } });
const specs = ["Cn\u00B2=5\u00D710\u207B\u00B9\u2074", "\u03BB=1.55\u03BCm", "5-layer D2NN", "f=6.5mm (SMF)", "N=1024"];
specs.forEach((s, i) => {
  pres.addSlide; // no-op
  s1.addText(s, { x: 0.3 + i * 1.95, y: 4.9, w: 1.8, h: 0.55, fontSize: 12, fontFace: "Calibri",
    color: C.accent, align: "center", valign: "middle", margin: 0 });
});


// ═══════════════════════════════════════════════════════════
// SLIDE 2: Q1 — Can D2NN correct turbulence?
// ═══════════════════════════════════════════════════════════
let s2 = pres.addSlide();
s2.background = { color: C.bg_light };
addQ(s2, 1);
s2.addText("D2NN\uC774 \uB300\uAE30 \uB09C\uB958\uB97C \uBCF4\uC815\uD560 \uC218 \uC788\uB294\uAC00?", {
  x: 1.2, y: 0.3, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_d, margin: 0,
});

// Left: setup
s2.addText("Simulation Setup", { x: 0.5, y: 1.2, w: 4.5, h: 0.4, fontSize: 16, fontFace: "Arial", bold: true, color: C.th, margin: 0 });
s2.addText([
  { text: "5 phase-only layers, 10mm spacing\n", options: { breakLine: true } },
  { text: "Receiver: 2mm aperture, f=6.5mm lens\n", options: { breakLine: true } },
  { text: "SMF-28 bucket: 10\u03BCm radius\n", options: { breakLine: true } },
  { text: "Data: pitch_rescale (2000 pairs/distance)", options: {} },
], { x: 0.5, y: 1.7, w: 4.5, h: 1.8, fontSize: 13, fontFace: "Calibri", color: C.text_d, margin: 0 });

// Right: key numbers
const nums = [
  { val: "443 nm", label: "WFE RMS", sub: ">> Mar\u00E9chal 111nm" },
  { val: "0.04", label: "Strehl", sub: "exp(-\u03C3\u00B2)" },
  { val: "-1.04 dB", label: "Vac\u2013Turb gap", sub: "@10\u03BCm bucket" },
];
nums.forEach((n, i) => {
  const cy = 1.2 + i * 1.2;
  s2.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: cy, w: 4.0, h: 1.0, fill: { color: C.card },
    shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.1 } });
  s2.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: cy, w: 0.07, h: 1.0, fill: { color: C.accent } });
  s2.addText(n.val, { x: 5.8, y: cy + 0.05, w: 2.0, h: 0.55, fontSize: 26, fontFace: "Arial Black", color: C.accent, margin: 0 });
  s2.addText(n.label, { x: 7.8, y: cy + 0.05, w: 1.5, h: 0.3, fontSize: 13, fontFace: "Calibri", bold: true, color: C.text_d, margin: 0 });
  s2.addText(n.sub, { x: 7.8, y: cy + 0.4, w: 1.5, h: 0.3, fontSize: 11, fontFace: "Calibri", color: C.text_m, margin: 0 });
});

// Bottom equation
s2.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.6, w: 9.0, h: 0.7, fill: { color: C.bg_mid } });
s2.addText("Bucket Power  \u2248  TP  \u00D7  exp(\u2013\u03C3\u00B2_WFE)  \u00D7  P_diffraction       \u2192  Static mask: same correction for ALL samples", {
  x: 0.7, y: 4.65, w: 8.6, h: 0.6, fontSize: 13, fontFace: "Consolas", color: C.accent, valign: "middle", margin: 0,
});


// ═══════════════════════════════════════════════════════════
// SLIDE 3: Q2 — PIB 90% but power drops?
// ═══════════════════════════════════════════════════════════
let s3 = pres.addSlide();
s3.background = { color: C.bg_light };
addQ(s3, 2);
s3.addText("PIB 90%\uC778\uB370 \uC65C \uC218\uC2E0\uC804\uB825\uC740 \uB5A8\uC5B4\uC9C0\uB294\uAC00?", {
  x: 1.2, y: 0.3, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_d, margin: 0,
});

// Comparison table
const t3 = [
  [
    { text: "Strategy", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "PIB@10\u03BCm", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "Throughput", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "Abs Power", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
  ],
  [
    "PIB Only",
    { text: "90.1%", options: { bold: true, color: C.accent3 } },
    { text: "50.7%", options: { bold: true, color: C.red } },
    { text: "59% of turb", options: { color: C.red } },
  ],
  [
    { text: "Abs Bucket", options: { fill: { color: C.ta } } },
    { text: "81.4%", options: { fill: { color: C.ta } } },
    { text: "98.6%", options: { bold: true, color: C.accent3, fill: { color: C.ta } } },
    { text: "+5.8%", options: { bold: true, color: C.accent3, fill: { color: C.ta } } },
  ],
];
s3.addTable(t3, { x: 0.5, y: 1.1, w: 4.5, colW: [1.3, 1.0, 1.1, 1.1], fontSize: 12, fontFace: "Calibri",
  border: { pt: 0.5, color: "D5D8DC" }, rowH: [0.35, 0.35, 0.35] });

// Insight box
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.4, w: 4.5, h: 1.0, fill: { color: "FFF8E1" },
  shadow: { type: "outer", blur: 3, offset: 1, angle: 135, color: "000000", opacity: 0.08 } });
s3.addText([
  { text: "Spatial Filter \uD6A8\uACFC: ", options: { bold: true, color: C.accent2 } },
  { text: "D2NN\uC774 \uC5D0\uB108\uC9C0\uB97C scatter\uD574\uC11C PIB \uC0C1\uC2B9, \uD558\uC9C0\uB9CC \uCD1D \uC5D0\uB108\uC9C0(TP) \uAC10\uC18C. Normalized PIB \u2260 \uC218\uC2E0\uC804\uB825.", options: { color: C.text_d } },
], { x: 0.7, y: 2.5, w: 4.1, h: 0.8, fontSize: 12, fontFace: "Calibri", margin: 0 });

// Right: encircled energy image
s3.addImage({ path: IMG.ee, x: 5.2, y: 1.0, w: 4.5, h: 2.8 });
s3.addText("Encircled Energy vs Radius", { x: 5.2, y: 3.9, w: 4.5, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.text_m, align: "center", margin: 0 });

// Bottom stat
s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.5, w: 9.0, h: 0.8, fill: { color: C.bg_dark } });
s3.addText([
  { text: "Peak irradiance 25\u00D7 \uCC28\uC774 \u2260 \uC218\uC2E0\uC804\uB825 25\u00D7 \uCC28\uC774.  ", options: { color: C.accent } },
  { text: "\uC218\uC2E0\uAE30\uB294 \uBA74\uC801\uC5D0 \uAC78\uCCD0 \uC801\uBD84 \u2192 1.27\u00D7 \uCC28\uC774\uB9CC \uBC1C\uC0DD.", options: { color: C.text_w } },
], { x: 0.7, y: 4.55, w: 8.6, h: 0.7, fontSize: 13, fontFace: "Calibri", valign: "middle", margin: 0 });


// ═══════════════════════════════════════════════════════════
// SLIDE 4: Q3 — Optimal loss function?
// ═══════════════════════════════════════════════════════════
let s4 = pres.addSlide();
s4.background = { color: C.bg_light };
addQ(s4, 3);
s4.addText("\uCD5C\uC801\uC758 \uC190\uC2E4\uD568\uC218\uB294 \uBB34\uC5C7\uC778\uAC00?", {
  x: 1.2, y: 0.3, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_d, margin: 0,
});

// Image: histogram
s4.addImage({ path: IMG.hist, x: 0.3, y: 1.0, w: 9.4, h: 2.5 });

// Bottom: winner callout
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.7, w: 9.0, h: 1.5, fill: { color: C.card },
  shadow: { type: "outer", blur: 4, offset: 2, angle: 135, color: "000000", opacity: 0.1 } });
s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.7, w: 0.08, h: 1.5, fill: { color: C.accent3 } });
s4.addText([
  { text: "Winner: Raw RP Loss", options: { bold: true, color: C.accent3, fontSize: 18, breakLine: true } },
  { text: "\n", options: { fontSize: 6, breakLine: true } },
  { text: "\u2112 = \u2013log(bucket_energy / input_energy)\n", options: { fontFace: "Consolas", fontSize: 14, color: C.text_d, breakLine: true } },
  { text: "\n", options: { fontSize: 4, breakLine: true } },
  { text: "Vacuum reference \uBD88\uD544\uC694. \uC808\uB300 \uC218\uC2E0\uC804\uB825 \uC9C1\uC811 \uCD5C\uB300\uD654. +6.4% vs turbulent.", options: { fontSize: 13, color: C.text_m } },
], { x: 0.8, y: 3.8, w: 8.5, h: 1.3, fontFace: "Calibri", margin: 0 });


// ═══════════════════════════════════════════════════════════
// SLIDE 5: Q4 — Distance effect?
// ═══════════════════════════════════════════════════════════
let s5 = pres.addSlide();
s5.background = { color: C.bg_light };
addQ(s5, 4);
s5.addText("\uAC70\uB9AC\uC5D0 \uB530\uB77C D2NN \uD6A8\uACFC\uB294 \uC5B4\uB5BB\uAC8C \uBCC0\uD558\uB294\uAC00?", {
  x: 1.2, y: 0.3, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_d, margin: 0,
});

s5.addImage({ path: IMG.dist6, x: 0.2, y: 1.0, w: 9.6, h: 4.3 });


// ═══════════════════════════════════════════════════════════
// SLIDE 6: Q5 — What is D2NN actually doing?
// ═══════════════════════════════════════════════════════════
let s6 = pres.addSlide();
s6.background = { color: C.bg_dark };
addQ(s6, 5);
s6.addText("D2NN\uC774 \uC2E4\uC81C\uB85C \uBB34\uC5C7\uC744 \uD558\uACE0 \uC788\uB294\uAC00?", {
  x: 1.2, y: 0.25, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_w, margin: 0,
});

// Effect decomposition table
const t6 = [
  [
    { text: "L", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "Beam Shaping", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "Turb Correction", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "Total", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
  ],
  ["100m", { text: "+84.7%", options: { bold: true, color: C.accent2 } }, "+5.9%", "+90.6%"],
  [
    { text: "1km", options: { fill: { color: "1E3A5F" } } },
    { text: "+4.5%", options: { fill: { color: "1E3A5F" } } },
    { text: "+4.5%", options: { fill: { color: "1E3A5F" } } },
    { text: "+9.0%", options: { fill: { color: "1E3A5F" } } },
  ],
  ["3km", "+93.2%", { text: "+129.8%", options: { bold: true, color: C.accent3 } },
   { text: "+223%", options: { bold: true, color: C.accent3 } }],
];
s6.addTable(t6, { x: 0.5, y: 1.1, w: 4.2, colW: [0.7, 1.2, 1.2, 1.1], fontSize: 12, fontFace: "Calibri",
  color: "CADCFC", border: { pt: 0.5, color: "2D4A6F" }, rowH: [0.35, 0.35, 0.35, 0.35] });

s6.addText([
  { text: "Beam Shaping: ", options: { bold: true, color: C.accent } },
  { text: "Airy sidelobe \u2192 central peak redistribution (vacuum\uC5D0\uB3C4 \uC791\uB3D9)\n", options: { color: "CADCFC", breakLine: true } },
  { text: "Turb Correction: ", options: { bold: true, color: C.accent3 } },
  { text: "\uAC15\uD55C \uB09C\uB958\uC5D0\uC11C \uD1B5\uACC4\uC801 \uBAA8\uB4DC \uBCF4\uC815 (\uACE0\uCC28 Zernike \u039434~94nm)", options: { color: "CADCFC" } },
], { x: 0.5, y: 2.7, w: 4.2, h: 1.0, fontSize: 12, fontFace: "Calibri", margin: 0 });

// Right: phase mask comparison
s6.addImage({ path: IMG.pm100, x: 5.0, y: 1.0, w: 4.7, h: 1.4 });
s6.addText("L=100m: Concentric rings (beam shaper)", { x: 5.0, y: 2.45, w: 4.7, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.text_m, align: "center", margin: 0 });

s6.addImage({ path: IMG.pm3k, x: 5.0, y: 2.9, w: 4.7, h: 1.4 });
s6.addText("L=3km: Complex pattern (shaping + correction)", { x: 5.0, y: 4.35, w: 4.7, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.text_m, align: "center", margin: 0 });

// Bottom insight
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.8, w: 9.0, h: 0.6, fill: { color: C.accent } });
s6.addText("D2NN = Diffractive Beam Shaper + Statistical Turbulence Corrector", {
  x: 0.7, y: 4.8, w: 8.6, h: 0.6, fontSize: 15, fontFace: "Arial Black", color: C.text_w, valign: "middle", margin: 0,
});


// ═══════════════════════════════════════════════════════════
// SLIDE 7: Q6 — Generalization?
// ═══════════════════════════════════════════════════════════
let s7 = pres.addSlide();
s7.background = { color: C.bg_light };
addQ(s7, 6);
s7.addText("\uB2E4\uB978 \uC870\uAC74\uC5D0\uC11C\uB3C4 \uC4F8 \uC218 \uC788\uB294\uAC00?", {
  x: 1.2, y: 0.3, w: 8.3, h: 0.6, fontSize: 28, fontFace: "Arial Black", color: C.text_d, margin: 0,
});

// Cross-distance table
const t7 = [
  [
    { text: "L", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "No D2NN", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "D2NN(3km)", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
    { text: "D2NN(local)", options: { bold: true, color: "FFFFFF", fill: { color: C.th } } },
  ],
  ["100m", "baseline", { text: "\u20133.2%", options: { color: C.red } }, { text: "+90.6%", options: { bold: true, color: C.accent3 } }],
  [
    { text: "500m", options: { fill: { color: C.ta } } },
    { text: "baseline", options: { fill: { color: C.ta } } },
    { text: "\u201360.5%", options: { bold: true, color: C.red, fill: { color: C.ta } } },
    { text: "+6.2%", options: { color: C.accent3, fill: { color: C.ta } } },
  ],
  ["1km", "baseline", { text: "\u201370.5%", options: { bold: true, color: C.red } }, "+9.0%"],
  [
    { text: "3km", options: { fill: { color: C.ta } } },
    { text: "baseline", options: { fill: { color: C.ta } } },
    { text: "+223%", options: { bold: true, color: C.accent3, fill: { color: C.ta } } },
    { text: "+223%", options: { bold: true, color: C.accent3, fill: { color: C.ta } } },
  ],
];
s7.addTable(t7, { x: 0.5, y: 1.1, w: 4.5, colW: [0.7, 1.1, 1.3, 1.4], fontSize: 12, fontFace: "Calibri",
  border: { pt: 0.5, color: "D5D8DC" }, rowH: [0.35, 0.35, 0.35, 0.35, 0.35] });

// Insight
s7.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.4, w: 4.5, h: 0.9, fill: { color: "FEE2E2" } });
s7.addText([
  { text: "3km \uBAA8\uB378 \u2192 500m: \u201360.5%\n", options: { bold: true, color: C.red, breakLine: true } },
  { text: "Static D2NN = operating condition \uC804\uC6A9", options: { color: C.text_d } },
], { x: 0.7, y: 3.5, w: 4.1, h: 0.7, fontSize: 13, fontFace: "Calibri", margin: 0 });

// Right: focal plane comparison
s7.addImage({ path: IMG.cross, x: 5.2, y: 1.0, w: 4.5, h: 3.5 });
s7.addText("L=1km: 3km model(3\uBC88\uC9F8) vs local model(4\uBC88\uC9F8)", {
  x: 5.2, y: 4.6, w: 4.5, h: 0.3, fontSize: 10, fontFace: "Calibri", color: C.text_m, align: "center", margin: 0 });


// ═══════════════════════════════════════════════════════════
// SLIDE 8: Conclusions
// ═══════════════════════════════════════════════════════════
let s8 = pres.addSlide();
s8.background = { color: C.bg_dark };
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
s8.addText("Conclusions & Next Steps", {
  x: 0.6, y: 0.3, w: 9, h: 0.6, fontSize: 32, fontFace: "Arial Black", color: C.text_w, margin: 0,
});

const conclusions = [
  { num: "1", text: "Normalized PIB \u2260 \uC218\uC2E0\uC804\uB825. Throughput\uC774 \uD575\uC2EC.", color: C.accent },
  { num: "2", text: "Raw RP loss\uAC00 \uCD5C\uC801. Vacuum reference \uBD88\uD544\uC694.", color: C.accent2 },
  { num: "3", text: "D2NN = beam shaper + statistical corrector.\n\uC7A5\uAC70\uB9AC(D/r\u2080>7)\uC5D0\uC11C +5 dB, deep fade +8 dB.", color: C.accent3 },
  { num: "4", text: "Static D2NN\uC740 \uC870\uAC74 \uC804\uC6A9. \uBC94\uC6A9 \uBD88\uAC00.", color: C.red },
];
conclusions.forEach((c, i) => {
  const cy = 1.2 + i * 0.95;
  s8.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: cy, w: 4.5, h: 0.8, fill: { color: C.bg_mid } });
  s8.addShape(pres.shapes.OVAL, { x: 0.75, y: cy + 0.12, w: 0.5, h: 0.5, fill: { color: c.color } });
  s8.addText(c.num, { x: 0.75, y: cy + 0.12, w: 0.5, h: 0.5, fontSize: 18, fontFace: "Arial Black", color: C.text_w, align: "center", valign: "middle", margin: 0 });
  s8.addText(c.text, { x: 1.4, y: cy + 0.05, w: 3.5, h: 0.7, fontSize: 13, fontFace: "Calibri", color: "CADCFC", margin: 0 });
});

// Right: next steps
s8.addText("Next Steps", { x: 5.5, y: 1.1, w: 4, h: 0.4, fontSize: 18, fontFace: "Arial Black", color: C.accent, margin: 0 });
const nexts = [
  "Cn\u00B2 sweep \u2014 \uB09C\uB958 \uC138\uAE30\uBCC4 D2NN \uCD5C\uC801 \uC870\uAC74",
  "Switchable masks \u2014 \uAC70\uB9AC/\uB09C\uB958\uBCC4 2\u20133\uC138\uD2B8",
  "Dynamic D2NN \u2014 Zernike \uAE30\uBC18 (100 params)",
  "Hybrid AO(\uC800\uCC28) + D2NN(\uACE0\uCC28)",
];
nexts.forEach((n, i) => {
  const cy = 1.7 + i * 0.7;
  s8.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: cy, w: 4.0, h: 0.55, fill: { color: C.bg_mid } });
  s8.addText(n, { x: 5.7, y: cy + 0.05, w: 3.6, h: 0.45, fontSize: 12, fontFace: "Calibri", color: "CADCFC", margin: 0 });
});

// Bottom bar
s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.0, w: 10, h: 0.625, fill: { color: C.accent } });
s8.addText("Static D2NN\uC758 \uD55C\uACC4\uB97C \uC774\uD574\uD558\uB294 \uAC83\uC774 Dynamic D2NN \uC124\uACC4\uC758 \uCD9C\uBC1C\uC810", {
  x: 0.5, y: 5.05, w: 9.0, h: 0.52, fontSize: 15, fontFace: "Arial Black", color: C.text_w, valign: "middle", margin: 0,
});


// Save
const outPath = "/root/dj/D2NN/kim2026/autoresearch/runs/0406-d2nn-seminar-questions.pptx";
pres.writeFile({ fileName: outPath }).then(() => console.log("Saved: " + outPath));
