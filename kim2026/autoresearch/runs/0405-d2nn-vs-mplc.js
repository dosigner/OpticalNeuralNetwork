const pptxgen = require("pptxgenjs");
const fs = require("fs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33" x 7.5"
pres.author = "Kim Lab";
pres.title = "Static D2NN vs MPLC: Atmospheric Turbulence Mitigation";

// Color palette: Ocean Gradient
const C = {
  primary: "065A82",    // deep blue
  secondary: "1C7293",  // teal
  accent: "21295C",     // midnight
  light: "F0F4F8",      // ice
  white: "FFFFFF",
  text: "1E293B",       // slate
  muted: "64748B",      // gray
  positive: "059669",   // green
  negative: "DC2626",   // red
  orange: "D97706",     // amber
  card: "F8FAFC",       // very light
};

const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.12 });

// ============================================================
// SLIDE 1: Title
// ============================================================
let s1 = pres.addSlide();
s1.background = { color: C.accent };
// Title
s1.addText("Static D2NN vs MPLC", {
  x: 0.8, y: 1.5, w: 11.7, h: 1.5,
  fontSize: 44, fontFace: "Arial Black", color: C.white, bold: true, margin: 0,
});
s1.addText("Atmospheric Turbulence Mitigation Approaches", {
  x: 0.8, y: 3.0, w: 11.7, h: 0.8,
  fontSize: 24, fontFace: "Calibri", color: "CADCFC", margin: 0,
});
// Divider line
s1.addShape(pres.shapes.RECTANGLE, {
  x: 0.8, y: 4.0, w: 3.0, h: 0.06, fill: { color: "CADCFC" },
});
s1.addText("Kim Lab  |  April 2026", {
  x: 0.8, y: 4.4, w: 11.7, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: "8BADC7", margin: 0,
});
s1.addText("Structural Similarity, Fundamental Differences", {
  x: 0.8, y: 5.2, w: 11.7, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: "6B8DAF", italic: true, margin: 0,
});

// ============================================================
// SLIDE 2: Architecture Comparison
// ============================================================
let s2 = pres.addSlide();
s2.background = { color: C.white };
s2.addText("Both Are Multi-Plane Phase Systems", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// Left card: Shared Architecture
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.3, w: 5.8, h: 4.5, fill: { color: C.card },
  line: { color: "E2E8F0", width: 1 }, shadow: makeShadow(),
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.3, w: 5.8, h: 0.06, fill: { color: C.primary },
});
s2.addText("SHARED ARCHITECTURE", {
  x: 0.9, y: 1.5, w: 5.2, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
});
s2.addText([
  { text: "Multiple phase-only masks + free-space propagation", options: { bullet: true, breakLine: true } },
  { text: "Unitary transformation (energy conservation)", options: { bullet: true, breakLine: true } },
  { text: "Cascaded Fourier / Fresnel transforms", options: { bullet: true, breakLine: true } },
  { text: "Phase mask → Propagate → Phase mask → ...", options: { bullet: true, breakLine: true } },
  { text: "U\u2097\u208A\u2081 = P_d { U\u2097 \u00B7 exp(i\u03C6\u2097) }", options: { bullet: true, breakLine: true, fontFace: "Consolas", fontSize: 13 } },
], {
  x: 0.9, y: 2.1, w: 5.2, h: 3.5,
  fontSize: 14, fontFace: "Calibri", color: C.text, paraSpaceAfter: 8,
});

// Right card: Key Difference
s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.9, y: 1.3, w: 5.8, h: 4.5, fill: { color: C.card },
  line: { color: "E2E8F0", width: 1 }, shadow: makeShadow(),
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.9, y: 1.3, w: 5.8, h: 0.06, fill: { color: C.negative },
});
s2.addText("KEY DIFFERENCE", {
  x: 7.2, y: 1.5, w: 5.2, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: C.negative, bold: true, margin: 0,
});
s2.addText([
  { text: "MPLC: ", options: { bold: true } },
  { text: "Analytically designed (wavefront matching algorithm)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "D2NN: ", options: { bold: true } },
  { text: "Data-driven training (stochastic gradient descent)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "MPLC: ", options: { bold: true } },
  { text: "Adaptive — different response per realization", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "D2NN: ", options: { bold: true } },
  { text: "Static — same fixed mask for all inputs", options: { breakLine: true } },
], {
  x: 7.2, y: 2.1, w: 5.2, h: 3.5,
  fontSize: 14, fontFace: "Calibri", color: C.text,
});

// ============================================================
// SLIDE 3: MPLC — Mode Decomposition
// ============================================================
let s3 = pres.addSlide();
s3.background = { color: C.white };
s3.addText("MPLC: Hermite-Gaussian Mode Decomposition", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 30, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// Equation box
s3.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.2, w: 12.1, h: 1.2, fill: { color: "EEF2FF" },
  line: { color: "C7D2FE", width: 1 },
});
s3.addText([
  { text: "U", options: { fontFace: "Cambria", italic: true } },
  { text: "in", options: { fontFace: "Cambria", fontSize: 10 } },
  { text: "(x,y) = \u03A3 c", options: { fontFace: "Cambria" } },
  { text: "n", options: { fontFace: "Cambria", fontSize: 10 } },
  { text: " \u03C8", options: { fontFace: "Cambria", italic: true } },
  { text: "n", options: { fontFace: "Cambria", fontSize: 10 } },
  { text: "(x,y)      \u2192      MPLC separates each \u03C8", options: { fontFace: "Cambria" } },
  { text: "n", options: { fontFace: "Cambria", fontSize: 10 } },
  { text: " to a different spatial port", options: { fontFace: "Cambria" } },
], {
  x: 0.8, y: 1.35, w: 11.7, h: 0.9,
  fontSize: 18, color: C.accent, align: "center", valign: "middle",
});

// Content
s3.addText([
  { text: "Decomposes input into orthogonal HG/LG modes", options: { bullet: true, breakLine: true } },
  { text: "Each mode sorted to separate spatial channel", options: { bullet: true, breakLine: true } },
  { text: "Per-mode phase/amplitude measurement possible", options: { bullet: true, breakLine: true } },
  { text: "Reconstruct corrected beam by selective recombination", options: { bullet: true, breakLine: true } },
  { text: "Requires ADAPTIVE control for turbulence correction", options: { bullet: true, breakLine: true, bold: true, color: C.negative } },
  { text: "Real-time mode sensing + feedback loop needed", options: { bullet: true, breakLine: true } },
  { text: "Ref: Cailabs (commercial), Morizur et al. JOSA A 2010", options: { bullet: true, italic: true, color: C.muted } },
], {
  x: 0.8, y: 2.7, w: 11.7, h: 4.2,
  fontSize: 15, fontFace: "Calibri", color: C.text, paraSpaceAfter: 6,
});

// ============================================================
// SLIDE 4: Our D2NN — Statistical Energy Redistribution
// ============================================================
let s4 = pres.addSlide();
s4.background = { color: C.white };
s4.addText("Our D2NN: Statistical Energy Redistribution", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 30, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// Loss function box
s4.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.2, w: 12.1, h: 1.0, fill: { color: "ECFDF5" },
  line: { color: "A7F3D0", width: 1 },
});
s4.addText("L = \u2212log( E_bucket / E_input + \u03B5 )", {
  x: 0.8, y: 1.3, w: 11.7, h: 0.8,
  fontSize: 22, fontFace: "Consolas", color: C.positive, align: "center", valign: "middle", bold: true,
});

s4.addText([
  { text: "Fixed (static) masks trained on turbulence ensemble", options: { bullet: true, breakLine: true } },
  { text: "No mode decomposition \u2014 same transformation for ALL realizations", options: { bullet: true, breakLine: true } },
  { text: "Optimizes ensemble-average received power in focal bucket", options: { bullet: true, breakLine: true } },
  { text: "Mechanism: NOT wavefront correction (WF RMS unchanged at ~442nm)", options: { bullet: true, breakLine: true, bold: true, color: C.negative } },
  { text: "Mechanism: PSF reshaping + deep fade floor lifting", options: { bullet: true, breakLine: true, bold: true, color: C.positive } },
  { text: "PASSIVE \u2014 zero power consumption, no sensing required", options: { bullet: true, breakLine: true } },
  { text: "Throughput preserved (0.88\u20130.996) without explicit penalty", options: { bullet: true } },
], {
  x: 0.8, y: 2.5, w: 11.7, h: 4.2,
  fontSize: 15, fontFace: "Calibri", color: C.text, paraSpaceAfter: 6,
});

// ============================================================
// SLIDE 5: Head-to-Head Comparison Table
// ============================================================
let s5 = pres.addSlide();
s5.background = { color: C.white };
s5.addText("Head-to-Head Comparison", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

const headerOpts = { fill: { color: C.primary }, color: C.white, bold: true, fontSize: 12, fontFace: "Calibri", align: "center", valign: "middle" };
const cellOpts = { fontSize: 12, fontFace: "Calibri", color: C.text, valign: "middle" };
const cellCenter = { ...cellOpts, align: "center" };

const tableRows = [
  [
    { text: "Aspect", options: headerOpts },
    { text: "MPLC (Adaptive)", options: headerOpts },
    { text: "Our D2NN (Static)", options: headerOpts },
  ],
  [{ text: "Design Method", options: cellOpts }, { text: "Analytical (WFM)", options: cellCenter }, { text: "Data-driven (SGD)", options: cellCenter }],
  [{ text: "Operation", options: cellOpts }, { text: "Mode decomposition", options: cellCenter }, { text: "Energy redistribution", options: cellCenter }],
  [{ text: "Adaptivity", options: cellOpts }, { text: "Per-realization", options: cellCenter }, { text: "Fixed mask", options: cellCenter }],
  [{ text: "WF Correction", options: cellOpts }, { text: "\u2714 Yes (per mode)", options: { ...cellCenter, color: C.positive } }, { text: "\u2718 No (WF RMS unchanged)", options: { ...cellCenter, color: C.negative } }],
  [{ text: "Turbulence Handling", options: cellOpts }, { text: "Adaptive \u2014 real-time", options: cellCenter }, { text: "Statistical \u2014 ensemble avg", options: cellCenter }],
  [{ text: "Power Consumption", options: cellOpts }, { text: "Active (sensing + control)", options: { ...cellCenter, color: C.orange } }, { text: "Zero (passive)", options: { ...cellCenter, color: C.positive, bold: true } }],
  [{ text: "System Complexity", options: cellOpts }, { text: "High (detector + SLM + FPGA)", options: cellCenter }, { text: "Low (fixed diffractive element)", options: { ...cellCenter, color: C.positive } }],
  [{ text: "Performance", options: cellOpts }, { text: "Full correction possible", options: { ...cellCenter, color: C.positive } }, { text: "+6% to +223% RP improvement", options: cellCenter }],
  [{ text: "Fade Mitigation", options: cellOpts }, { text: "Complete (adaptive)", options: cellCenter }, { text: "Partial (8 dB at 3 km)", options: cellCenter }],
  [{ text: "Cost", options: cellOpts }, { text: "$$$ (SLM, detector, FPGA)", options: { ...cellCenter, color: C.negative } }, { text: "$ (fabricated once)", options: { ...cellCenter, color: C.positive, bold: true } }],
];

s5.addTable(tableRows, {
  x: 0.6, y: 1.2, w: 12.1,
  colW: [2.8, 4.65, 4.65],
  border: { pt: 0.5, color: "E2E8F0" },
  rowH: [0.45, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
  autoPage: false,
});

// ============================================================
// SLIDE 6: Performance by Turbulence Regime
// ============================================================
let s6 = pres.addSlide();
s6.background = { color: C.white };
s6.addText("Performance by Turbulence Regime", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// Three regime cards
const regimes = [
  { label: "WEAK", sub: "D/r\u2080 < 2", color: "059669", bgColor: "ECFDF5", borderColor: "A7F3D0",
    mplc: "Near-perfect correction\nStrehl \u2192 1.0", d2nn: "PSF reshaping\n+91% RP (diffractive lens effect)" },
  { label: "MODERATE", sub: "D/r\u2080 = 3\u20135", color: "D97706", bgColor: "FFFBEB", borderColor: "FDE68A",
    mplc: "Good correction\nLimited by mode count", d2nn: "\"Correction gap\"\nOnly +6\u20139% (static vs random)" },
  { label: "STRONG", sub: "D/r\u2080 > 7", color: "DC2626", bgColor: "FEF2F2", borderColor: "FECACA",
    mplc: "Challenging\nScintillation limits sensing", d2nn: "Surprisingly effective\n+77% to +223% fade mitigation" },
];

regimes.forEach((r, i) => {
  const x = 0.6 + i * 4.1;
  // Card
  s6.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.2, w: 3.8, h: 5.5, fill: { color: r.bgColor },
    line: { color: r.borderColor, width: 1 }, shadow: makeShadow(),
  });
  // Top bar
  s6.addShape(pres.shapes.RECTANGLE, {
    x: x, y: 1.2, w: 3.8, h: 0.06, fill: { color: r.color },
  });
  // Label
  s6.addText(r.label, {
    x: x + 0.2, y: 1.4, w: 3.4, h: 0.5,
    fontSize: 20, fontFace: "Arial Black", color: r.color, bold: true, margin: 0,
  });
  s6.addText(r.sub, {
    x: x + 0.2, y: 1.9, w: 3.4, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: C.muted, margin: 0,
  });
  // MPLC section
  s6.addText("MPLC", {
    x: x + 0.2, y: 2.5, w: 3.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s6.addText(r.mplc, {
    x: x + 0.2, y: 2.9, w: 3.4, h: 1.2,
    fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
  });
  // D2NN section
  s6.addText("D2NN", {
    x: x + 0.2, y: 4.3, w: 3.4, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: C.positive, bold: true, margin: 0,
  });
  s6.addText(r.d2nn, {
    x: x + 0.2, y: 4.7, w: 3.4, h: 1.2,
    fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
  });
});

// Key insight bar
s6.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 6.85, w: 12.1, h: 0.5, fill: { color: C.accent },
});
s6.addText("D2NN excels where MPLC struggles (deep turbulence) and vice versa (moderate turbulence)", {
  x: 0.8, y: 6.85, w: 11.7, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle",
});

// ============================================================
// SLIDE 7: What Should We Call This?
// ============================================================
let s7 = pres.addSlide();
s7.background = { color: C.white };
s7.addText("What Should We Call This?", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// NOT boxes
s7.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.2, w: 5.8, h: 1.6, fill: { color: "FEF2F2" },
  line: { color: "FECACA", width: 1 },
});
s7.addText("NOT \"turbulence compensation\"", {
  x: 0.8, y: 1.3, w: 5.4, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: C.negative, bold: true, margin: 0,
});
s7.addText("No wavefront correction occurs", {
  x: 0.8, y: 1.8, w: 5.4, h: 0.3,
  fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
});
s7.addText("NOT \"adaptive optics\"", {
  x: 0.8, y: 2.2, w: 5.4, h: 0.5,
  fontSize: 16, fontFace: "Calibri", color: C.negative, bold: true, margin: 0,
});

// Accurate terms box
s7.addShape(pres.shapes.RECTANGLE, {
  x: 6.9, y: 1.2, w: 5.8, h: 1.6, fill: { color: "ECFDF5" },
  line: { color: "A7F3D0", width: 1 },
});
s7.addText("MORE ACCURATE TERMS", {
  x: 7.1, y: 1.3, w: 5.4, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: C.positive, bold: true, margin: 0,
});
s7.addText([
  { text: "Turbulence-resilient passive beam shaping", options: { bullet: true, breakLine: true } },
  { text: "Ensemble-optimized diffractive spatial filter", options: { bullet: true, breakLine: true } },
  { text: "Static diffractive fade mitigation", options: { bullet: true } },
], {
  x: 7.1, y: 1.7, w: 5.4, h: 1.0,
  fontSize: 13, fontFace: "Calibri", color: C.text,
});

// Evidence table
const evRows = [
  [{ text: "Evidence", options: headerOpts }, { text: "Supports \"Mitigation\"?", options: headerOpts }],
  [{ text: "Received power +6\u2013223%", options: cellOpts }, { text: "\u2714 Yes", options: { ...cellCenter, color: C.positive, bold: true } }],
  [{ text: "Deep fades reduced by 8 dB", options: cellOpts }, { text: "\u2714 Yes", options: { ...cellCenter, color: C.positive, bold: true } }],
  [{ text: "Zero power consumption", options: cellOpts }, { text: "\u2714 Yes", options: { ...cellCenter, color: C.positive, bold: true } }],
  [{ text: "WF RMS unchanged (~442 nm)", options: cellOpts }, { text: "\u2718 No", options: { ...cellCenter, color: C.negative, bold: true } }],
  [{ text: "No per-realization correction", options: cellOpts }, { text: "\u2718 No", options: { ...cellCenter, color: C.negative, bold: true } }],
  [{ text: "Limited at moderate turbulence", options: cellOpts }, { text: "\u2718 No", options: { ...cellCenter, color: C.negative, bold: true } }],
];
s7.addTable(evRows, {
  x: 0.6, y: 3.2, w: 12.1, colW: [8.0, 4.1],
  border: { pt: 0.5, color: "E2E8F0" }, rowH: [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
  autoPage: false,
});

// ============================================================
// SLIDE 8: Hybrid Architecture Proposal
// ============================================================
let s8 = pres.addSlide();
s8.background = { color: C.white };
s8.addText("Complementary Hybrid: D2NN + MPLC", {
  x: 0.6, y: 0.3, w: 12, h: 0.7,
  fontSize: 32, fontFace: "Arial Black", color: C.primary, bold: true, margin: 0,
});

// Pipeline: 3 boxes with arrows
const stages = [
  { label: "TURBULENT\nBEAM", color: C.negative, x: 0.6 },
  { label: "D2NN\n(Passive Stage 1)", color: C.positive, x: 3.4 },
  { label: "MPLC\n(Active Stage 2)", color: C.primary, x: 6.2 },
  { label: "CORRECTED\nBEAM", color: "059669", x: 9.0 },
];

stages.forEach((st) => {
  s8.addShape(pres.shapes.RECTANGLE, {
    x: st.x, y: 1.3, w: 2.5, h: 1.2, fill: { color: st.color },
    shadow: makeShadow(),
  });
  s8.addText(st.label, {
    x: st.x, y: 1.3, w: 2.5, h: 1.2,
    fontSize: 13, fontFace: "Calibri", color: C.white, bold: true,
    align: "center", valign: "middle",
  });
});
// Arrows
[3.1, 5.9, 8.7].forEach((ax) => {
  s8.addText("\u25B6", {
    x: ax, y: 1.55, w: 0.3, h: 0.7,
    fontSize: 22, color: C.muted, align: "center", valign: "middle",
  });
});

// Benefits
s8.addText("Why This Works", {
  x: 0.6, y: 2.9, w: 12, h: 0.5,
  fontSize: 20, fontFace: "Arial Black", color: C.accent, margin: 0,
});

const benefits = [
  { title: "Reduced Dynamic Range", desc: "D2NN compresses fade depth \u2192 MPLC detector needs less dynamic range" },
  { title: "Lower Update Rate", desc: "Pre-stabilized beam \u2192 MPLC can operate at slower refresh rate" },
  { title: "Graceful Degradation", desc: "If MPLC fails, D2NN still provides baseline improvement" },
  { title: "Cost Optimization", desc: "Simpler MPLC (fewer modes) needed after D2NN pre-conditioning" },
];

benefits.forEach((b, i) => {
  const bx = 0.6 + (i % 2) * 6.2;
  const by = 3.6 + Math.floor(i / 2) * 1.8;
  s8.addShape(pres.shapes.RECTANGLE, {
    x: bx, y: by, w: 5.8, h: 1.4, fill: { color: C.card },
    line: { color: "E2E8F0", width: 1 }, shadow: makeShadow(),
  });
  s8.addShape(pres.shapes.RECTANGLE, {
    x: bx, y: by, w: 0.08, h: 1.4, fill: { color: C.primary },
  });
  s8.addText(b.title, {
    x: bx + 0.3, y: by + 0.15, w: 5.3, h: 0.4,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, margin: 0,
  });
  s8.addText(b.desc, {
    x: bx + 0.3, y: by + 0.6, w: 5.3, h: 0.6,
    fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
  });
});

// ============================================================
// SLIDE 9: Summary
// ============================================================
let s9 = pres.addSlide();
s9.background = { color: C.accent };
s9.addText("Summary", {
  x: 0.8, y: 0.4, w: 11.7, h: 0.8,
  fontSize: 36, fontFace: "Arial Black", color: C.white, bold: true, margin: 0,
});

const summaryItems = [
  "D2NN and MPLC share the same architecture (multi-plane phase + propagation)",
  "MPLC = adaptive mode decomposition;  D2NN = static statistical optimization",
  "D2NN is NOT turbulence \"correction\" \u2014 it's turbulence \"resilience\"",
  "D2NN advantage: zero power, simple fabrication, effective in deep turbulence",
  "MPLC advantage: true wavefront correction, full mode control",
  "Future: D2NN (passive) + MPLC (active) hybrid could combine strengths",
];

summaryItems.forEach((item, i) => {
  const num = String(i + 1);
  // Number circle
  s9.addShape(pres.shapes.OVAL, {
    x: 0.8, y: 1.6 + i * 0.9, w: 0.5, h: 0.5,
    fill: { color: "CADCFC" },
  });
  s9.addText(num, {
    x: 0.8, y: 1.6 + i * 0.9, w: 0.5, h: 0.5,
    fontSize: 16, fontFace: "Arial Black", color: C.accent, align: "center", valign: "middle",
  });
  s9.addText(item, {
    x: 1.5, y: 1.6 + i * 0.9, w: 11.0, h: 0.5,
    fontSize: 15, fontFace: "Calibri", color: "CADCFC", valign: "middle", margin: 0,
  });
});

// Write
const outPath = "/root/dj/D2NN/kim2026/autoresearch/runs/0405-d2nn-vs-mplc-comparison.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPTX saved to: " + outPath);
}).catch(err => {
  console.error("Error:", err);
});
