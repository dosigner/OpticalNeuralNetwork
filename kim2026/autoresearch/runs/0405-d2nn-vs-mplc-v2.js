const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Kim Lab";
pres.title = "D2NN vs MPLC: Physics-Based Comparison";

const FIG = "/root/dj/D2NN/kim2026/autoresearch/runs/0405-mplc-figs";
const SWEEP = "/root/dj/D2NN/kim2026/autoresearch/runs/0405-distance-sweep-rawrp-f6p5mm";

const C = {
  bg: "0F172A", primary: "1E40AF", accent: "7C3AED", teal: "0D9488",
  green: "059669", red: "DC2626", amber: "D97706",
  white: "FFFFFF", light: "F1F5F9", text: "1E293B", muted: "64748B",
  card: "F8FAFC", ice: "EFF6FF",
};

const makeShadow = () => ({ type: "outer", blur: 5, offset: 2, angle: 135, color: "000000", opacity: 0.10 });

// Helper: slide number
let slideNum = 0;
function addSlideNum(slide) {
  slideNum++;
  slide.addText(String(slideNum), {
    x: 12.6, y: 7.0, w: 0.5, h: 0.35,
    fontSize: 10, color: C.muted, align: "center", fontFace: "Calibri",
  });
}

// ============================================================
// SLIDE 1: Title
// ============================================================
let s1 = pres.addSlide();
s1.background = { color: C.bg };
s1.addText("D2NN vs MPLC", {
  x: 0.8, y: 1.8, w: 11.7, h: 1.2,
  fontSize: 48, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
});
s1.addText("Physics of Multi-Plane Phase Systems\nfor Atmospheric Turbulence Mitigation", {
  x: 0.8, y: 3.2, w: 11.7, h: 1.0,
  fontSize: 22, fontFace: "Calibri", color: "94A3B8", margin: 0,
});
s1.addText("Kim Lab  \u00B7  April 2026", {
  x: 0.8, y: 5.0, w: 5, h: 0.4,
  fontSize: 14, fontFace: "Calibri", color: "64748B", margin: 0,
});
s1.addText("\u03BB = 1.55 \u03BCm  \u00B7  Cn\u00B2 = 5\u00D710\u207B\u00B9\u2074  \u00B7  100m \u2013 3km  \u00B7  5-layer D2NN", {
  x: 0.8, y: 5.5, w: 11, h: 0.4,
  fontSize: 13, fontFace: "Consolas", color: "475569", margin: 0,
});
addSlideNum(s1);

// ============================================================
// SLIDE 2: Shared Architecture — Unitary Decomposition
// ============================================================
let s2 = pres.addSlide();
s2.background = { color: C.white };
s2.addText("The Shared Foundation: Unitary Decomposition Theorem", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s2.addText("Both D2NN and MPLC implement the same physics: cascaded phase masks with free-space propagation can realize any unitary spatial transformation (Reck et al. 1994, Morizur et al. 2010).", {
  x: 0.6, y: 1.0, w: 12, h: 0.6,
  fontSize: 14, fontFace: "Calibri", color: C.text, margin: 0,
});
s2.addImage({ path: FIG + "/fig4_unitary_theorem.png", x: 0.3, y: 1.7, w: 12.7, h: 3.5 });
s2.addText([
  { text: "Key implication: ", options: { bold: true } },
  { text: "With sufficient layers L, both systems can implement ANY mode transformation \u2014 the difference lies in ", options: {} },
  { text: "how", options: { bold: true, italic: true } },
  { text: " the phase masks are determined.", options: {} },
], {
  x: 0.6, y: 5.5, w: 12, h: 0.7,
  fontSize: 15, fontFace: "Calibri", color: C.text, margin: 0,
  valign: "middle",
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 6.4, w: 5.5, h: 0.7, fill: { color: "DBEAFE" },
});
s2.addText("MPLC: masks designed analytically via wavefront matching algorithm", {
  x: 0.8, y: 6.4, w: 5.3, h: 0.7,
  fontSize: 13, fontFace: "Calibri", color: C.primary, bold: true, valign: "middle", margin: 0,
});
s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.4, y: 6.4, w: 5.5, h: 0.7, fill: { color: "DCFCE7" },
});
s2.addText("D2NN: masks trained end-to-end via stochastic gradient descent", {
  x: 6.6, y: 6.4, w: 5.3, h: 0.7,
  fontSize: 13, fontFace: "Calibri", color: C.green, bold: true, valign: "middle", margin: 0,
});
addSlideNum(s2);

// ============================================================
// SLIDE 3: HG Mode Basis
// ============================================================
let s3 = pres.addSlide();
s3.background = { color: C.white };
s3.addText("Hermite-Gaussian Mode Basis", {
  x: 0.6, y: 0.35, w: 8, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s3.addImage({ path: FIG + "/fig1_hg_modes.png", x: 0.3, y: 1.1, w: 7.5, h: 5.8 });
// Right side: explanation
s3.addShape(pres.shapes.RECTANGLE, {
  x: 8.1, y: 1.1, w: 4.8, h: 5.8, fill: { color: C.ice },
});
s3.addText("Mode Decomposition", {
  x: 8.3, y: 1.3, w: 4.4, h: 0.4,
  fontSize: 18, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s3.addText([
  { text: "Any beam can be decomposed into orthogonal HG modes:", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "U(x,y) = \u03A3 c\u2099 \u03C8\u2099(x,y)", options: { fontFace: "Consolas", fontSize: 15, color: C.primary, bold: true, breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "where \u03C8\u2099 = HG mode function", options: { breakLine: true, fontFace: "Consolas", fontSize: 12, color: C.muted } },
  { text: "c\u2099 = complex mode coefficient", options: { breakLine: true, fontFace: "Consolas", fontSize: 12, color: C.muted } },
  { text: "", options: { breakLine: true, fontSize: 10 } },
  { text: "Turbulence redistributes energy across modes:", options: { breakLine: true, bold: true } },
  { text: "", options: { breakLine: true, fontSize: 6 } },
  { text: "\u2022 Weak turbulence: most energy in HG\u2080\u2080 (fundamental)", options: { breakLine: true } },
  { text: "\u2022 Strong turbulence: energy spreads to many higher-order modes", options: { breakLine: true } },
  { text: "\u2022 Number of significant modes \u2248 (D/r\u2080)\u00B2", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 10 } },
  { text: "MPLC sorts these modes spatially.", options: { bold: true, color: C.primary, breakLine: true } },
  { text: "D2NN ignores mode structure entirely.", options: { bold: true, color: C.green } },
], {
  x: 8.3, y: 1.8, w: 4.4, h: 4.8,
  fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
});
addSlideNum(s3);

// ============================================================
// SLIDE 4: MPLC Architecture
// ============================================================
let s4 = pres.addSlide();
s4.background = { color: C.white };
s4.addText("MPLC: Adaptive Mode Sorting Architecture", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s4.addImage({ path: FIG + "/fig2_mplc_architecture.png", x: 0.2, y: 1.1, w: 12.9, h: 4.2 });
s4.addText([
  { text: "How it works: ", options: { bold: true, color: C.primary } },
  { text: "Each phase plane \u03C6\u2097 is designed so that after L planes, mode \u03C8\u2099 emerges at spatial port n. ", options: {} },
  { text: "The wavefront matching algorithm (Sakamaki et al.) iteratively optimizes each plane to maximize mode separation.", options: {} },
], {
  x: 0.6, y: 5.5, w: 12, h: 0.7,
  fontSize: 14, fontFace: "Calibri", color: C.text, margin: 0,
});
s4.addText([
  { text: "For turbulence compensation: ", options: { bold: true, color: C.red } },
  { text: "requires coherent detection at each port + real-time DSP to measure c\u2099 and recombine. This makes it an ", options: {} },
  { text: "active", options: { bold: true, italic: true, color: C.red } },
  { text: " system with significant power consumption and latency requirements.", options: {} },
], {
  x: 0.6, y: 6.3, w: 12, h: 0.7,
  fontSize: 14, fontFace: "Calibri", color: C.text, margin: 0,
});
addSlideNum(s4);

// ============================================================
// SLIDE 5: Cailabs PROTEUS
// ============================================================
let s5 = pres.addSlide();
s5.background = { color: C.white };
s5.addText("Cailabs PROTEUS: Commercial MPLC for FSO", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s5.addImage({ path: FIG + "/fig5_cailabs_workflow.png", x: 0.2, y: 1.1, w: 12.9, h: 5.2 });
s5.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 6.5, w: 12.1, h: 0.7, fill: { color: "FEF2F2" },
});
s5.addText("Cailabs approach: MPLC sorts turbulent beam into N HG modes \u2192 coherent Rx per port \u2192 DSP recombines. Requires SLM or fixed MPLC + N photodetectors + FPGA. Cost: tens of k$.", {
  x: 0.8, y: 6.5, w: 11.7, h: 0.7,
  fontSize: 13, fontFace: "Calibri", color: C.red, valign: "middle", margin: 0,
});
addSlideNum(s5);

// ============================================================
// SLIDE 6: D2NN Architecture
// ============================================================
let s6 = pres.addSlide();
s6.background = { color: C.white };
s6.addText("Our D2NN: Static Ensemble-Optimized Phase Masks", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.green, bold: true, margin: 0,
});
s6.addImage({ path: FIG + "/fig3_d2nn_architecture.png", x: 0.2, y: 1.1, w: 12.9, h: 4.2 });
s6.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 5.5, w: 5.8, h: 1.7, fill: { color: "ECFDF5" },
});
s6.addText("Loss Function", {
  x: 0.8, y: 5.6, w: 5.4, h: 0.3,
  fontSize: 14, fontFace: "Calibri", color: C.green, bold: true, margin: 0,
});
s6.addText("L = \u2212log( E_bucket / E_input + \u03B5 )", {
  x: 0.8, y: 5.9, w: 5.4, h: 0.5,
  fontSize: 18, fontFace: "Consolas", color: C.text, bold: true, margin: 0,
});
s6.addText("Directly maximizes absolute focal bucket power.\nNo vacuum reference. Throughput intrinsically preserved.", {
  x: 0.8, y: 6.4, w: 5.4, h: 0.6,
  fontSize: 12, fontFace: "Calibri", color: C.muted, margin: 0,
});
s6.addShape(pres.shapes.RECTANGLE, {
  x: 6.8, y: 5.5, w: 5.8, h: 1.7, fill: { color: C.card },
});
s6.addText("Key Differences from MPLC", {
  x: 7.0, y: 5.6, w: 5.4, h: 0.3,
  fontSize: 14, fontFace: "Calibri", color: C.text, bold: true, margin: 0,
});
s6.addText([
  { text: "\u2022 Same masks for ALL turbulence realizations", options: { breakLine: true } },
  { text: "\u2022 No mode decomposition or sensing", options: { breakLine: true } },
  { text: "\u2022 Zero power consumption (passive)", options: { breakLine: true } },
  { text: "\u2022 WF RMS unchanged \u2014 NOT wavefront correction", options: { breakLine: true, color: C.red, bold: true } },
  { text: "\u2022 Mechanism: energy redistribution + fade floor lifting", options: { color: C.green, bold: true } },
], {
  x: 7.0, y: 5.95, w: 5.4, h: 1.2,
  fontSize: 12, fontFace: "Calibri", color: C.text, margin: 0,
});
addSlideNum(s6);

// ============================================================
// SLIDE 7: Key Equations
// ============================================================
let s7 = pres.addSlide();
s7.background = { color: C.white };
s7.addText("Physics Equations: MPLC vs D2NN", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s7.addImage({ path: FIG + "/fig8_equations.png", x: 0.1, y: 1.0, w: 13.1, h: 6.0 });
addSlideNum(s7);

// ============================================================
// SLIDE 8: Performance Comparison
// ============================================================
let s8 = pres.addSlide();
s8.background = { color: C.white };
s8.addText("Distance Sweep Results: D2NN Received Power", {
  x: 0.6, y: 0.2, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s8.addImage({ path: FIG + "/fig6_performance_comparison.png", x: 0.1, y: 0.9, w: 13.1, h: 5.5 });

// Data table
const hOpts = { fill: { color: C.primary }, color: C.white, bold: true, fontSize: 10, fontFace: "Calibri", align: "center", valign: "middle" };
const cOpts = { fontSize: 10, fontFace: "Calibri", color: C.text, align: "center", valign: "middle" };
const gOpts = { ...cOpts, color: C.green, bold: true };
s8.addTable([
  [{ text: "Distance", options: hOpts }, { text: "D/r\u2080", options: hOpts }, { text: "\u03C3\u1D3F\u00B2", options: hOpts },
   { text: "Vac", options: hOpts }, { text: "Turb", options: hOpts }, { text: "D2NN", options: hOpts },
   { text: "\u0394 (dB)", options: hOpts }, { text: "Fade \u0394", options: hOpts }],
  [{ text: "100m", options: cOpts }, { text: "1.26", options: cOpts }, { text: "0.015", options: cOpts },
   { text: "405.6", options: cOpts }, { text: "382.1", options: cOpts }, { text: "728.4", options: gOpts },
   { text: "+2.8", options: gOpts }, { text: "+3.2 dB", options: gOpts }],
  [{ text: "500m", options: cOpts }, { text: "3.31", options: cOpts }, { text: "0.28", options: cOpts },
   { text: "628.3", options: cOpts }, { text: "499.5", options: cOpts }, { text: "530.5", options: gOpts },
   { text: "+0.26", options: cOpts }, { text: "+1.3 dB", options: cOpts }],
  [{ text: "1km", options: cOpts }, { text: "5.02", options: cOpts }, { text: "1.0", options: cOpts },
   { text: "275.1", options: cOpts }, { text: "166.0", options: cOpts }, { text: "180.9", options: gOpts },
   { text: "+0.37", options: cOpts }, { text: "+2.3 dB", options: cOpts }],
  [{ text: "2km", options: cOpts }, { text: "7.61", options: cOpts }, { text: "3.6", options: cOpts },
   { text: "34.0", options: cOpts }, { text: "9.3", options: cOpts }, { text: "16.4", options: gOpts },
   { text: "+2.5", options: gOpts }, { text: "+7.0 dB", options: gOpts }],
  [{ text: "3km", options: cOpts }, { text: "9.70", options: cOpts }, { text: "7.5", options: cOpts },
   { text: "4.6", options: cOpts }, { text: "0.64", options: cOpts }, { text: "2.08", options: gOpts },
   { text: "+5.1", options: gOpts }, { text: "+8.1 dB", options: gOpts }],
], {
  x: 0.6, y: 6.45, w: 12.1, colW: [1.2, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.9],
  border: { pt: 0.5, color: "E2E8F0" }, rowH: [0.3, 0.25, 0.25, 0.25, 0.25, 0.25],
  autoPage: false,
});
addSlideNum(s8);

// ============================================================
// SLIDE 9: Three Physical Mechanisms
// ============================================================
let s9 = pres.addSlide();
s9.background = { color: C.white };
s9.addText("Physical Mechanisms by Turbulence Regime", {
  x: 0.6, y: 0.2, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});
s9.addImage({ path: FIG + "/fig7_mechanism_diagram.png", x: 0.1, y: 0.9, w: 13.1, h: 5.2 });
s9.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 6.3, w: 12.1, h: 1.0, fill: { color: C.bg },
});
s9.addText("Static D2NN cannot correct individual wavefronts. It performs ensemble-averaged energy redistribution: PSF reshaping (weak), statistical mode filtering (moderate), deep fade floor lifting (strong).", {
  x: 0.8, y: 6.3, w: 11.7, h: 1.0,
  fontSize: 14, fontFace: "Calibri", color: "CBD5E1", valign: "middle", margin: 0,
});
addSlideNum(s9);

// ============================================================
// SLIDE 10: Head-to-Head Comparison
// ============================================================
let s10 = pres.addSlide();
s10.background = { color: C.white };
s10.addText("MPLC vs D2NN: Head-to-Head", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});

const th = { fill: { color: C.primary }, color: C.white, bold: true, fontSize: 13, fontFace: "Calibri", align: "center", valign: "middle" };
const td = { fontSize: 13, fontFace: "Calibri", color: C.text, valign: "middle", align: "center" };
const tg = { ...td, color: C.green, bold: true };
const tr = { ...td, color: C.red };

s10.addTable([
  [{ text: "", options: th }, { text: "MPLC (Adaptive)", options: th }, { text: "D2NN (Static)", options: th }],
  [{ text: "Design", options: { ...td, bold: true, align: "left" } }, { text: "Wavefront matching algorithm", options: td }, { text: "SGD on turbulence ensemble", options: td }],
  [{ text: "Operation", options: { ...td, bold: true, align: "left" } }, { text: "Mode decomposition \u2192 sorting", options: td }, { text: "Energy redistribution", options: td }],
  [{ text: "Adaptivity", options: { ...td, bold: true, align: "left" } }, { text: "Per-realization (real-time)", options: tg }, { text: "Fixed mask (static)", options: tr }],
  [{ text: "WF Correction", options: { ...td, bold: true, align: "left" } }, { text: "\u2714 Yes", options: tg }, { text: "\u2718 No (WF RMS unchanged)", options: tr }],
  [{ text: "Power", options: { ...td, bold: true, align: "left" } }, { text: "Active (sensing + DSP)", options: tr }, { text: "Zero (passive element)", options: tg }],
  [{ text: "Complexity", options: { ...td, bold: true, align: "left" } }, { text: "SLM + N detectors + FPGA", options: td }, { text: "Fixed diffractive element", options: tg }],
  [{ text: "Weak Turb", options: { ...td, bold: true, align: "left" } }, { text: "Strehl \u2192 1.0", options: tg }, { text: "+91% RP (PSF reshaping)", options: tg }],
  [{ text: "Moderate Turb", options: { ...td, bold: true, align: "left" } }, { text: "Good (mode-limited)", options: tg }, { text: "+6\u20139% only (correction gap)", options: tr }],
  [{ text: "Strong Turb", options: { ...td, bold: true, align: "left" } }, { text: "Difficult (scintillation)", options: tr }, { text: "+77\u2013223% (fade mitigation)", options: tg }],
  [{ text: "Cost", options: { ...td, bold: true, align: "left" } }, { text: "$$$ (tens of k$)", options: tr }, { text: "$ (one-time fabrication)", options: tg }],
], {
  x: 0.6, y: 1.1, w: 12.1, colW: [2.5, 4.8, 4.8],
  border: { pt: 0.5, color: "E2E8F0" }, rowH: [0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42],
  autoPage: false,
});
s10.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 6.0, w: 12.1, h: 0.6, fill: { color: C.ice },
});
s10.addText("D2NN excels where MPLC struggles (deep turbulence / cost) and vice versa (moderate turbulence / accuracy)", {
  x: 0.8, y: 6.0, w: 11.7, h: 0.6,
  fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, valign: "middle", margin: 0,
});
addSlideNum(s10);

// ============================================================
// SLIDE 11: Terminology + Hybrid Proposal
// ============================================================
let s11 = pres.addSlide();
s11.background = { color: C.white };
s11.addText("Terminology and Hybrid Architecture", {
  x: 0.6, y: 0.35, w: 12, h: 0.6,
  fontSize: 28, fontFace: "Georgia", color: C.primary, bold: true, margin: 0,
});

// Left: Terminology
s11.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.1, w: 5.8, h: 3.0, fill: { color: C.card },
  shadow: makeShadow(),
});
s11.addShape(pres.shapes.RECTANGLE, {
  x: 0.6, y: 1.1, w: 0.08, h: 3.0, fill: { color: C.accent },
});
s11.addText("What to Call D2NN?", {
  x: 0.9, y: 1.2, w: 5.3, h: 0.4,
  fontSize: 18, fontFace: "Georgia", color: C.accent, bold: true, margin: 0,
});
s11.addText([
  { text: "\u2718 ", options: { color: C.red, bold: true } },
  { text: "\"Turbulence compensation\" (no WF correction)", options: { breakLine: true } },
  { text: "\u2718 ", options: { color: C.red, bold: true } },
  { text: "\"Adaptive optics\" (static, no feedback)", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "\u2714 ", options: { color: C.green, bold: true } },
  { text: "\"Turbulence-resilient passive beam shaping\"", options: { breakLine: true, bold: true } },
  { text: "\u2714 ", options: { color: C.green, bold: true } },
  { text: "\"Static diffractive fade mitigation\"", options: { breakLine: true, bold: true } },
  { text: "\u2714 ", options: { color: C.green, bold: true } },
  { text: "\"Ensemble-optimized spatial filter\"", options: { bold: true } },
], {
  x: 0.9, y: 1.7, w: 5.3, h: 2.2,
  fontSize: 14, fontFace: "Calibri", color: C.text, margin: 0,
});

// Right: Hybrid
s11.addShape(pres.shapes.RECTANGLE, {
  x: 6.9, y: 1.1, w: 5.8, h: 3.0, fill: { color: C.card },
  shadow: makeShadow(),
});
s11.addShape(pres.shapes.RECTANGLE, {
  x: 6.9, y: 1.1, w: 0.08, h: 3.0, fill: { color: C.green },
});
s11.addText("Proposed Hybrid: D2NN + MPLC", {
  x: 7.2, y: 1.2, w: 5.3, h: 0.4,
  fontSize: 18, fontFace: "Georgia", color: C.green, bold: true, margin: 0,
});
s11.addText([
  { text: "Stage 1 (passive): ", options: { bold: true, color: C.green } },
  { text: "D2NN reduces fade depth, stabilizes beam", options: { breakLine: true } },
  { text: "Stage 2 (active): ", options: { bold: true, color: C.primary } },
  { text: "MPLC mode correction on pre-stabilized beam", options: { breakLine: true } },
  { text: "", options: { breakLine: true, fontSize: 8 } },
  { text: "\u2022 D2NN reduces MPLC dynamic range requirement", options: { breakLine: true } },
  { text: "\u2022 Lower MPLC update rate needed", options: { breakLine: true } },
  { text: "\u2022 Graceful degradation if MPLC fails", options: { breakLine: true } },
  { text: "\u2022 Simpler (fewer-mode) MPLC sufficient", options: {} },
], {
  x: 7.2, y: 1.7, w: 5.3, h: 2.2,
  fontSize: 13, fontFace: "Calibri", color: C.text, margin: 0,
});

// Bottom: Summary stats
const stats = [
  { label: "D2NN Gain\n(weak)", value: "+2.8 dB", color: C.green },
  { label: "D2NN Gain\n(strong)", value: "+5.1 dB", color: C.green },
  { label: "Fade Floor\nImprovement", value: "+8.1 dB", color: C.accent },
  { label: "D2NN Power\nConsumption", value: "0 W", color: C.teal },
  { label: "D2NN\nCost", value: "< $100", color: C.amber },
];

stats.forEach((st, i) => {
  const sx = 0.6 + i * 2.5;
  s11.addShape(pres.shapes.RECTANGLE, {
    x: sx, y: 4.5, w: 2.2, h: 2.5, fill: { color: C.card },
    shadow: makeShadow(),
  });
  s11.addText(st.value, {
    x: sx, y: 4.7, w: 2.2, h: 1.2,
    fontSize: 32, fontFace: "Georgia", color: st.color, bold: true,
    align: "center", valign: "middle", margin: 0,
  });
  s11.addText(st.label, {
    x: sx, y: 5.9, w: 2.2, h: 0.8,
    fontSize: 12, fontFace: "Calibri", color: C.muted,
    align: "center", valign: "top", margin: 0,
  });
});
addSlideNum(s11);

// ============================================================
// SLIDE 12: Summary
// ============================================================
let s12 = pres.addSlide();
s12.background = { color: C.bg };
s12.addText("Key Takeaways", {
  x: 0.8, y: 0.4, w: 11.7, h: 0.8,
  fontSize: 36, fontFace: "Georgia", color: C.white, bold: true, margin: 0,
});

const takeaways = [
  { num: "1", text: "D2NN and MPLC share identical physics: cascaded phase masks + free-space propagation = unitary transformation", color: "93C5FD" },
  { num: "2", text: "MPLC decomposes into HG modes and corrects adaptively; D2NN applies a fixed, ensemble-optimized spatial filter", color: "93C5FD" },
  { num: "3", text: "D2NN is NOT wavefront correction \u2014 it's passive turbulence resilience via energy redistribution", color: "FCA5A5" },
  { num: "4", text: "D2NN advantage: zero power, $-cost, fabricate-once, effective in deep turbulence (+5.1 dB, +8 dB fade floor)", color: "86EFAC" },
  { num: "5", text: "MPLC advantage: true per-realization mode correction, full wavefront recovery, but complex and expensive", color: "93C5FD" },
  { num: "6", text: "Future: D2NN (passive pre-conditioning) + MPLC (active fine correction) = complementary hybrid architecture", color: "FDE68A" },
];

takeaways.forEach((t, i) => {
  const ty = 1.5 + i * 0.9;
  s12.addShape(pres.shapes.OVAL, {
    x: 0.8, y: ty, w: 0.5, h: 0.5,
    fill: { color: t.color },
  });
  s12.addText(t.num, {
    x: 0.8, y: ty, w: 0.5, h: 0.5,
    fontSize: 18, fontFace: "Georgia", color: C.bg, align: "center", valign: "middle", bold: true,
  });
  s12.addText(t.text, {
    x: 1.5, y: ty, w: 11.3, h: 0.5,
    fontSize: 15, fontFace: "Calibri", color: t.color, valign: "middle", margin: 0,
  });
});
addSlideNum(s12);

// Write
const outPath = "/root/dj/D2NN/kim2026/autoresearch/runs/0405-d2nn-vs-mplc-v2.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log("PPTX saved: " + outPath);
}).catch(err => console.error("Error:", err));
