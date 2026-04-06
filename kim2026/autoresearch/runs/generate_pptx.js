const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const IMG = "autoresearch/runs/ppt_white";
const OUT = "autoresearch/runs/0401-d2nn-artifact-analysis.pptx";

function imgData(name) {
  const buf = fs.readFileSync(path.join(IMG, name));
  return "image/png;base64," + buf.toString("base64");
}

// Colors
const C = {
  navy: "0f172a", blue: "1e40af", lightBlue: "dbeafe",
  red: "dc2626", lightRed: "fee2e2",
  green: "15803d", lightGreen: "dcfce7",
  amber: "d97706", lightAmber: "fef3c7",
  gray: "64748b", lightGray: "f1f5f9",
  dark: "1e293b", white: "ffffff", offWhite: "f8fafc",
  accent: "7c3aed",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Kim 2026";
pres.title = "D2NN Beam Cleanup: Data Pipeline Artifacts";

// ========== SLIDE 1: TITLE ==========
let s1 = pres.addSlide();
s1.background = { color: C.navy };
s1.addText("D2NN Beam Cleanup for FSO", {
  x: 0.8, y: 1.0, w: 8.4, h: 1.2,
  fontSize: 40, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});
s1.addText("Data Pipeline Artifacts & Physical Analysis", {
  x: 0.8, y: 2.2, w: 8.4, h: 0.7,
  fontSize: 22, fontFace: "Calibri", color: "94a3b8", margin: 0,
});
s1.addText("How numerical artifacts in beam reducer simulation\ncorrupted training data and masked the true D2NN performance", {
  x: 0.8, y: 3.2, w: 8.4, h: 0.9,
  fontSize: 15, fontFace: "Calibri", color: "64748b", margin: 0,
});
s1.addText("Kim 2026  ·  April 1, 2026", {
  x: 0.8, y: 4.5, w: 8.4, h: 0.5,
  fontSize: 14, fontFace: "Calibri", color: "475569", margin: 0,
});
s1.addShape(pres.shapes.RECTANGLE, { x: 0.8, y: 4.2, w: 2.0, h: 0.03, fill: { color: "3b82f6" } });

// ========== SLIDE 2: SYSTEM OVERVIEW ==========
let s2 = pres.addSlide();
s2.background = { color: C.white };
s2.addText("FSO Receiver with Static D2NN", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});

// System flow boxes
const boxes = [
  { label: "TX\n0.3mrad", color: "3b82f6", x: 0.3 },
  { label: "1km Atm\nCn²=5e-14", color: "64748b", x: 2.0 },
  { label: "Tel\n15cm", color: "22c55e", x: 3.7 },
  { label: "75:1\nReducer", color: "f59e0b", x: 5.4 },
  { label: "D2NN\n5 layers", color: "ef4444", x: 7.1 },
  { label: "f=4.5mm\n→ Det", color: "8b5cf6", x: 8.8 },
];
boxes.forEach((b, i) => {
  s2.addShape(pres.shapes.RECTANGLE, {
    x: b.x, y: 1.1, w: 1.4, h: 0.8,
    fill: { color: b.color, transparency: 80 },
    line: { color: b.color, width: 2 },
  });
  s2.addText(b.label, {
    x: b.x, y: 1.1, w: 1.4, h: 0.8,
    fontSize: 10, fontFace: "Calibri", bold: true, color: C.dark, align: "center", valign: "middle", margin: 0,
  });
  if (i < boxes.length - 1) {
    s2.addShape(pres.shapes.LINE, {
      x: b.x + 1.4, y: 1.5, w: boxes[i+1].x - b.x - 1.4, h: 0,
      line: { color: "94a3b8", width: 1.5 },
    });
  }
});

s2.addText("D/r₀ = 5.02  |  dx = 2μm  |  1024×1024  |  Phase-only  |  5.2M params", {
  x: 0.5, y: 2.05, w: 9.5, h: 0.35, fontSize: 12, fontFace: "Calibri", color: C.gray, italic: true, align: "center", margin: 0,
});

// Equations
s2.addText("Unitary Theorem — D2NN operator H conserves:", {
  x: 0.5, y: 2.6, w: 9, h: 0.4, fontSize: 16, fontFace: "Calibri", bold: true, color: C.dark, margin: 0,
});
s2.addImage({ data: imgData("eq_co.png"), x: 0.8, y: 3.0, w: 5.5, h: 0.6 });
s2.addImage({ data: imgData("eq_l2.png"), x: 0.8, y: 3.6, w: 5.0, h: 0.6 });

s2.addShape(pres.shapes.RECTANGLE, {
  x: 6.5, y: 2.9, w: 3.3, h: 1.5, fill: { color: C.lightRed },
  line: { color: C.red, width: 1 },
});
s2.addText("Cannot correct random\nturbulence with a\nstatic phase mask", {
  x: 6.5, y: 2.9, w: 3.3, h: 1.5,
  fontSize: 14, fontFace: "Calibri", bold: true, color: C.red, align: "center", valign: "middle", margin: 0,
});

s2.addText("Question: Can static D2NN improve focal-plane PIB through mode conversion?", {
  x: 0.5, y: 4.6, w: 9, h: 0.5, fontSize: 15, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

// ========== SLIDE 3: THE BUG ==========
let s3 = pres.addSlide();
s3.background = { color: C.white };
s3.addText("The Bug: Bilinear Beam Reducer", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.red, margin: 0,
});

// Wrong side
s3.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 1.1, w: 4.5, h: 3.8, fill: { color: C.lightRed }, line: { color: C.red, width: 1 } });
s3.addText("WRONG: Bilinear on Re/Im separately", {
  x: 0.5, y: 1.2, w: 4.1, h: 0.4, fontSize: 14, fontFace: "Calibri", bold: true, color: C.red, margin: 0,
});
s3.addImage({ data: imgData("eq_bilinear.png"), x: 0.5, y: 1.7, w: 4.0, h: 0.5 });
s3.addText([
  { text: "75:1 reduction (150mm → 2mm)", options: { bullet: true, breakLine: true } },
  { text: "cos(φ) and sin(φ) oscillate rapidly", options: { bullet: true, breakLine: true } },
  { text: "Separate interpolation → aliasing", options: { bullet: true, breakLine: true } },
  { text: "Like Moiré pattern: fake aberrations", options: { bullet: true } },
], { x: 0.5, y: 2.4, w: 4.1, h: 2.0, fontSize: 13, fontFace: "Calibri", color: C.dark, margin: 0 });
s3.addText("Vacuum WFE = 420nm (should be ~0)", {
  x: 0.5, y: 4.3, w: 4.1, h: 0.35, fontSize: 13, fontFace: "Calibri", bold: true, color: C.red, margin: 0,
});

// Right side
s3.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.5, h: 3.8, fill: { color: C.lightGreen }, line: { color: C.green, width: 1 } });
s3.addText("CORRECT: Complex Lanczos sinc", {
  x: 5.4, y: 1.2, w: 4.1, h: 0.4, fontSize: 14, fontFace: "Calibri", bold: true, color: C.green, margin: 0,
});
s3.addImage({ data: imgData("eq_lanczos.png"), x: 5.4, y: 1.7, w: 3.8, h: 0.5 });
s3.addText([
  { text: "Complex field directly (phase preserved)", options: { bullet: true, breakLine: true } },
  { text: "Optimal per Nyquist-Shannon theorem", options: { bullet: true, breakLine: true } },
  { text: "Windowed-sinc: no aliasing", options: { bullet: true, breakLine: true } },
  { text: "Also: defocus compensated + 2× pad", options: { bullet: true } },
], { x: 5.4, y: 2.4, w: 4.1, h: 2.0, fontSize: 13, fontFace: "Calibri", color: C.dark, margin: 0 });
s3.addText("Vacuum WFE = 3.4nm ✓", {
  x: 5.4, y: 4.3, w: 4.1, h: 0.35, fontSize: 13, fontFace: "Calibri", bold: true, color: C.green, margin: 0,
});

// ========== SLIDE 4: WFE ANALYSIS ==========
let s4 = pres.addSlide();
s4.background = { color: C.white };
s4.addText("Vacuum Wavefront Error Analysis", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});
s4.addImage({ data: imgData("fig2_wfe.png"), x: 0.3, y: 1.0, w: 9.4, h: 3.8 });
s4.addText("97.9% of vacuum WFE was higher-order artifact from bilinear interpolation — NOT physical", {
  x: 0.5, y: 4.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Calibri", bold: true, color: C.red, align: "center", margin: 0,
});

// ========== SLIDE 5: ARTIFACT COMPENSATION ==========
let s5 = pres.addSlide();
s5.background = { color: C.white };
s5.addText("D2NN Was Compensating Artifacts", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});
s5.addImage({ data: imgData("fig3_effect.png"), x: 0.2, y: 0.95, w: 9.6, h: 4.2 });
s5.addText("D2NN improved vacuum PIB from 16.5% → 99.0% (6.0×) — it was fixing data damage, not turbulence", {
  x: 0.3, y: 5.0, w: 9.4, h: 0.4, fontSize: 13, fontFace: "Calibri", bold: true, color: C.red, align: "center", margin: 0,
});

// ========== SLIDE 6: STREHL FAILURES ==========
let s6 = pres.addSlide();
s6.background = { color: C.white };
s6.addText("Why Strehl Ratio Fails for D2NN", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});
s6.addImage({ data: imgData("fig4_strehl.png"), x: 0.3, y: 0.9, w: 9.4, h: 3.2 });

// Cauchy-Schwarz box
s6.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.15, w: 9.0, h: 1.2, fill: { color: C.lightGray }, line: { color: "cbd5e1", width: 1 } });
s6.addText("Cauchy-Schwarz:", {
  x: 0.7, y: 4.2, w: 2.0, h: 0.35, fontSize: 13, fontFace: "Calibri", bold: true, color: C.dark, margin: 0,
});
s6.addImage({ data: imgData("eq_cauchy.png"), x: 0.7, y: 4.5, w: 5.5, h: 0.45 });
s6.addText("S ≤ 1 only for same amplitude.\nD2NN changes amplitude → no valid reference.\nUse PIB + WF RMS instead.", {
  x: 6.3, y: 4.2, w: 3.0, h: 1.1, fontSize: 12, fontFace: "Calibri", bold: true, color: C.accent, margin: 0,
});

// ========== SLIDE 7: OLD vs NEW ==========
let s7 = pres.addSlide();
s7.background = { color: C.white };
s7.addText("Old Data vs New Data: Complete Comparison", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});
s7.addImage({ data: imgData("fig1_old_vs_new_pib.png"), x: 0.3, y: 0.95, w: 9.4, h: 4.3 });

// ========== SLIDE 8: CLEAN BASELINE ==========
let s8 = pres.addSlide();
s8.background = { color: C.white };
s8.addText("Clean Data Baseline", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});
s8.addImage({ data: imgData("fig5_baseline.png"), x: 0.3, y: 0.95, w: 9.4, h: 3.8 });

s8.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 4.85, w: 7.0, h: 0.6, fill: { color: C.lightAmber }, line: { color: C.amber, width: 1 } });
s8.addText("D2NN headroom: Can it recover the 15.5%p turbulence gap? (80% → 95.5%)", {
  x: 1.5, y: 4.85, w: 7.0, h: 0.6, fontSize: 15, fontFace: "Calibri", bold: true, color: "92400e", align: "center", valign: "middle", margin: 0,
});

// ========== SLIDE 9: MODE CONVERSION ==========
let s9 = pres.addSlide();
s9.background = { color: C.white };
s9.addText("D2NN Mode Conversion Physics", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.dark, margin: 0,
});

s9.addImage({ data: imgData("fig6_modes.png"), x: 0.3, y: 0.9, w: 9.4, h: 2.8 });

// Equations
s9.addImage({ data: imgData("eq_modes.png"), x: 0.5, y: 3.7, w: 3.5, h: 0.45 });
s9.addImage({ data: imgData("eq_mode_convert.png"), x: 0.5, y: 4.15, w: 5.0, h: 0.45 });

// Cannot / Can boxes
s9.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 3.65, w: 4.3, h: 0.9, fill: { color: C.lightRed }, line: { color: C.red, width: 1 } });
s9.addText("Cannot: correct random WF, improve CO", {
  x: 5.6, y: 3.7, w: 4.1, h: 0.8, fontSize: 12, fontFace: "Calibri", bold: true, color: C.red, valign: "middle", margin: 0,
});

s9.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 4.6, w: 4.3, h: 0.9, fill: { color: C.lightGreen }, line: { color: C.green, width: 1 } });
s9.addText("Can: redistribute modal energy\nPassive, zero-latency, zero-power", {
  x: 5.6, y: 4.6, w: 4.1, h: 0.9, fontSize: 12, fontFace: "Calibri", bold: true, color: C.green, valign: "middle", margin: 0,
});

// ========== SLIDE 10: LESSONS ==========
let s10 = pres.addSlide();
s10.background = { color: C.navy };
s10.addText("Lessons & Next Steps", {
  x: 0.5, y: 0.3, w: 9, h: 0.6, fontSize: 28, fontFace: "Georgia", bold: true, color: C.white, margin: 0,
});

// Lessons (left)
s10.addShape(pres.shapes.RECTANGLE, { x: 0.3, y: 1.0, w: 4.5, h: 4.2, fill: { color: "1e293b" }, line: { color: "334155", width: 1 } });
s10.addText("Numerical Pitfalls", {
  x: 0.5, y: 1.1, w: 4.1, h: 0.4, fontSize: 16, fontFace: "Calibri", bold: true, color: "f87171", margin: 0,
});
s10.addText([
  { text: "Never separate Re/Im for interpolation", options: { bullet: true, breakLine: true } },
  { text: "PSF sampling: Nyquist ≥ 2px/Airy", options: { bullet: true, breakLine: true } },
  { text: "FFT propagation: zero-pad for anti-alias", options: { bullet: true, breakLine: true } },
  { text: "Always validate vacuum beam first", options: { bullet: true, breakLine: true } },
  { text: "Strehl needs identical amplitude", options: { bullet: true, breakLine: true } },
  { text: "Verify beam-lens matching before D2NN", options: { bullet: true } },
], { x: 0.5, y: 1.6, w: 4.1, h: 3.3, fontSize: 13, fontFace: "Calibri", color: "e2e8f0", margin: 0 });

// Next steps (right)
s10.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.5, h: 4.2, fill: { color: "1e293b" }, line: { color: "334155", width: 1 } });
s10.addText("Experimental Plan", {
  x: 5.4, y: 1.1, w: 4.1, h: 0.4, fontSize: 16, fontFace: "Calibri", bold: true, color: "4ade80", margin: 0,
});
s10.addText([
  { text: "Train D2NN on clean data (5000 samples)", options: { bullet: true, breakLine: true } },
  { text: "True test: PIB from 80% → 95%?", options: { bullet: true, breakLine: true } },
  { text: "Layer sweep: 1, 3, 5, 7, 10 layers", options: { bullet: true, breakLine: true } },
  { text: "Turbulence sweep: Cn²=1e-14 to 1e-13", options: { bullet: true, breakLine: true } },
  { text: "Validate Lanczos vs physical reducer", options: { bullet: true, breakLine: true } },
  { text: "Is passive PIB recovery achievable?", options: { bullet: true } },
], { x: 5.4, y: 1.6, w: 4.1, h: 3.3, fontSize: 13, fontFace: "Calibri", color: "e2e8f0", margin: 0 });

// Save
pres.writeFile({ fileName: OUT }).then(() => {
  console.log("PPTX saved to: " + OUT);
});
