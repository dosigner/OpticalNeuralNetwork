const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const SHAPES = new pptxgen().shapes;

const RUNS_DIR = __dirname;
const IMG_DIR = process.env.HYBRID_PPT_IMG_DIR
  ? path.resolve(process.env.HYBRID_PPT_IMG_DIR)
  : path.join(RUNS_DIR, "ppt_white_hybrid");
const OUT = process.env.HYBRID_PPT_OUT
  ? path.resolve(process.env.HYBRID_PPT_OUT)
  : path.join(RUNS_DIR, "0401-d2nn-hybrid-analysis.pptx");

const C = {
  white: "FFFFFF",
  offWhite: "F8FAFC",
  ink: "0F172A",
  slate: "334155",
  gray: "64748B",
  lightGray: "E2E8F0",
  border: "CBD5E1",
  blue: "1D4ED8",
  lightBlue: "DBEAFE",
  green: "15803D",
  lightGreen: "DCFCE7",
  red: "B91C1C",
  lightRed: "FEE2E2",
  amber: "B45309",
  lightAmber: "FEF3C7",
  purple: "7C3AED",
  lightPurple: "EDE9FE",
};

const DECK_DATA = {
  title: "D2NN Beam Cleanup for FSO",
  subtitle: "Run chronology, artifact analysis, and physics hardening",
  chronology: [
    {
      date: "0325",
      label: "Spatial-domain sweep",
      problem: "CO-only loss left phase/intensity concentration ambiguous.",
      fix: "Screen amplitude / phasor / full-field-phase / ROI auxiliary losses.",
    },
    {
      date: "0327",
      label: "Detector metric shift",
      problem: "Output-plane CO diverged from PIB / Strehl objectives.",
      fix: "Switch losses to PIB, Strehl, IO, and CO+PIB hybrid.",
    },
    {
      date: "0328",
      label: "Strong turbulence",
      problem: "Weak-condition trend might not survive Cn²=5e-14.",
      fix: "Repeat the same family under strong turbulence with 5-layer D2NN.",
    },
    {
      date: "0330",
      label: "Focal-plane objective",
      problem: "Pre-lens metrics did not match the detector plane.",
      fix: "Move PIB / Strehl optimization to the focal plane after the lens.",
    },
    {
      date: "0401",
      label: "Physics hardening",
      problem: "Legacy propagation and Strehl definitions distorted interpretation.",
      fix: "Add padded propagation, corrected Strehl, and post-train sanity gates.",
    },
  ],
  run0325: {
    baselineCO: 0.6456,
    baselineEE: 0.8399,
    coFFP_CO: 0.6407,
    coFFP_Strehl: 6.4718,
    roi80_CO: 0.5457,
    roi80_EE: 0.9880,
  },
  run0327: {
    pibOnlyPIB50: 0.8340,
    pibOnlyCO: 0.0163,
    hybridPIB50: 0.5082,
    hybridCO: 0.2938,
    ioOnlyIO: 0.8189,
    ioOnlyPIB50: 0.0024,
  },
  run0328: {
    baselineCO: 0.3286,
    baselineIO: 0.6843,
    coFFP_CO: 0.3295,
    coFFP_Strehl: 12.9250,
    roi80_EE: 0.9905,
    roi80_CO: 0.2733,
  },
  run0330: {
    outCOBefore: 0.2799,
    outCOAfter: 0.0979,
    pib10Before: 0.1722,
    pib10After: 0.8128,
    pib25After: 0.9651,
    strehlCorrect: 0.5464,
  },
  run0401: {
    paddedPIB10: 0.9001,
    paddedStrehl: 0.6384,
    throughput: 0.4381,
    coDelta: 0.0896,
    strehlOnlyThroughput: 0.0218,
    strehlOnlyStrehl: 1.2014,
  },
  artifacts: {
    oldVacuumWfeNm: 420.0,
    newVacuumWfeNm: 3.4,
    oldVacuumPIB10: 16.5,
    newVacuumPIB10: 95.5,
    oldD2nnPIB10: 90.0,
    newTurbPIB10: 80.0,
    headroom: 15.5,
    defocusShare: 7.5,
    higherOrderShare: 97.9,
  },
};

function ensureDir(targetPath) {
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
}

function imgData(name) {
  const fullPath = path.join(IMG_DIR, name);
  const buf = fs.readFileSync(fullPath);
  return `data:image/${path.extname(name).slice(1)};base64,${buf.toString("base64")}`;
}

function escapeXml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function svgTextDataUri(lines, options = {}) {
  const width = options.width || 1200;
  const fontSize = options.fontSize || 34;
  const lineHeight = options.lineHeight || Math.round(fontSize * 1.45);
  const paddingX = options.paddingX || 24;
  const paddingY = options.paddingY || 22;
  const color = options.color || "#0F172A";
  const height =
    options.height ||
    paddingY * 2 + lineHeight * lines.length - Math.round(fontSize * 0.35);
  const fontFamily =
    options.fontFamily || "Asana Math, Latin Modern Math, DejaVu Serif, serif";

  const textNodes = lines
    .map(
      (line, idx) =>
        `<text x="${paddingX}" y="${paddingY + idx * lineHeight + fontSize}" font-family="${fontFamily}" font-size="${fontSize}" fill="${color}">${escapeXml(
          line
        )}</text>`
    )
    .join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${textNodes}</svg>`;
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
}

function bulletRuns(items) {
  return items.map((text, idx) => ({
    text,
    options: {
      bullet: true,
      breakLine: idx !== items.length - 1,
    },
  }));
}

function createDeck() {
  const deck = new pptxgen();
  deck.layout = "LAYOUT_WIDE";
  deck.theme = {
    headFontFace: "DejaVu Serif",
    bodyFontFace: "DejaVu Sans",
    lang: "en-US",
  };
  deck.author = "Codex";
  deck.company = "OpenAI";
  deck.subject = "D2NN chronology, artifact analysis, and physics hardening";
  deck.title = "D2NN Beam Cleanup for FSO: Hybrid Analysis";
  deck.lang = "en-US";
  return deck;
}

function addPageFrame(slide, title, subtitle = "", accent = C.blue, footer = "") {
  slide.background = { color: C.white };
  slide.addText(title, {
    x: 0.45,
    y: 0.22,
    w: 8.2,
    h: 0.46,
    fontFace: "DejaVu Serif",
    fontSize: 26,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  slide.addShape(SHAPES.RECTANGLE, {
    x: 0.45,
    y: 0.74,
    w: 1.25,
    h: 0.035,
    line: { color: accent, transparency: 100 },
    fill: { color: accent },
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 8.85,
      y: 0.31,
      w: 3.8,
      h: 0.18,
      fontFace: "DejaVu Sans",
      fontSize: 10,
      italic: true,
      color: C.gray,
      margin: 0,
      align: "right",
    });
  }
  slide.addShape(SHAPES.LINE, {
    x: 0.45,
    y: 0.83,
    w: 12.2,
    h: 0,
    line: { color: C.border, width: 1.0 },
  });
  if (footer) {
    slide.addShape(SHAPES.LINE, {
      x: 0.45,
      y: 7.02,
      w: 12.2,
      h: 0,
      line: { color: C.border, width: 0.8 },
    });
    slide.addText(footer, {
      x: 0.45,
      y: 7.08,
      w: 12.2,
      h: 0.18,
      fontFace: "DejaVu Sans",
      fontSize: 8,
      color: C.gray,
      margin: 0,
    });
  }
}

function addCallout(slide, { x, y, w, h, title, body, fill, line, titleColor, bodyColor }) {
  slide.addShape(SHAPES.ROUNDED_RECTANGLE, {
    x,
    y,
    w,
    h,
    fill: { color: fill || C.offWhite },
    line: { color: line || C.border, width: 1.1 },
    radius: 0.08,
  });
  slide.addText(title, {
    x: x + 0.16,
    y: y + 0.12,
    w: w - 0.32,
    h: 0.22,
    fontFace: "DejaVu Sans",
    fontSize: 13,
    bold: true,
    color: titleColor || C.ink,
    margin: 0,
  });
  if (body) {
    slide.addText(body, {
      x: x + 0.16,
      y: y + 0.42,
      w: w - 0.32,
      h: h - 0.54,
      fontFace: "DejaVu Sans",
      fontSize: 11,
      color: bodyColor || C.slate,
      valign: "top",
      margin: 0,
    });
  }
}

function addMetricRow(slide, x, y, w, label, value, color, scale = 1.0) {
  const barW = 1.95;
  slide.addText(label, {
    x,
    y,
    w: 1.55,
    h: 0.18,
    fontFace: "DejaVu Sans",
    fontSize: 11,
    color: C.slate,
    margin: 0,
  });
  slide.addShape(SHAPES.RECTANGLE, {
    x: x + 1.7,
    y: y + 0.03,
    w: barW,
    h: 0.12,
    fill: { color: C.lightGray },
    line: { color: C.lightGray, transparency: 100 },
  });
  slide.addShape(SHAPES.RECTANGLE, {
    x: x + 1.7,
    y: y + 0.03,
    w: Math.max(0.02, Math.min(barW, barW * (value / scale))),
    h: 0.12,
    fill: { color },
    line: { color, transparency: 100 },
  });
  slide.addText(String(value), {
    x: x + 3.9,
    y: y - 0.03,
    w: w - 3.9,
    h: 0.22,
    fontFace: "DejaVu Sans",
    fontSize: 11,
    bold: true,
    color: C.ink,
    align: "right",
    margin: 0,
  });
}

function addEquationPanel(slide, x, y, w, h, title, lines, accent = C.purple) {
  slide.addShape(SHAPES.ROUNDED_RECTANGLE, {
    x,
    y,
    w,
    h,
    fill: { color: C.offWhite },
    line: { color: accent, width: 1.0 },
    radius: 0.08,
  });
  slide.addText(title, {
    x: x + 0.18,
    y: y + 0.12,
    w: w - 0.36,
    h: 0.2,
    fontFace: "DejaVu Sans",
    fontSize: 12,
    bold: true,
    color: accent,
    margin: 0,
  });
  slide.addImage({
    data: svgTextDataUri(lines, {
      width: 1000,
      fontSize: 30,
      lineHeight: 40,
      color: "#0F172A",
    }),
    x: x + 0.15,
    y: y + 0.42,
    w: w - 0.30,
    h: h - 0.55,
  });
}

function addTwoColumnBullets(
  slide,
  leftTitle,
  leftBullets,
  rightTitle,
  rightBullets,
  options = {}
) {
  const y = options.y || 1.1;
  const h = options.h || 2.0;
  addCallout(slide, {
    x: 0.55,
    y,
    w: 5.9,
    h,
    title: leftTitle,
    body: "",
    fill: C.offWhite,
    line: C.border,
  });
  slide.addText(bulletRuns(leftBullets), {
    x: 0.72,
    y: y + 0.38,
    w: 5.55,
    h: h - 0.5,
    fontFace: "DejaVu Sans",
    fontSize: 11,
    color: C.slate,
    margin: 0,
    breakLine: false,
  });
  addCallout(slide, {
    x: 6.78,
    y,
    w: 5.95,
    h,
    title: rightTitle,
    body: "",
    fill: C.offWhite,
    line: C.border,
  });
  slide.addText(bulletRuns(rightBullets), {
    x: 6.95,
    y: y + 0.38,
    w: 5.55,
    h: h - 0.5,
    fontFace: "DejaVu Sans",
    fontSize: 11,
    color: C.slate,
    margin: 0,
  });
}

function finalizeSlide(slide, deck) {
  warnIfSlideHasOverlaps(slide, deck, {
    muteContainment: true,
    ignoreLines: true,
  });
  warnIfSlideElementsOutOfBounds(slide, deck);
}

function buildSlides(deck) {
  const s1 = deck.addSlide();
  s1.background = { color: C.white };
  s1.addShape(deck.shapes.RECTANGLE, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.6,
    fill: { color: C.blue },
    line: { color: C.blue, transparency: 100 },
  });
  s1.addText(DECK_DATA.title, {
    x: 0.7,
    y: 1.1,
    w: 10.8,
    h: 0.9,
    fontFace: "DejaVu Serif",
    fontSize: 30,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  s1.addText("Data pipeline artifacts, run chronology, and physical interpretation", {
    x: 0.72,
    y: 2.08,
    w: 9.4,
    h: 0.4,
    fontFace: "DejaVu Sans",
    fontSize: 17,
    color: C.slate,
    margin: 0,
  });
  s1.addText(
    "From 0325 auxiliary-loss sweeps to 0401 physics hardening: what each run failed at, what it tried to fix, and which claims remained valid",
    {
      x: 0.72,
      y: 2.72,
      w: 10.6,
      h: 0.8,
      fontFace: "DejaVu Sans",
      fontSize: 15,
      color: C.gray,
      margin: 0,
    }
  );
  addCallout(s1, {
    x: 0.78,
    y: 4.1,
    w: 5.7,
    h: 1.45,
    title: "Deck logic",
    body:
      "Part I reconstructs the problem-fix chronology across 0325/0327/0328/0330/0401. Part II explains why several apparently strong results were really data or metric artifacts.",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s1, {
    x: 6.85,
    y: 4.1,
    w: 5.7,
    h: 1.45,
    title: "Primary evidence",
    body:
      "summary.json, focal sweep logs, physics reassessment notes, and the static artifact-analysis figures already generated on 2026-04-01.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  s1.addText("Prepared from /root/dj/D2NN/kim2026/autoresearch/runs · April 1, 2026", {
    x: 0.75,
    y: 6.84,
    w: 12,
    h: 0.2,
    fontFace: "DejaVu Sans",
    fontSize: 10,
    color: C.gray,
    margin: 0,
  });
  finalizeSlide(s1, deck);

  const s2 = deck.addSlide();
  addPageFrame(
    s2,
    "Executive Summary",
    "Internal review deck",
    C.blue,
    "Sources: 0325/0327/0328 summary.json, 0330 reassessment note, 0401 focal sweep log"
  );
  addCallout(s2, {
    x: 0.62,
    y: 1.05,
    w: 6.0,
    h: 1.18,
    title: "1. The problem changed over time",
    body:
      "The project moved from output-plane CO matching, to detector-centric PIB/Strehl objectives, and finally to a hardened physics stack with explicit sanity checks.",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s2, {
    x: 6.76,
    y: 1.05,
    w: 5.92,
    h: 1.18,
    title: "2. Several early wins were real but incomplete",
    body:
      "0325 and 0328 exposed genuine trade-offs, but they still optimized the wrong plane for focal detection and over-trusted Strehl-like summaries.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  addCallout(s2, {
    x: 0.62,
    y: 2.5,
    w: 6.0,
    h: 1.18,
    title: "3. 0330 fixed the objective, not the full physics",
    body:
      "Moving PIB/Strehl to the focal plane was the right conceptual step, but the run still used the old propagation and legacy Strehl path, so it had to be reinterpreted.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addCallout(s2, {
    x: 6.76,
    y: 2.5,
    w: 5.92,
    h: 1.18,
    title: "4. 0401 hardened the stack and exposed new failures",
    body:
      "Padded propagation and corrected Strehl removed one class of artifact, but severe throughput loss and unitary-consistency failures showed the system was still not physically trustworthy.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addEquationPanel(s2, 0.68, 4.1, 5.8, 1.6, "Key narrative equation", [
    "Wrong objective  +  wrong physics  →  misleadingly strong conclusions",
    "Better objective +  better physics →  smaller but more honest headroom",
  ]);
  addEquationPanel(s2, 6.82, 4.1, 5.8, 1.6, "The operational question", [
    "Can a passive, static D2NN recover PIB headroom without violating physics?",
    "The rest of the deck explains why the answer became narrower over time.",
  ], C.green);
  finalizeSlide(s2, deck);

  const s3 = deck.addSlide();
  addPageFrame(
    s3,
    "System and Unitary Constraints",
    "Why output-plane CO and detector-plane PIB are not interchangeable",
    C.purple,
    "System: TX → 1 km atmosphere → 15 cm telescope → beam reducer → D2NN → f=4.5 mm lens → detector"
  );
  const boxes = [
    { label: "TX\n0.3 mrad", color: C.blue, x: 0.55 },
    { label: "1 km\nCn²=5e-14", color: C.gray, x: 2.45 },
    { label: "Telescope\n15 cm", color: C.green, x: 4.45 },
    { label: "Reducer\n75:1", color: C.amber, x: 6.45 },
    { label: "D2NN\n5 layers", color: C.red, x: 8.45 },
    { label: "Lens +\nDetector", color: C.purple, x: 10.45 },
  ];
  boxes.forEach((box, idx) => {
    s3.addShape(deck.shapes.ROUNDED_RECTANGLE, {
      x: box.x,
      y: 1.35,
      w: 1.55,
      h: 0.82,
      fill: { color: "FFFFFF" },
      line: { color: box.color, width: 2 },
      radius: 0.06,
    });
    s3.addText(box.label, {
      x: box.x,
      y: 1.42,
      w: 1.55,
      h: 0.66,
      fontFace: "DejaVu Sans",
      fontSize: 11,
      bold: true,
      color: C.ink,
      align: "center",
      valign: "middle",
      margin: 0,
    });
    if (idx < boxes.length - 1) {
      s3.addShape(deck.shapes.LINE, {
        x: box.x + 1.55,
        y: 1.76,
        w: boxes[idx + 1].x - box.x - 1.55,
        h: 0,
        line: { color: C.border, width: 1.2 },
      });
    }
  });
  s3.addImage({ data: imgData("eq_co.png"), x: 0.78, y: 2.65, w: 5.45, h: 0.48 });
  s3.addImage({ data: imgData("eq_l2.png"), x: 0.78, y: 3.22, w: 4.75, h: 0.48 });
  addCallout(s3, {
    x: 6.7,
    y: 2.58,
    w: 5.55,
    h: 1.55,
    title: "Interpretation",
    body:
      "A static phase-only D2NN is unitary at the output plane. It can redistribute modal energy, but it cannot genuinely erase random turbulence in a single-shot field-by-field sense.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addTwoColumnBullets(
    s3,
    "What remains meaningful",
    [
      "Output-plane CO / WF RMS for theorem checks",
      "Detector-plane PIB for coupling-oriented concentration",
      "Energy throughput and passive-limit sanity checks",
    ],
    "What caused confusion later",
    [
      "Optimizing the wrong plane for the intended detector",
      "Treating Strehl as valid after amplitude-changing optics",
      "Believing strong PIB gains without asking what was being fixed",
    ],
    { y: 4.45, h: 1.9 }
  );
  finalizeSlide(s3, deck);

  const s4 = deck.addSlide();
  addPageFrame(
    s4,
    "Research Chronology Map",
    "Problem → attempted fix → new failure",
    C.green,
    "0329 acted mainly as packaging: theorem / pareto / loss-strategy figures for paper narrative"
  );
  s4.addShape(deck.shapes.LINE, {
    x: 1.0,
    y: 3.45,
    w: 11.1,
    h: 0,
    line: { color: C.border, width: 2 },
  });
  DECK_DATA.chronology.forEach((item, idx) => {
    const x = 0.72 + idx * 2.45;
    s4.addShape(deck.shapes.OVAL, {
      x,
      y: 3.03,
      w: 0.78,
      h: 0.78,
      fill: { color: idx === 4 ? C.lightRed : C.lightBlue },
      line: { color: idx === 4 ? C.red : C.blue, width: 1.5 },
    });
    s4.addText(item.date, {
      x,
      y: 3.18,
      w: 0.78,
      h: 0.3,
      fontFace: "DejaVu Sans",
      fontSize: 11,
      bold: true,
      color: C.ink,
      align: "center",
      margin: 0,
    });
    addCallout(s4, {
      x: x - 0.22,
      y: idx % 2 === 0 ? 1.1 : 4.1,
      w: 1.92,
      h: 1.7,
      title: item.label,
      body: `Problem: ${item.problem}\nFix: ${item.fix}`,
      fill: idx === 4 ? C.lightRed : C.offWhite,
      line: idx === 4 ? C.red : C.border,
      titleColor: idx === 4 ? C.red : C.ink,
    });
  });
  addEquationPanel(s4, 0.78, 6.15, 11.85, 0.62, "Chronology takeaway", [
    "Each run fixed one failure mode but simultaneously revealed a deeper mismatch in metric plane or physical assumptions.",
  ], C.amber);
  finalizeSlide(s4, deck);

  const s5 = deck.addSlide();
  addPageFrame(
    s5,
    "0325: Spatial-Domain Loss Screening",
    "First attempt to move beyond CO-only supervision",
    C.blue,
    "Run: 0325-telescope-sweep-cn2-5e14-15cm"
  );
  addTwoColumnBullets(
    s5,
    "Problem observed",
    [
      "Complex-overlap optimization alone did not tell you whether energy landed in the detector-friendly core.",
      "Phase quality, Strehl-like sharpness, and encircled energy moved differently.",
      "The sweep needed to expose which auxiliary term best aligned with concentration.",
    ],
    "Attempted fixes",
    [
      "co_amp: amplitude regularization on top of CO",
      "co_phasor / co_ffp: stronger phase-aware terms",
      "roi80: ROI leakage penalty to force concentration",
    ]
  );
  addCallout(s5, {
    x: 0.66,
    y: 3.45,
    w: 5.8,
    h: 2.2,
    title: "What the sweep taught",
    body:
      "No auxiliary loss dominated every metric. co_ffp preserved CO while pushing Strehl upward; roi80 maximized encircled energy (0.9880) but sacrificed overlap (0.5457).",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s5, {
    x: 6.78,
    y: 3.45,
    w: 5.84,
    h: 2.2,
    title: "Interpretation",
    body:
      "This run established the basic trade-off surface: output-plane fidelity and detector-style concentration were not the same optimization problem.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addCallout(s5, {
    x: 0.72,
    y: 5.9,
    w: 3.85,
    h: 0.86,
    title: "baseline_co",
    body: `CO ${DECK_DATA.run0325.baselineCO.toFixed(4)}  |  EE ${DECK_DATA.run0325.baselineEE.toFixed(4)}`,
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  addCallout(s5, {
    x: 4.76,
    y: 5.9,
    w: 3.85,
    h: 0.86,
    title: "co_ffp",
    body: `CO ${DECK_DATA.run0325.coFFP_CO.toFixed(4)}  |  Strehl ${DECK_DATA.run0325.coFFP_Strehl.toFixed(4)}`,
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  addCallout(s5, {
    x: 8.8,
    y: 5.9,
    w: 3.1,
    h: 0.86,
    title: "roi80",
    body: `CO ${DECK_DATA.run0325.roi80_CO.toFixed(4)}  |  EE ${DECK_DATA.run0325.roi80_EE.toFixed(4)}`,
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  finalizeSlide(s5, deck);

  const s6 = deck.addSlide();
  addPageFrame(
    s6,
    "0327: Shift to PIB / Strehl / IO Objectives",
    "The detector objective forced a new loss family",
    C.purple,
    "Run: 0327-loss-sweep-prelens-pib-cn2-5e14"
  );
  addTwoColumnBullets(
    s6,
    "Problem observed",
    [
      "Even the best CO-family runs did not directly optimize bucket power or focal sharpness.",
      "Detector success had to be expressed in intensity-space metrics instead of output-plane field similarity alone.",
      "The question became: which detector-centric loss breaks least badly?",
    ],
    "Attempted fixes",
    [
      "pib_only: direct bucket-power maximization",
      "strehl_only: peak sharpness objective",
      "intensity_overlap and co_pib_hybrid as compromise objectives",
    ]
  );
  addCallout(s6, {
    x: 0.68,
    y: 3.48,
    w: 4.0,
    h: 2.05,
    title: "PIB-only",
    body: "PIB50 = 0.8340\nCO = 0.0163\nStrong detector concentration but catastrophic field overlap collapse.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s6, {
    x: 4.94,
    y: 3.48,
    w: 4.0,
    h: 2.05,
    title: "CO + PIB hybrid",
    body: "PIB50 = 0.5082\nCO = 0.2938\nA workable compromise: worse bucket power, much less destructive to overlap.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addCallout(s6, {
    x: 9.2,
    y: 3.48,
    w: 3.5,
    h: 2.05,
    title: "IO-only",
    body: "IO = 0.8189\nPIB50 = 0.0024\nIntensity similarity alone did not produce useful concentration.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addEquationPanel(s6, 0.72, 5.95, 12.0, 0.72, "Main lesson", [
    "Detector-centric metrics exposed a new trade-off: maximizing PIB could destroy the very field structure that CO was tracking.",
  ], C.purple);
  finalizeSlide(s6, deck);

  const s7 = deck.addSlide();
  addPageFrame(
    s7,
    "0328: Strong-Turbulence Re-test",
    "Did the same trade-offs survive Cn² = 5e-14?",
    C.green,
    "Run: 0328-co-sweep-strong-turb-cn2-5e14"
  );
  addTwoColumnBullets(
    s7,
    "Problem observed",
    [
      "Weak-condition intuition might fail under stronger turbulence and larger phase variance.",
      "A loss that looked attractive on milder data could collapse under D/r₀ ≈ 5.",
      "The sweep needed to confirm whether any 'winner' persisted.",
    ],
    "Attempted fix",
    [
      "Re-run the 0325 auxiliary-loss family on the strong-turbulence dataset.",
      "Keep architecture fixed at 5 layers so the comparison stays about objectives, not capacity.",
      "Track CO, IO, Strehl, and encircled energy together.",
    ]
  );
  addCallout(s7, {
    x: 0.72,
    y: 3.46,
    w: 3.88,
    h: 2.0,
    title: "baseline_co / co_amp",
    body:
      "CO ≈ 0.3286 with IO ≈ 0.684. Good overlap preservation, limited concentration gain.",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s7, {
    x: 4.74,
    y: 3.46,
    w: 3.88,
    h: 2.0,
    title: "co_ffp",
    body:
      "CO = 0.3295 with Strehl = 12.925. Best sharpness among the family, but IO dropped to 0.607.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  addCallout(s7, {
    x: 8.76,
    y: 3.46,
    w: 3.88,
    h: 2.0,
    title: "roi80",
    body:
      "Encircled energy = 0.9905, but CO fell to 0.2733. Detector concentration still opposed field fidelity.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addEquationPanel(s7, 0.74, 5.92, 11.96, 0.75, "Takeaway", [
    "0328 confirmed the same structure as 0325: there was still no single objective that optimized CO, intensity similarity, and concentration together.",
  ], C.green);
  finalizeSlide(s7, deck);

  const s8 = deck.addSlide();
  addPageFrame(
    s8,
    "0330: Focal-Plane Objective Redesign",
    "The conceptual move from output-plane metrics to detector-plane metrics",
    C.amber,
    "Run: 0330-focal-pib-sweep-4loss-cn2-5e14/focal_pib_only"
  );
  addEquationPanel(s8, 0.64, 1.06, 6.0, 1.95, "Metric-plane separation", [
    "CO, WF RMS   →   D2NN output plane",
    "PIB, Strehl  →   focal plane after the f = 4.5 mm lens",
  ], C.amber);
  addEquationPanel(s8, 6.85, 1.06, 5.82, 1.95, "Representative loss", [
    "Lₚᵢᵦ = 1 − PIB₁₀μm(focal)",
    "Lₕᵧᵦᵣᵢd = (1 − CO_out) + 0.5(1 − PIB₁₀μm,focal)",
  ], C.purple);
  addCallout(s8, {
    x: 0.7,
    y: 3.42,
    w: 3.82,
    h: 2.0,
    title: "Problem solved",
    body:
      "The detector objective was finally optimized at the detector plane instead of being approximated from the D2NN output plane.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addCallout(s8, {
    x: 4.74,
    y: 3.42,
    w: 3.82,
    h: 2.0,
    title: "What looked strong",
    body:
      `PIB@10μm: ${DECK_DATA.run0330.pib10Before.toFixed(4)} → ${DECK_DATA.run0330.pib10After.toFixed(4)}\nPIB@25μm: ${DECK_DATA.run0330.pib25After.toFixed(4)}\nCorrected Strehl (recomputed): ${DECK_DATA.run0330.strehlCorrect.toFixed(4)}`,
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s8, {
    x: 8.78,
    y: 3.42,
    w: 3.82,
    h: 2.0,
    title: "What remained wrong",
    body:
      "The checkpoint was still trained under the old propagation / legacy Strehl stack, so the result was useful as evidence of direction, not as a final physics-clean conclusion.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addEquationPanel(s8, 0.74, 5.9, 11.92, 0.78, "Takeaway", [
    "0330 fixed the objective mismatch, but later notes made it explicit that it did not yet fix the forward-physics mismatch.",
  ], C.amber);
  finalizeSlide(s8, deck);

  const s9 = deck.addSlide();
  addPageFrame(
    s9,
    "Physics Hardening and Residual Failures",
    "0401 added better guards, then exposed deeper non-passive behavior",
    C.red,
    "Evidence: 0401-focal-sweep.log and 0401-focal-pib-sweep-padded-4loss-cn2-5e14/results.json"
  );
  addTwoColumnBullets(
    s9,
    "Hardening steps",
    [
      "propagation_pad_factor = 2",
      "corrected Strehl with pad factor 4",
      "post-train sanity gates for throughput, unitary CO preservation, and Strehl upper bound",
    ],
    "Why this mattered",
    [
      "The old periodic same-window propagation was no longer trusted.",
      "Strehl had to be redefined against a physically meaningful reference.",
      "Strong-looking PIB gains now had to survive explicit passivity checks.",
    ]
  );
  addCallout(s9, {
    x: 0.72,
    y: 3.48,
    w: 3.86,
    h: 2.0,
    title: "focal_pib_only",
    body:
      `PIB@10μm = ${DECK_DATA.run0401.paddedPIB10.toFixed(4)}\nThroughput = ${DECK_DATA.run0401.throughput.toFixed(4)}\nCO delta = ${DECK_DATA.run0401.coDelta.toFixed(4)}\nSanity result: FAIL`,
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addCallout(s9, {
    x: 4.75,
    y: 3.48,
    w: 3.86,
    h: 2.0,
    title: "focal_strehl_only",
    body:
      `Strehl = ${DECK_DATA.run0401.strehlOnlyStrehl.toFixed(4)} (> 1)\nThroughput = ${DECK_DATA.run0401.strehlOnlyThroughput.toFixed(4)}\nSanity result: FAIL`,
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s9, {
    x: 8.78,
    y: 3.48,
    w: 3.86,
    h: 2.0,
    title: "Interpretation",
    body:
      "0401 is more trustworthy as a diagnostic stack than as a victory lap. It explains why later claims had to become more conservative, not more ambitious.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  addEquationPanel(s9, 0.74, 5.94, 11.92, 0.78, "Hard conclusion", [
    "Better physics removed some artifacts, but it also shrank the plausible performance envelope and surfaced non-passive failure modes that could not be ignored.",
  ], C.red);
  finalizeSlide(s9, deck);

  const s10 = deck.addSlide();
  addPageFrame(
    s10,
    "The Bug: Bilinear Beam Reducer",
    "The artifact source that made several early gains look larger than they were",
    C.red,
    "Reference: ppt_fig2_wfe_decomposition.png companion equations"
  );
  addCallout(s10, {
    x: 0.55,
    y: 1.05,
    w: 5.95,
    h: 1.15,
    title: "Wrong implementation",
    body: "Interpolate Re{U} and Im{U} separately during 75:1 reduction.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  s10.addImage({ data: imgData("eq_bilinear.png"), x: 0.72, y: 2.32, w: 5.2, h: 0.42 });
  s10.addText(bulletRuns([
    "Rapid phase oscillations alias under separate bilinear interpolation.",
    "The reducer injects fake high-order structure into vacuum data.",
    "The downstream D2NN then appears to 'correct turbulence' while actually undoing preprocessing damage.",
  ]), {
    x: 0.72,
    y: 2.95,
    w: 5.48,
    h: 1.6,
    fontFace: "DejaVu Sans",
    fontSize: 12,
    color: C.slate,
    margin: 0,
  });
  addCallout(s10, {
    x: 6.82,
    y: 1.05,
    w: 5.95,
    h: 1.15,
    title: "Correct implementation",
    body: "Use complex-field Lanczos resampling, plus defocus compensation and padded propagation where needed.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  s10.addImage({ data: imgData("eq_lanczos.png"), x: 7.02, y: 2.32, w: 4.95, h: 0.42 });
  s10.addText(bulletRuns([
    "Complex-field resampling preserves phase relationships during aggressive reduction.",
    "Vacuum WFE collapses from 420 nm to 3.4 nm.",
    "That redefines the true amount of headroom available to any passive D2NN.",
  ]), {
    x: 7.02,
    y: 2.95,
    w: 5.42,
    h: 1.6,
    fontFace: "DejaVu Sans",
    fontSize: 12,
    color: C.slate,
    margin: 0,
  });
  addEquationPanel(s10, 0.72, 5.3, 11.88, 0.95, "Bottom line", [
    "The preprocessing bug did not merely perturb metrics; it changed what the model learned to compensate.",
  ], C.red);
  finalizeSlide(s10, deck);

  const s11 = deck.addSlide();
  addPageFrame(
    s11,
    "Vacuum Wavefront Error Decomposition",
    "Most of the old vacuum WFE was a numerical artifact, not physical optics",
    C.blue,
    "Figure asset: fig2_wfe.png"
  );
  s11.addImage({ data: imgData("fig2_wfe.png"), x: 0.45, y: 1.0, w: 9.0, h: 4.9 });
  addCallout(s11, {
    x: 9.72,
    y: 1.18,
    w: 2.75,
    h: 1.15,
    title: "Defocus share",
    body: `${DECK_DATA.artifacts.defocusShare.toFixed(1)}%`,
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s11, {
    x: 9.72,
    y: 2.55,
    w: 2.75,
    h: 1.15,
    title: "Higher-order share",
    body: `${DECK_DATA.artifacts.higherOrderShare.toFixed(1)}%`,
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s11, {
    x: 9.72,
    y: 3.92,
    w: 2.75,
    h: 1.55,
    title: "Interpretation",
    body:
      "The dominant 'aberration budget' was not a real optical defect. It was a reducer artifact that the rest of the pipeline quietly inherited.",
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  finalizeSlide(s11, deck);

  const s12 = deck.addSlide();
  addPageFrame(
    s12,
    "D2NN Was Compensating Data Artifacts",
    "Why apparently large PIB gains on the old dataset could not be trusted",
    C.red,
    "Figure asset: fig3_effect.png"
  );
  s12.addImage({ data: imgData("fig3_effect.png"), x: 0.4, y: 1.0, w: 8.75, h: 5.1 });
  addCallout(s12, {
    x: 9.46,
    y: 1.12,
    w: 2.95,
    h: 1.25,
    title: "Old vacuum PIB@10μm",
    body: `${DECK_DATA.artifacts.oldVacuumPIB10.toFixed(1)}%`,
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s12, {
    x: 9.46,
    y: 2.56,
    w: 2.95,
    h: 1.25,
    title: "Old D2NN PIB@10μm",
    body: `${DECK_DATA.artifacts.oldD2nnPIB10.toFixed(1)}%`,
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s12, {
    x: 9.46,
    y: 4.0,
    w: 2.95,
    h: 1.7,
    title: "Interpretation",
    body:
      "A large fraction of the old gain came from repairing the vacuum artifact floor, not from recovering genuine turbulence damage.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  finalizeSlide(s12, deck);

  const s13 = deck.addSlide();
  addPageFrame(
    s13,
    "Why Strehl Ratio Fails Here",
    "Amplitude-changing passive optics break the naive reference assumption",
    C.purple,
    "Figure asset: fig4_strehl.png"
  );
  s13.addImage({ data: imgData("fig4_strehl.png"), x: 0.42, y: 0.98, w: 8.9, h: 3.7 });
  addCallout(s13, {
    x: 9.55,
    y: 1.12,
    w: 2.8,
    h: 1.25,
    title: "Failure mode 1",
    body: "Legacy Strehl could exceed 1 under an invalid reference.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s13, {
    x: 9.55,
    y: 2.56,
    w: 2.8,
    h: 1.25,
    title: "Failure mode 2",
    body: "Even corrected forms can mislead if amplitude preservation is assumed where it does not hold.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  s13.addImage({ data: imgData("eq_cauchy.png"), x: 0.74, y: 5.05, w: 5.55, h: 0.44 });
  addCallout(s13, {
    x: 6.6,
    y: 4.84,
    w: 5.75,
    h: 1.05,
    title: "Operational conclusion",
    body:
      "Use PIB, throughput, and explicit passive-limit checks. Treat Strehl only as a carefully qualified diagnostic.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  finalizeSlide(s13, deck);

  const s14 = deck.addSlide();
  addPageFrame(
    s14,
    "Clean Baseline After Fixing Data",
    "The honest headroom is smaller, but finally interpretable",
    C.green,
    "Reference numbers from ppt_requirements.md and clean-data analysis"
  );
  addCallout(s14, {
    x: 0.62,
    y: 1.08,
    w: 3.85,
    h: 1.2,
    title: "New vacuum PIB@10μm",
    body: `${DECK_DATA.artifacts.newVacuumPIB10.toFixed(1)}%`,
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addCallout(s14, {
    x: 4.74,
    y: 1.08,
    w: 3.85,
    h: 1.2,
    title: "New turbulent PIB@10μm",
    body: `${DECK_DATA.artifacts.newTurbPIB10.toFixed(1)}%`,
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addCallout(s14, {
    x: 8.86,
    y: 1.08,
    w: 3.85,
    h: 1.2,
    title: "True headroom",
    body: `${DECK_DATA.artifacts.headroom.toFixed(1)} percentage points`,
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  s14.addImage({ data: imgData("fig5_baseline.png"), x: 0.55, y: 2.55, w: 8.1, h: 3.85 });
  addCallout(s14, {
    x: 8.92,
    y: 2.66,
    w: 3.5,
    h: 1.55,
    title: "Old vs new vacuum",
    body:
      `${DECK_DATA.artifacts.oldVacuumPIB10.toFixed(1)}% → ${DECK_DATA.artifacts.newVacuumPIB10.toFixed(1)}%\nThis is why old D2NN gains were overstated.`,
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s14, {
    x: 8.92,
    y: 4.45,
    w: 3.5,
    h: 1.55,
    title: "Interpretation",
    body:
      "The corrected baseline no longer promises spectacular gains. It gives a narrower but defensible target: recover part of a 15.5-point PIB gap without violating passivity.",
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  finalizeSlide(s14, deck);

  const s15 = deck.addSlide();
  addPageFrame(
    s15,
    "Mode Conversion Interpretation and Next Steps",
    "What remains physically plausible after the artifact cleanup",
    C.blue,
    "Figure asset: fig6_modes.png"
  );
  s15.addImage({ data: imgData("fig6_modes.png"), x: 0.45, y: 1.0, w: 8.35, h: 3.2 });
  s15.addImage({ data: imgData("eq_modes.png"), x: 0.72, y: 4.45, w: 3.8, h: 0.38 });
  s15.addImage({ data: imgData("eq_mode_convert.png"), x: 0.72, y: 4.95, w: 5.15, h: 0.38 });
  addCallout(s15, {
    x: 9.05,
    y: 1.08,
    w: 3.2,
    h: 1.6,
    title: "Physically plausible claim",
    body:
      "A static D2NN can redistribute modal energy and improve detector concentration without claiming per-sample wavefront correction.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addCallout(s15, {
    x: 9.05,
    y: 2.95,
    w: 3.2,
    h: 1.8,
    title: "Next experiments",
    body:
      "1) Re-train under the hardened stack\n2) Track throughput and unitary deltas during training\n3) Test whether any PIB gain survives passive constraints",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s15, {
    x: 6.15,
    y: 5.05,
    w: 6.05,
    h: 1.0,
    title: "Final research question",
    body:
      "Can a passive, fabrication-feasible D2NN recover some of the 15.5-point clean-data PIB headroom while remaining energy-consistent and unitary-compatible?",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  finalizeSlide(s15, deck);

  const s16 = deck.addSlide();
  addPageFrame(
    s16,
    "Appendix A: Equation Sheet",
    "Compact formula reference used throughout the deck",
    C.purple,
    "Equations embedded as SVG panels for editable, white-background slides"
  );
  addEquationPanel(s16, 0.58, 1.0, 5.95, 1.28, "Unitary overlap preservation", [
    "CO(HUₜ, Uᵥ) is the wrong statement.",
    "Correct: CO(HUₜ, HUᵥ) = CO(Uₜ, Uᵥ)",
  ], C.blue);
  addEquationPanel(s16, 6.78, 1.0, 5.95, 1.28, "Unitary distance preservation", [
    "||HUₜ − HUᵥ||₂ = ||Uₜ − Uᵥ||₂",
  ], C.blue);
  addEquationPanel(s16, 0.58, 2.62, 5.95, 1.28, "Focal PIB loss", [
    "Lₚᵢᵦ = 1 − PIB₁₀μm(focal)",
  ], C.green);
  addEquationPanel(s16, 6.78, 2.62, 5.95, 1.28, "Hybrid loss", [
    "Lₕᵧᵦᵣᵢd = (1 − CO_out) + 0.5(1 − PIB₁₀μm,focal)",
  ], C.green);
  addEquationPanel(s16, 0.58, 4.24, 5.95, 1.28, "Mode expansion", [
    "U = Σ aₘₙ ψₘₙ",
    "After D2NN: U′ = Σ bₘₙ ψₘₙ",
  ], C.amber);
  addEquationPanel(s16, 6.78, 4.24, 5.95, 1.28, "Passive energy conservation", [
    "Σ|bₘₙ|² = Σ|aₘₙ|²",
    "Concentration gain must come from redistribution, not free energy.",
  ], C.amber);
  finalizeSlide(s16, deck);

  const s17 = deck.addSlide();
  addPageFrame(
    s17,
    "Appendix B: Metric Definitions and Sanity Checks",
    "What each number meant once the physics hardened",
    C.red,
    "Use these checks before trusting any apparently strong PIB result"
  );
  addCallout(s17, {
    x: 0.62,
    y: 1.02,
    w: 3.75,
    h: 1.5,
    title: "CO / WF RMS",
    body:
      "Output-plane theorem diagnostics. Useful for checking unitary consistency, not for claiming detector coupling by themselves.",
    fill: C.lightBlue,
    line: C.blue,
    titleColor: C.blue,
  });
  addCallout(s17, {
    x: 4.78,
    y: 1.02,
    w: 3.75,
    h: 1.5,
    title: "PIB",
    body:
      "Detector-plane concentration inside a chosen radius. The most relevant scalar for coupling-oriented focal optimization in this project.",
    fill: C.lightGreen,
    line: C.green,
    titleColor: C.green,
  });
  addCallout(s17, {
    x: 8.94,
    y: 1.02,
    w: 3.75,
    h: 1.5,
    title: "Strehl",
    body:
      "Only meaningful under carefully matched references. It became a diagnostic with explicit upper-bound checks, not a standalone headline metric.",
    fill: C.lightAmber,
    line: C.amber,
    titleColor: C.amber,
  });
  addCallout(s17, {
    x: 0.62,
    y: 3.0,
    w: 3.75,
    h: 1.55,
    title: "Throughput gate",
    body:
      "Require output energy / input energy near 1 for a passive interpretation. Large PIB gains with throughput collapse are not acceptable wins.",
    fill: C.lightRed,
    line: C.red,
    titleColor: C.red,
  });
  addCallout(s17, {
    x: 4.78,
    y: 3.0,
    w: 3.75,
    h: 1.55,
    title: "Unitary CO delta gate",
    body:
      "If CO changes too much between equivalent unitary stages, the stack is no longer behaving like the physics model you think you are optimizing.",
    fill: C.lightPurple,
    line: C.purple,
    titleColor: C.purple,
  });
  addCallout(s17, {
    x: 8.94,
    y: 3.0,
    w: 3.75,
    h: 1.55,
    title: "Passive-limit gate",
    body:
      "Corrected Strehl should not exceed the passive upper bound. If it does, the reference or implementation is still inconsistent.",
    fill: C.offWhite,
    line: C.border,
    titleColor: C.ink,
  });
  addEquationPanel(s17, 0.72, 5.15, 11.9, 1.02, "Checklist before trusting a new run", [
    "Correct objective plane?   Correct physics stack?   Throughput near 1?   Unitary delta small?   Passive-limit metrics satisfied?",
  ], C.red);
  finalizeSlide(s17, deck);
}

async function main() {
  ensureDir(OUT);
  const deck = createDeck();
  buildSlides(deck);
  await deck.writeFile({ fileName: OUT });
  console.log(`PPTX saved to: ${OUT}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
