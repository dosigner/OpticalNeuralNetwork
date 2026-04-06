const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const {
  imageSizingContain,
} = require("./pptxgenjs_helpers/image");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");

const pptx = new pptxgen();
const SHAPES = new pptxgen().shapes;

const ROOT = __dirname;
const RUNS = path.resolve(ROOT, "..");
const DIST = path.join(RUNS, "0405-distance-sweep-rawrp-f6p5mm");
const DATA_SCRIPT = path.resolve(ROOT, "..", "..", "..", "scripts", "generate_data_distance_sweep.py");
const TRAIN_SCRIPT = path.resolve(ROOT, "..", "..", "..", "scripts", "train_distance_sweep.sh");
const OUT =
  process.env.DISTANCE_SEMINAR_KO_OUT ||
  path.join(ROOT, "0406-distance-sweep-seminar-ko.pptx");

const SUMMARY = JSON.parse(
  fs.readFileSync(path.join(DIST, "distance_sweep_summary.json"), "utf8")
);
const PER_DISTANCE = Object.fromEntries(
  [100, 500, 1000, 2000, 3000].map((distanceM) => {
    const resultPath = path.join(
      DIST,
      `L${distanceM}m`,
      "focal_raw_received_power",
      "results.json"
    );
    return [distanceM, JSON.parse(fs.readFileSync(resultPath, "utf8"))];
  })
);

const IMG = {
  summary6: path.join(DIST, "21_distance_sweep_summary_6panel.png"),
  focal100: path.join(
    DIST,
    "L100m",
    "focal_raw_received_power",
    "06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png"
  ),
  focal3k: path.join(
    DIST,
    "L3000m",
    "focal_raw_received_power",
    "06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png"
  ),
  phase100: path.join(
    DIST,
    "L100m",
    "focal_raw_received_power",
    "15_phase_masks_5layers.png"
  ),
  phase3k: path.join(
    DIST,
    "L3000m",
    "focal_raw_received_power",
    "15_phase_masks_5layers.png"
  ),
};

const C = {
  ink: "162334",
  slate: "5B6B79",
  navy: "12304A",
  blue: "1E6091",
  cyan: "2A9DCE",
  softBlue: "EAF4FB",
  pale: "F7FAFC",
  card: "FFFFFF",
  line: "D7E2EB",
  green: "1E8E5A",
  softGreen: "EAF7EF",
  amber: "C77A1A",
  softAmber: "FFF4E3",
  red: "B54A46",
  softRed: "FBECEC",
  violet: "6B5CB3",
  softViolet: "F2EEFB",
  white: "FFFFFF",
};

const FONTS = {
  header: "Noto Sans CJK KR",
  body: "Noto Sans CJK KR",
  mono: "Noto Sans Mono CJK KR",
};

pptx.layout = "LAYOUT_WIDE";
pptx.author = "D2NN Lab";
pptx.company = "D2NN Lab";
pptx.subject = "Distance sweep seminar deck";
pptx.title = "0405 Distance Sweep Seminar (Korean)";
pptx.lang = "ko-KR";

function pct(value, digits = 1) {
  return `${value.toFixed(digits)}%`;
}

function fmt(value, digits = 2) {
  return value.toFixed(digits);
}

function fmtDb(value, digits = 2) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)} dB`;
}

function shortDistance(distanceM) {
  if (distanceM >= 1000) {
    const km = distanceM / 1000;
    return Number.isInteger(km) ? `${km} km` : `${km.toFixed(1)} km`;
  }
  return `${distanceM} m`;
}

function airyRadiusUm(fMm) {
  const wavelengthM = 1.55e-6;
  const diameterM = 2.048e-3;
  return (1.22 * wavelengthM * (fMm * 1e-3) / diameterM) * 1e6;
}

function deckShadow() {
  return {
    type: "outer",
    color: "000000",
    blur: 3,
    offset: 1.2,
    angle: 45,
    opacity: 0.08,
  };
}

function addTitleBar(slide, title, subtitle = "") {
  slide.background = { color: C.pale };
  slide.addShape(SHAPES.RECTANGLE, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.08,
    line: { color: C.blue, transparency: 100 },
    fill: { color: C.blue },
  });
  slide.addText(title, {
    x: 0.6,
    y: 0.22,
    w: 8.8,
    h: 0.42,
    fontFace: FONTS.header,
    fontSize: 24,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 9.75,
      y: 0.28,
      w: 2.9,
      h: 0.2,
      fontFace: FONTS.body,
      fontSize: 9,
      italic: true,
      color: C.slate,
      align: "right",
      margin: 0,
    });
  }
  slide.addShape(SHAPES.LINE, {
    x: 0.6,
    y: 0.78,
    w: 12.1,
    h: 0,
    line: { color: C.line, width: 1 },
  });
}

function addFooter(slide, index, text) {
  slide.addText(text, {
    x: 0.6,
    y: 7.02,
    w: 10.7,
    h: 0.15,
    fontFace: FONTS.body,
    fontSize: 7.5,
    color: C.slate,
    margin: 0,
  });
  slide.addText(`${index}/11`, {
    x: 11.9,
    y: 7.0,
    w: 0.8,
    h: 0.15,
    fontFace: FONTS.body,
    fontSize: 8,
    color: C.slate,
    align: "right",
    margin: 0,
  });
}

function addCard(slide, opts) {
  slide.addShape(SHAPES.ROUNDED_RECTANGLE, {
    x: opts.x,
    y: opts.y,
    w: opts.w,
    h: opts.h,
    rectRadius: 0.05,
    line: { color: opts.line || C.line, width: 1 },
    fill: { color: opts.fill || C.card },
    shadow: deckShadow(),
  });
  if (opts.title) {
    slide.addText(opts.title, {
      x: opts.x + 0.18,
      y: opts.y + 0.13,
      w: opts.w - 0.36,
      h: 0.22,
      fontFace: FONTS.body,
      fontSize: 11,
      bold: true,
      color: opts.titleColor || C.blue,
      margin: 0,
    });
  }
}

function addBullets(slide, items, box) {
  const runs = [];
  items.forEach((item, idx) => {
    runs.push({
      text: item,
      options: { bullet: true, breakLine: idx !== items.length - 1 },
    });
  });
  slide.addText(runs, {
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    fontFace: FONTS.body,
    fontSize: box.fontSize || 11.5,
    color: box.color || C.ink,
    margin: 0,
    valign: "top",
  });
}

function addMetricPill(slide, opts) {
  slide.addShape(SHAPES.ROUNDED_RECTANGLE, {
    x: opts.x,
    y: opts.y,
    w: opts.w,
    h: opts.h,
    rectRadius: 0.05,
    line: { color: opts.line || C.line, width: 1 },
    fill: { color: opts.fill || C.card },
  });
  slide.addText(opts.label, {
    x: opts.x + 0.12,
    y: opts.y + 0.07,
    w: opts.w - 0.24,
    h: 0.11,
    fontFace: FONTS.body,
    fontSize: 8.5,
    color: opts.labelColor || C.slate,
    margin: 0,
    align: "center",
  });
  slide.addText(opts.value, {
    x: opts.x + 0.12,
    y: opts.y + 0.31,
    w: opts.w - 0.24,
    h: 0.14,
    fontFace: FONTS.body,
    fontSize: 15,
    bold: true,
    color: opts.valueColor || C.ink,
    margin: 0,
    align: "center",
  });
}

function placeImage(slide, imgPath, box, border = false) {
  slide.addImage({
    path: imgPath,
    ...imageSizingContain(imgPath, box.x, box.y, box.w, box.h),
  });
  if (border) {
    slide.addShape(SHAPES.RECTANGLE, {
      x: box.x,
      y: box.y,
      w: box.w,
      h: box.h,
      line: { color: C.line, width: 1 },
      fill: { color: C.white, transparency: 100 },
    });
  }
}

function finalize(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function readScriptFacts() {
  const dataScript = fs.readFileSync(DATA_SCRIPT, "utf8");
  const trainScript = fs.readFileSync(TRAIN_SCRIPT, "utf8");
  const realizationMatch = dataScript.match(/N_REALIZATIONS\s*=\s*(\d+)/);
  const splitMatch = dataScript.match(/TRAIN_N,\s*VAL_N,\s*TEST_N\s*=\s*(\d+),\s*(\d+),\s*(\d+)/);
  const focusMatch = trainScript.match(/focus_f_m':\s*([0-9.e+-]+)/);
  return {
    realizations: realizationMatch ? Number(realizationMatch[1]) : 2000,
    train: splitMatch ? Number(splitMatch[1]) : 1600,
    val: splitMatch ? Number(splitMatch[2]) : 200,
    test: splitMatch ? Number(splitMatch[3]) : 200,
    focusFmm: focusMatch ? Number(focusMatch[1]) * 1e3 : 6.5,
  };
}

const SCRIPT_FACTS = readScriptFacts();
const RESULTS = Object.fromEntries(SUMMARY.map((row) => [row.L_m, row]));
const AIRY_45 = airyRadiusUm(4.5);
const AIRY_65 = airyRadiusUm(6.5);
const WFE_BASELINE_VALUES = SUMMARY.map((row) => PER_DISTANCE[row.L_m].wf_rms_baseline_nm);
const WFE_MIN = Math.min(...WFE_BASELINE_VALUES);
const WFE_MAX = Math.max(...WFE_BASELINE_VALUES);
const TP_MIN = Math.min(...SUMMARY.map((row) => row.tp_d2nn));
const TP_MAX = Math.max(...SUMMARY.map((row) => row.tp_d2nn));

function slide1() {
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape(SHAPES.RECTANGLE, {
    x: 0,
    y: 0,
    w: 0.16,
    h: 7.5,
    line: { color: C.cyan, transparency: 100 },
    fill: { color: C.cyan },
  });
  s.addText("Static D2NN Distance Sweep", {
    x: 0.72,
    y: 0.9,
    w: 5.8,
    h: 0.5,
    fontFace: FONTS.header,
    fontSize: 27,
    bold: true,
    color: C.white,
    margin: 0,
  });
  s.addText("Distance Sweep 결과는 무엇을 보여주는가?", {
    x: 0.72,
    y: 1.55,
    w: 7.0,
    h: 0.55,
    fontFace: FONTS.header,
    fontSize: 22,
    bold: true,
    color: "D4EDF9",
    margin: 0,
  });
  s.addShape(SHAPES.LINE, {
    x: 0.72,
    y: 2.35,
    w: 2.2,
    h: 0,
    line: { color: C.cyan, width: 2.5 },
  });
  addBullets(
    s,
    [
      "목표는 PIB가 아니라 10 μm focal bucket 안 absolute received power 증가 여부를 보는 것이다.",
      "이번 deck은 거리 sweep 결과를 재배치하면서, quantity 정의와 해석 가능한 범위를 먼저 분리한다.",
      `표시된 sweep은 ${SCRIPT_FACTS.realizations} realizations / 거리, split ${SCRIPT_FACTS.train}/${SCRIPT_FACTS.val}/${SCRIPT_FACTS.test} 기준의 100 m–3 km subset만 다룬다.`,
    ],
    { x: 0.9, y: 2.8, w: 6.4, h: 1.7, fontSize: 13, color: C.white }
  );
  addCard(s, { x: 8.0, y: 1.0, w: 4.45, h: 4.2, fill: "17344E", line: "31516D" });
  s.addText("핵심 숫자", {
    x: 8.22,
    y: 1.18,
    w: 1.2,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.cyan,
    margin: 0,
  });
  const highlightRows = [
    ["100 m gain", fmtDb(RESULTS[100].improvement_db, 2)],
    ["1 km gain", fmtDb(RESULTS[1000].improvement_db, 2)],
    ["3 km gain", fmtDb(RESULTS[3000].improvement_db, 2)],
    ["3 km fade floor", `${fmt(RESULTS[3000].turb_min_db, 1)} → ${fmt(RESULTS[3000].d2nn_min_db, 1)} dB`],
    ["D2NN throughput", `${fmt(TP_MIN * 100, 1)}–${fmt(TP_MAX * 100, 1)}%`],
  ];
  highlightRows.forEach((row, idx) => {
    const y = 1.65 + idx * 0.58;
    s.addText(row[0], {
      x: 8.22,
      y,
      w: 1.55,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 10.5,
      color: "C7D7E3",
      margin: 0,
    });
    s.addText(row[1], {
      x: 9.9,
      y: y - 0.02,
      w: 2.1,
      h: 0.22,
      fontFace: FONTS.body,
      fontSize: idx === 2 ? 16 : 14,
      bold: true,
      color: idx === 2 ? "B8F3C6" : C.white,
      align: "right",
      margin: 0,
    });
  });
  s.addText("메시지: 평균 gain 자체보다, quantity 정의를 바르게 두었을 때 어떤 관측만 안전하게 말할 수 있는지가 중요하다.", {
    x: 0.72,
    y: 6.55,
    w: 11.2,
    h: 0.26,
    fontFace: FONTS.body,
    fontSize: 10,
    color: C.slate,
    margin: 0,
  });
  addFooter(s, 1, "Source: 0405 distance_sweep_summary.json, per-distance results.json");
  finalize(s);
}

function slide2() {
  const s = pptx.addSlide();
  addTitleBar(s, "질문과 발표 흐름", "세미나 설득형 구성");
  addCard(s, { x: 0.65, y: 1.05, w: 3.95, h: 4.95, fill: C.card });
  s.addText("이번 발표에서 먼저 고정할 질문", {
    x: 0.9,
    y: 1.25,
    w: 2.3,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  s.addText("정적 phase-only D2NN가 각 샘플의 난류 wavefront를 복원한다고 주장할 수는 없어도,\n거리 증가에 따라 central bucket 안 absolute received power와 fade tail을 어떻게 바꾸는가?", {
    x: 0.9,
    y: 1.65,
    w: 3.45,
    h: 1.25,
    fontFace: FONTS.header,
    fontSize: 18,
    bold: true,
    color: C.ink,
    margin: 0,
  });
  addBullets(
    s,
    [
      "먼저 quantity 정의와 해석 가능한 범위를 분리한다.",
      "그다음 거리별 수치와 6-panel figure를 보여준다.",
      "마지막에 100 m와 2–3 km에서 관측되는 현상을 보수적 문장으로 요약한다.",
    ],
    { x: 0.9, y: 3.35, w: 3.25, h: 1.8, fontSize: 11.5 }
  );

  addCard(s, { x: 4.9, y: 1.05, w: 7.8, h: 4.95, fill: C.softBlue, line: "C7DDEB" });
  const flow = [
    ["1", "지표 정의", "bucket power, throughput, Raw RP loss, D/r₀"],
    ["2", "geometry 확인", "15 cm telescope aperture와 2.048 mm internal window 구분"],
    ["3", "distance 결과", "100 m–3 km table + 6-panel evidence"],
    ["4", "보수 해석", "mode reshaping / fade-tail compression 관측 수준으로만 정리"],
    ["5", "한계", "200 held-out samples, finite-NA/SMF mode overlap 미모델링"],
  ];
  flow.forEach((row, idx) => {
    const y = 1.4 + idx * 0.82;
    s.addShape(SHAPES.ROUNDED_RECTANGLE, {
      x: 5.2,
      y,
      w: 7.2,
      h: 0.55,
      rectRadius: 0.04,
      line: { color: idx === 2 ? C.green : C.line, width: 1 },
      fill: { color: idx === 2 ? "F0FAF4" : C.card },
    });
    s.addShape(SHAPES.OVAL, {
      x: 5.37,
      y: y + 0.08,
      w: 0.34,
      h: 0.34,
      line: { color: C.blue, transparency: 100 },
      fill: { color: C.blue },
    });
    s.addText(row[0], {
      x: 5.37,
      y: y + 0.085,
      w: 0.34,
      h: 0.3,
      fontFace: FONTS.body,
      fontSize: 11,
      bold: true,
      color: C.white,
      align: "center",
      margin: 0,
    });
    s.addText(row[1], {
      x: 5.86,
      y: y + 0.08,
      w: 1.3,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 11,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    s.addText(row[2], {
      x: 7.18,
      y: y + 0.08,
      w: 4.9,
      h: 0.2,
      fontFace: FONTS.body,
      fontSize: 10.3,
      color: C.slate,
      margin: 0,
    });
  });
  addFooter(s, 2, "Flow driver: quantity mismatch를 먼저 정리하고 결과를 읽는 순서");
  finalize(s);
}

function slide3() {
  const s = pptx.addSlide();
  addTitleBar(s, "지표 정의와 해석 범위", "quantity 먼저 고정");
  const cardY = 1.15;
  [
    {
      x: 0.65,
      title: "Bucket Power",
      fill: C.softBlue,
      line: "C8DCE8",
      body: "10 μm focal bucket 안 absolute energy.\n이번 sweep의 primary observable.",
    },
    {
      x: 4.42,
      title: "Throughput",
      fill: C.softGreen,
      line: "CDE3D3",
      body: "output energy / input energy.\n수동 소자 해석에서 반드시 같이 봐야 하는 항목.",
    },
    {
      x: 8.19,
      title: "Raw RP loss",
      fill: C.softAmber,
      line: "E8D9B6",
      body: "−log(E_bucket / E_input + ε).\nVacuum reference 없이 absolute bucket power를 직접 최적화.",
    },
  ].forEach((card) => {
    addCard(s, { x: card.x, y: cardY, w: 3.15, h: 2.0, fill: card.fill, line: card.line, title: card.title });
    s.addText(card.body, {
      x: card.x + 0.2,
      y: cardY + 0.55,
      w: 2.75,
      h: 1.0,
      fontFace: FONTS.body,
      fontSize: 12,
      color: C.ink,
      margin: 0,
      valign: "mid",
    });
  });

  addCard(s, { x: 0.8, y: 3.55, w: 11.75, h: 1.2, fill: C.card });
  s.addText("발표에서 허용하는 문장", {
    x: 1.02,
    y: 3.72,
    w: 1.45,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.green,
    margin: 0,
  });
  s.addText("“bucket power가 증가했다”, “fade-tail이 완화되는 관측이 있다”, “mode reshaping과 일관적인 그림이다”", {
    x: 3.2,
    y: 3.72,
    w: 8.45,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });
  s.addText("발표에서 피해야 하는 문장", {
    x: 1.02,
    y: 4.1,
    w: 1.6,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.red,
    margin: 0,
  });
  s.addText("“wavefront correction을 증명했다”, “link outage와 recovery를 단정한다”, “WFE가 전체 sweep에서 unchanged ≈ 442 nm다”", {
    x: 3.2,
    y: 4.1,
    w: 8.4,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });

  addCard(s, { x: 0.8, y: 5.15, w: 11.75, h: 1.15, fill: C.softViolet, line: "DDD3F3" });
  s.addText("D/r₀는 15 cm telescope aperture 기준의 난류 분류량이고, 2.048 mm는 beam-reduced D2NN internal window다. 두 길이 척도는 목적이 다르므로 같은 D로 읽으면 안 된다.", {
    x: 1.05,
    y: 5.5,
    w: 11.2,
    h: 0.4,
    fontFace: FONTS.header,
    fontSize: 18,
    bold: true,
    color: C.violet,
    align: "center",
    margin: 0,
  });
  addFooter(s, 3, "Guardrail: quantity와 해석 수준을 먼저 고정");
  finalize(s);
}

function slide4() {
  const s = pptx.addSlide();
  addTitleBar(s, "시스템 설정과 geometry", "distance sweep 입력/출력 정의");

  addCard(s, { x: 0.65, y: 1.05, w: 4.25, h: 4.9, fill: C.card });
  s.addText("실험 설정", {
    x: 0.9,
    y: 1.22,
    w: 1.1,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  addBullets(
    s,
    [
      `파장 1.55 μm, phase-only 5-layer D2NN, inter-layer spacing 10 mm`,
      `focus lens f = ${fmt(SCRIPT_FACTS.focusFmm, 1)} mm, detector distance 10 mm`,
      `beam-reduced internal window 2.048 mm, aperture parameter 2.0 mm`,
      `평가 bucket radius 10 μm, held-out test ${SCRIPT_FACTS.test} samples / 거리`,
      `distance subset: 100 m, 500 m, 1 km, 2 km, 3 km`,
    ],
    { x: 0.95, y: 1.6, w: 3.7, h: 2.55, fontSize: 11.3 }
  );
  addMetricPill(s, {
    x: 0.92, y: 4.5, w: 1.55, h: 0.78,
    label: "dx_focal", value: `${fmt(PER_DISTANCE[100].dx_focal_um, 2)} μm`,
    fill: C.softBlue, line: "C8DCE8", valueColor: C.blue,
  });
  addMetricPill(s, {
    x: 2.55, y: 4.5, w: 1.55, h: 0.78,
    label: "Airy@6.5", value: `${fmt(AIRY_65, 2)} μm`,
    fill: C.softGreen, line: "CDE3D3", valueColor: C.green,
  });

  addCard(s, { x: 5.2, y: 1.05, w: 7.45, h: 4.9, fill: C.softBlue, line: "C7DDEB" });
  s.addText("두 aperture를 분리해서 읽어야 하는 이유", {
    x: 5.45,
    y: 1.22,
    w: 2.6,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  const blocks = [
    ["15 cm telescope aperture", "난류가 receiver pupil에 걸리는 강도를 분류하는 physical entrance aperture"],
    ["75:1 beam reducer", "15 cm scale pupil을 2.048 mm 내부 계산 window로 매핑하는 relay surrogate"],
    ["2.048 mm internal window", "D2NN가 보는 계산 domain. sampling, dx, focal scaling 계산에 사용"],
    ["2.0 mm aperture parameter", "D2NN output aperture/training config에 들어간 loss support"],
  ];
  blocks.forEach((row, idx) => {
    const y = 1.62 + idx * 0.8;
    s.addShape(SHAPES.ROUNDED_RECTANGLE, {
      x: 5.45,
      y,
      w: 6.95,
      h: 0.58,
      rectRadius: 0.04,
      line: { color: C.line, width: 1 },
      fill: { color: idx % 2 === 0 ? C.card : "F4F8FB" },
    });
    s.addText(row[0], {
      x: 5.68,
      y: y + 0.08,
      w: 1.6,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 11,
      bold: true,
      color: C.ink,
      margin: 0,
    });
    s.addText(row[1], {
      x: 7.35,
      y: y + 0.08,
      w: 4.7,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 10.5,
      color: C.slate,
      margin: 0,
    });
  });
  addCard(s, { x: 5.45, y: 5.0, w: 6.95, h: 0.72, fill: C.softAmber, line: "E8D9B6" });
  s.addText("따라서 D/r₀ table의 D는 15 cm이고, Airy/bucket geometry의 D는 2.048 mm internal pupil 근사와 연결된다.", {
    x: 5.72,
    y: 5.25,
    w: 6.4,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11.2,
    bold: true,
    color: C.amber,
    margin: 0,
    align: "center",
  });
  addFooter(s, 4, "Source: generate_data_distance_sweep.py, train_distance_sweep.sh");
  finalize(s);
}

function slide5() {
  const s = pptx.addSlide();
  addTitleBar(s, "난류 regime과 quantity 정의", "D/r₀와 σ_R²는 분류량");
  const tableRows = [
    [
      { text: "거리", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "D/r₀", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "σ_R²", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "분류", options: { bold: true, color: C.white, fill: { color: C.blue } } },
    ],
  ];
  SUMMARY.forEach((row, idx) => {
    let regime = "weak";
    if (row.L_m >= 500 && row.L_m < 2000) regime = "moderate / strong onset";
    if (row.L_m >= 2000) regime = "strong";
    tableRows.push([
      shortDistance(row.L_m),
      fmt(row.D_over_r0, 2),
      fmt(row.sigma_R2, 2),
      regime,
    ]);
  });
  s.addTable(tableRows, {
    x: 0.8,
    y: 1.2,
    w: 5.2,
    colW: [1.0, 1.0, 1.0, 2.0],
    rowH: 0.42,
    fontFace: FONTS.body,
    fontSize: 11,
    border: { pt: 0.5, color: C.line },
    fill: C.card,
    color: C.ink,
  });
  addCard(s, { x: 6.35, y: 1.2, w: 6.0, h: 2.0, fill: C.softBlue, line: "C8DCE8", title: "식은 어떻게 읽을까?" });
  s.addText("r₀ = (0.423 k² Cn² L)^(-3/5),   σ_R² = 1.23 Cn² k^(7/6) L^(11/6)", {
    x: 6.65,
    y: 1.72,
    w: 5.45,
    h: 0.24,
    fontFace: FONTS.mono,
    fontSize: 14,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  s.addText("이 slide의 역할은 turbulence strength를 분류하는 것이다. D2NN가 무엇을 '증명했다'고 읽는 slide가 아니다.", {
    x: 6.7,
    y: 2.2,
    w: 5.3,
    h: 0.45,
    fontFace: FONTS.body,
    fontSize: 11.5,
    color: C.slate,
    margin: 0,
    align: "center",
  });
  addCard(s, { x: 6.35, y: 3.55, w: 6.0, h: 2.25, fill: C.softGreen, line: "CDE3D3", title: "발표에서 안전한 해석" });
  addBullets(
    s,
    [
      "100 m는 weak 쪽에 가깝고, 2–3 km는 strong scintillation regime에 들어간다.",
      "거리 증가와 함께 mean gain, variance reduction, minimum received power가 어떻게 움직이는지를 비교한다.",
      "regime 이름은 결과 읽기의 배경 설명이지, 메커니즘 단정의 근거가 아니다.",
    ],
    { x: 6.63, y: 4.0, w: 5.3, h: 1.4, fontSize: 11.2 }
  );
  addFooter(s, 5, "Source: distance_sweep_summary.json; D/r₀ uses telescope aperture = 15 cm");
  finalize(s);
}

function slide6() {
  const s = pptx.addSlide();
  addTitleBar(s, "거리별 핵심 결과", "absolute bucket power 기준");
  const rows = [
    [
      { text: "거리", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "Vac", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "Turb", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "D2NN", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "gain", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "throughput", options: { bold: true, color: C.white, fill: { color: C.blue } } },
    ],
  ];
  SUMMARY.forEach((row, idx) => {
    rows.push([
      shortDistance(row.L_m),
      fmt(row.vac_bucket, row.L_m >= 3000 ? 2 : 1),
      fmt(row.turb_bucket, row.L_m >= 3000 ? 2 : 1),
      fmt(row.d2nn_bucket, row.L_m >= 3000 ? 2 : 1),
      `${pct(row.improvement_pct, row.L_m >= 500 && row.L_m < 2000 ? 1 : 0)} (${fmtDb(row.improvement_db, 2)})`,
      `${fmt(row.tp_d2nn * 100, 1)}%`,
    ]);
  });
  s.addTable(rows, {
    x: 0.55,
    y: 1.22,
    w: 12.1,
    colW: [1.05, 1.7, 1.7, 1.7, 3.15, 1.7],
    rowH: 0.42,
    fontFace: FONTS.body,
    fontSize: 11,
    border: { pt: 0.5, color: C.line },
    fill: C.card,
    color: C.ink,
  });
  addCard(s, { x: 0.85, y: 4.15, w: 3.7, h: 1.4, fill: C.softGreen, line: "CDE3D3" });
  s.addText("mean gain이 거의 없는 구간", {
    x: 1.08,
    y: 4.35,
    w: 1.8,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.green,
    margin: 0,
  });
  s.addText("500 m–1 km는 +0.26 dB, +0.37 dB 수준이라 headline보다 error bar와 해석 문장이 더 중요하다.", {
    x: 1.08,
    y: 4.72,
    w: 3.25,
    h: 0.45,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });
  addCard(s, { x: 4.85, y: 4.15, w: 3.7, h: 1.4, fill: C.softAmber, line: "E8D9B6" });
  s.addText("headline이 되는 구간", {
    x: 5.08,
    y: 4.35,
    w: 1.6,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.amber,
    margin: 0,
  });
  s.addText("2 km와 3 km는 mean gain 자체도 크고, minimum received power와 variance reduction도 같이 개선된다.", {
    x: 5.08,
    y: 4.72,
    w: 3.2,
    h: 0.45,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });
  addCard(s, { x: 8.85, y: 4.15, w: 3.7, h: 1.4, fill: C.softRed, line: "E7CACA" });
  s.addText("weak regime 특이점", {
    x: 9.08,
    y: 4.35,
    w: 1.5,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.red,
    margin: 0,
  });
  s.addText("100 m에서 D2NN bucket이 vacuum보다 커진다. 이건 energy gain이 아니라 bucket-relative mode reshaping으로 읽어야 한다.", {
    x: 9.08,
    y: 4.72,
    w: 3.15,
    h: 0.45,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });
  addFooter(s, 6, "Source: 0405 distance_sweep_summary.json");
  finalize(s);
}

function slide7() {
  const s = pptx.addSlide();
  addTitleBar(s, "6-panel 요약 figure", "무엇을 읽어야 하는가");
  placeImage(s, IMG.summary6, { x: 0.45, y: 1.02, w: 12.45, h: 5.05 }, true);
  addCard(s, { x: 0.8, y: 6.16, w: 12.0, h: 0.56, fill: C.softBlue, line: "C8DCE8" });
  s.addText("왼쪽 위/아래는 mean bucket power와 vacuum 대비 gain, 가운데는 gain이 strong regime로 갈수록 커지는 패턴, 오른쪽은 scintillation std와 output throughput을 함께 읽으면 된다.", {
    x: 1.0,
    y: 6.34,
    w: 11.55,
    h: 0.16,
    fontFace: FONTS.body,
    fontSize: 11.4,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  addFooter(s, 7, "Source figure: 21_distance_sweep_summary_6panel.png");
  finalize(s);
}

function slide8() {
  const s = pptx.addSlide();
  addTitleBar(s, "100 m에서 관측되는 mode reshaping", "weak regime observation");
  placeImage(s, IMG.focal100, { x: 0.65, y: 1.2, w: 7.2, h: 3.7 }, true);
  placeImage(s, IMG.phase100, { x: 8.25, y: 1.2, w: 4.4, h: 1.6 }, true);
  addCard(s, { x: 8.25, y: 3.0, w: 4.4, h: 1.95, fill: C.softGreen, line: "CDE3D3" });
  s.addText("수치 메모", {
    x: 8.48,
    y: 3.18,
    w: 0.9,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.green,
    margin: 0,
  });
  addBullets(
    s,
    [
      `vac ${fmt(RESULTS[100].vac_bucket, 1)} → d2nn ${fmt(RESULTS[100].d2nn_bucket, 1)} (${fmtDb(RESULTS[100].improvement_db, 2)})`,
      `baseline WFE ${fmt(PER_DISTANCE[100].wf_rms_baseline_nm, 1)} nm, D2NN ${fmt(PER_DISTANCE[100].wf_rms_nm, 1)} nm`,
      `throughput ${fmt(RESULTS[100].tp_d2nn * 100, 1)}% 유지`,
    ],
    { x: 8.48, y: 3.55, w: 3.8, h: 1.1, fontSize: 11.2 }
  );
  addCard(s, { x: 0.75, y: 5.15, w: 11.9, h: 1.1, fill: C.softAmber, line: "E8D9B6" });
  s.addText("안전한 해석: weak regime에서는 D2NN가 vacuum focal spot과 다른 mode profile을 만들어 10 μm bucket 안 에너지를 더 모으는 관측이 있다. 이것을 wavefront correction이나 energy creation으로 읽으면 안 된다.", {
    x: 1.0,
    y: 5.45,
    w: 11.35,
    h: 0.38,
    fontFace: FONTS.body,
    fontSize: 12,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  addFooter(s, 8, "Source: L100m focal plane / phase mask figures and results.json");
  finalize(s);
}

function slide9() {
  const s = pptx.addSlide();
  addTitleBar(s, "2–3 km에서 보이는 평균 gain과 fade-tail 변화", "strong regime observation");
  placeImage(s, IMG.focal3k, { x: 0.65, y: 1.2, w: 7.2, h: 3.7 }, true);
  placeImage(s, IMG.phase3k, { x: 8.25, y: 1.2, w: 4.4, h: 1.6 }, true);

  const deepRows = [
    [
      { text: "거리", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "std (turb→d2nn)", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "min power (turb→d2nn)", options: { bold: true, color: C.white, fill: { color: C.blue } } },
    ],
    [shortDistance(2000), `${fmt(RESULTS[2000].turb_std_db, 2)} → ${fmt(RESULTS[2000].d2nn_std_db, 2)} dB`, `${fmt(RESULTS[2000].turb_min_db, 1)} → ${fmt(RESULTS[2000].d2nn_min_db, 1)} dB`],
    [shortDistance(3000), `${fmt(RESULTS[3000].turb_std_db, 2)} → ${fmt(RESULTS[3000].d2nn_std_db, 2)} dB`, `${fmt(RESULTS[3000].turb_min_db, 1)} → ${fmt(RESULTS[3000].d2nn_min_db, 1)} dB`],
  ];
  s.addTable(deepRows, {
    x: 8.0,
    y: 3.0,
    w: 4.7,
    colW: [0.9, 1.8, 2.0],
    rowH: 0.42,
    fontFace: FONTS.body,
    fontSize: 10.3,
    border: { pt: 0.5, color: C.line },
    fill: C.card,
    color: C.ink,
  });
  addCard(s, { x: 0.75, y: 5.15, w: 11.9, h: 1.1, fill: C.softBlue, line: "C8DCE8" });
  s.addText("안전한 해석: strong regime에서는 D2NN가 held-out ensemble에서 mean bucket power뿐 아니라 low-tail received power와 variance에도 개선을 보인다. 이 slide는 'fade-tail compression 관측'까지가 안전한 표현이다.", {
    x: 1.0,
    y: 5.45,
    w: 11.35,
    h: 0.38,
    fontFace: FONTS.body,
    fontSize: 12,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  addFooter(s, 9, "Source: L3000m focal plane / phase mask figures and distance_sweep_summary.json");
  finalize(s);
}

function slide10() {
  const s = pptx.addSlide();
  addTitleBar(s, "Objective와 f = 6.5 mm 설정", "design choice rationale");
  addCard(s, { x: 0.65, y: 1.15, w: 5.85, h: 4.75, fill: C.card, title: "왜 Raw RP loss인가?" });
  s.addText("L = −log(E_bucket / E_input + ε)", {
    x: 1.0,
    y: 1.72,
    w: 5.15,
    h: 0.24,
    fontFace: FONTS.mono,
    fontSize: 17,
    bold: true,
    color: C.ink,
    margin: 0,
    align: "center",
  });
  addBullets(
    s,
    [
      "vacuum reference가 없어도 absolute bucket power를 직접 최적화한다.",
      "throughput 저하가 있으면 objective 자체가 같이 불리해진다.",
      "따라서 PIB-only loss처럼 denominator를 줄여 점수를 높이는 경로가 상대적으로 약하다.",
      "이 식은 'physics를 증명'하는 것이 아니라, 발표 목표 quantity와 직접 연결되는 objective다.",
    ],
    { x: 0.95, y: 2.35, w: 5.2, h: 2.2, fontSize: 11.5 }
  );
  addMetricPill(s, {
    x: 1.0, y: 4.88, w: 2.0, h: 0.8,
    label: "D2NN throughput", value: `${fmt(TP_MIN * 100, 1)}–${fmt(TP_MAX * 100, 1)}%`,
    fill: C.softGreen, line: "CDE3D3", valueColor: C.green,
  });
  addCard(s, { x: 6.8, y: 1.15, w: 5.85, h: 4.75, fill: C.softBlue, line: "C7DDEB", title: "왜 f = 6.5 mm인가?" });
  const fRows = [
    ["f = 4.5 mm", `${fmt(AIRY_45, 2)} μm`, `${fmt(10 / AIRY_45, 2)} Airy radii`],
    ["f = 6.5 mm", `${fmt(AIRY_65, 2)} μm`, `${fmt(10 / AIRY_65, 2)} Airy radii`],
  ];
  const opticalRows = [
    [
      { text: "설정", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "Airy radius", options: { bold: true, color: C.white, fill: { color: C.blue } } },
      { text: "10 μm bucket / Airy", options: { bold: true, color: C.white, fill: { color: C.blue } } },
    ],
    ...fRows,
  ];
  s.addTable(opticalRows, {
    x: 7.1,
    y: 1.72,
    w: 5.2,
    colW: [1.5, 1.4, 2.1],
    rowH: 0.42,
    fontFace: FONTS.body,
    fontSize: 10.8,
    border: { pt: 0.5, color: C.line },
    fill: C.card,
    color: C.ink,
  });
  addBullets(
    s,
    [
      "f = 4.5 mm에서는 10 μm bucket이 diffraction-limited core를 이미 넓게 포함한다.",
      "f = 6.5 mm에서는 bucket / Airy ratio가 더 타이트해져 mode reshaping headroom이 커진다.",
      "이 slide 역시 'bandwidth가 더 잘 맞는다'를 증명하는 것이 아니라 geometry가 더 타이트해졌다는 수준으로만 읽는다.",
    ],
    { x: 7.15, y: 3.25, w: 5.0, h: 1.8, fontSize: 11.1 }
  );
  addFooter(s, 10, "Source: train_distance_sweep.sh, results.json, Airy calculation from internal pupil geometry");
  finalize(s);
}

function slide11() {
  const s = pptx.addSlide();
  addTitleBar(s, "한계와 다음 검증", "이번 deck의 멈춤선");
  addCard(s, { x: 0.7, y: 1.15, w: 5.95, h: 4.75, fill: C.softRed, line: "E8CACA", title: "현재 deck에서 반드시 남겨둘 caveat" });
  addBullets(
    s,
    [
      `각 거리의 held-out test는 ${SCRIPT_FACTS.test} samples이며, strong-tail percentile을 안정적으로 말하기엔 적다.`,
      `higher-order WFE baseline은 ${fmt(WFE_MIN, 0)}–${fmt(WFE_MAX, 0)} nm 범위로 여전히 크다. 전체 sweep을 하나의 "~442 nm unchanged" 문장으로 축약하면 안 된다.`,
      "bucket power는 SMF mode overlap이나 finite-NA detector collected power를 직접 모델링한 값이 아니다.",
      "100 m의 vacuum 초과 bucket은 real lens baseline과 relay optics 모델을 더 넣어 다시 검증할 필요가 있다.",
      "static phase-only mask이 sample-wise random wavefront를 복원한다는 식의 문장은 금지한다.",
    ],
    { x: 1.0, y: 1.6, w: 5.2, h: 3.65, fontSize: 11.3 }
  );
  addCard(s, { x: 6.95, y: 1.15, w: 5.7, h: 4.75, fill: C.softBlue, line: "C7DDEB", title: "다음 검증 우선순위" });
  addBullets(
    s,
    [
      "bucket radius / detector plane sweep: 10, 25, 50, 100 μm와 detector distance를 같이 바꿔 민감도를 본다.",
      "thin-lens or real-lens baseline 추가: 100 m weak regime에서 'super-vacuum' 관측을 재검증한다.",
      "SMF/MMF coupling efficiency 또는 finite-NA collected power를 추가해 수신기 의미를 강화한다.",
      "tail statistics용 test set 확장: strong regime percentile claim은 더 큰 sample 수에서 다시 본다.",
    ],
    { x: 7.25, y: 1.6, w: 5.0, h: 3.0, fontSize: 11.3 }
  );
  addCard(s, { x: 7.25, y: 4.95, w: 4.95, h: 0.68, fill: C.softGreen, line: "CDE3D3" });
  s.addText("현재 결론: 'absolute bucket power 개선'과 'fade-tail 개선 관측'까지는 말할 수 있다.", {
    x: 7.45,
    y: 5.2,
    w: 4.55,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.green,
    margin: 0,
    align: "center",
  });
  addFooter(s, 11, "Stop line: overclaim 대신 next validation list를 남긴다");
  finalize(s);
}

[
  slide1,
  slide2,
  slide3,
  slide4,
  slide5,
  slide6,
  slide7,
  slide8,
  slide9,
  slide10,
  slide11,
].forEach((builder) => builder());

pptx.writeFile({ fileName: OUT }).then(() => {
  console.log(`Saved: ${OUT}`);
});
