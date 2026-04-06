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

const ROOT = path.resolve(__dirname, "..");
const R0403 = path.join(ROOT, "0403-combined-6strat-pitchrescale-cn2-5e14");
const OUT = path.join(__dirname, "0406-rp-vs-pib-seminar.pptx");

const SUMMARY = JSON.parse(
  fs.readFileSync(path.join(R0403, "19_cross_strategy_summary.json"), "utf8")
);
const WFRMS = JSON.parse(
  fs.readFileSync(
    path.join(R0403, "focal_raw_received_power", "10_piston_removed_wfrms.json"),
    "utf8"
  )
);

const IMG = {
  cross: path.join(R0403, "19_cross_strategy_received_power.png"),
  rawTriptych: path.join(
    R0403,
    "focal_raw_received_power",
    "06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png"
  ),
  rawHist: path.join(
    R0403,
    "focal_raw_received_power",
    "18_received_power_histogram.png"
  ),
  zernike: path.join(
    R0403,
    "focal_raw_received_power",
    "14_zernike_delta_per_mode.png"
  ),
  phaseMasks: path.join(
    R0403,
    "focal_raw_received_power",
    "15_phase_masks_5layers.png"
  ),
  hardTpHist: path.join(R0403, "pib_hard_tp", "18_received_power_histogram.png"),
  w2Hist: path.join(R0403, "focal_tp_pib_w2", "18_received_power_histogram.png"),
};

const C = {
  ink: "102033",
  navy: "12253A",
  blue: "275D8C",
  cyan: "5FA7C8",
  softBlue: "EAF3F8",
  pale: "F7FAFC",
  white: "FFFFFF",
  line: "D5E0E8",
  gray: "6A7C8F",
  green: "1F8A54",
  red: "B74A46",
  amber: "C57B20",
  violet: "6D5AB8",
};

const FONTS = {
  header: "DejaVu Serif",
  body: "DejaVu Sans",
  mono: "DejaVu Sans Mono",
};

function pct(x, digits = 1) {
  return `${x.toFixed(digits)}%`;
}

function num(x, digits = 2) {
  return x.toFixed(digits);
}

function gainVsTurbulent(name) {
  const base = SUMMARY.Turbulent.rp_10um_mean;
  const val = SUMMARY[name].rp_10um_mean;
  return ((val / base) - 1) * 100;
}

function gainRadius(name, radius) {
  const base = SUMMARY.Turbulent[`rp_${radius}um_mean`];
  const val = SUMMARY[name][`rp_${radius}um_mean`];
  return ((val / base) - 1) * 100;
}

function shadow() {
  return { type: "outer", color: "000000", blur: 3, offset: 1.5, angle: 45, opacity: 0.1 };
}

function addFrame(slide, title, subtitle = "", dark = false) {
  slide.background = { color: dark ? C.navy : C.pale };
  slide.addShape(SHAPES.RECTANGLE, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.08,
    line: { color: dark ? C.cyan : C.blue, transparency: 100 },
    fill: { color: dark ? C.cyan : C.blue },
  });
  slide.addText(title, {
    x: 0.55,
    y: 0.22,
    w: 9.6,
    h: 0.48,
    fontFace: FONTS.header,
    fontSize: dark ? 24 : 25,
    bold: true,
    color: dark ? C.white : C.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 10.25,
      y: 0.28,
      w: 2.55,
      h: 0.24,
      fontFace: FONTS.body,
      fontSize: 9.5,
      italic: true,
      color: dark ? "BFD7E5" : C.gray,
      align: "right",
      margin: 0,
    });
  }
  slide.addShape(SHAPES.LINE, {
    x: 0.55,
    y: 0.78,
    w: 12.25,
    h: 0,
    line: { color: dark ? "3C5770" : C.line, width: 1 },
  });
}

function addFooter(slide, text) {
  slide.addText(text, {
    x: 0.55,
    y: 7.05,
    w: 12.1,
    h: 0.16,
    fontFace: FONTS.body,
    fontSize: 7.5,
    color: C.gray,
    margin: 0,
  });
}

function addCard(slide, opts) {
  slide.addShape(SHAPES.ROUNDED_RECTANGLE, {
    x: opts.x,
    y: opts.y,
    w: opts.w,
    h: opts.h,
    rectRadius: 0.06,
    line: { color: opts.line || C.line, width: 1 },
    fill: { color: opts.fill || C.white },
    shadow: shadow(),
  });
  if (opts.title) {
    slide.addText(opts.title, {
      x: opts.x + 0.18,
      y: opts.y + 0.13,
      w: opts.w - 0.36,
      h: 0.26,
      fontFace: FONTS.body,
      fontSize: 11.5,
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
      options: {
        bullet: true,
        breakLine: idx !== items.length - 1,
      },
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
      fill: { color: "FFFFFF", transparency: 100 },
    });
  }
}

function finalize(slide) {
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function slide1() {
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape(SHAPES.RECTANGLE, {
    x: 0,
    y: 0,
    w: 0.14,
    h: 7.5,
    line: { color: C.cyan, transparency: 100 },
    fill: { color: C.cyan },
  });
  s.addText("Static D2NN for FSO", {
    x: 0.7,
    y: 0.95,
    w: 6.6,
    h: 0.7,
    fontFace: FONTS.header,
    fontSize: 28,
    bold: true,
    color: C.white,
    margin: 0,
  });
  s.addText("PIB보다 Received Power가 더 중요하다", {
    x: 0.7,
    y: 1.7,
    w: 6.9,
    h: 0.65,
    fontFace: FONTS.header,
    fontSize: 22,
    color: "CFE7F2",
    bold: true,
    margin: 0,
  });
  s.addShape(SHAPES.LINE, {
    x: 0.7,
    y: 2.55,
    w: 2.2,
    h: 0,
    line: { color: C.cyan, width: 2.5 },
  });
  addBullets(
    s,
    [
      "0403 loss sweep의 수치와 0404 physics report의 해석을 하나의 세미나 흐름으로 재구성",
      "핵심 질문: 10 um focal bucket 안 absolute received power가 실제로 증가했는가",
      "결론: TP-preserving losses만 baseline을 넘고, Raw RP loss가 +6.37%로 최고",
    ],
    { x: 0.85, y: 3.0, w: 6.4, h: 1.75, fontSize: 13.2, color: C.white }
  );
  addCard(s, { x: 8.35, y: 1.15, w: 4.0, h: 3.75, fill: "173149", line: "2D516C" });
  s.addText("Core Numbers", {
    x: 8.35,
    y: 1.18,
    w: 3.9,
    h: 0.28,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.cyan,
    margin: 0,
  });
  const metrics = [
    ["Vacuum PIB@10um", pct(SUMMARY.Vacuum.pib_10um, 1)],
    ["Turbulent PIB@10um", pct(SUMMARY.Turbulent.pib_10um, 1)],
    ["Vacuum RP@10um", num(SUMMARY.Vacuum.rp_10um_mean, 2)],
    ["Turbulent RP@10um", num(SUMMARY.Turbulent.rp_10um_mean, 2)],
    ["Best ΔRP", `+${num(gainVsTurbulent("Raw RP"), 2)}%`],
  ];
  metrics.forEach((row, i) => {
    const y = 1.72 + i * 0.5;
    s.addText(row[0], {
      x: 8.35,
      y,
      w: 2.2,
      h: 0.2,
      fontFace: FONTS.body,
      fontSize: 11,
      color: "BFD7E5",
      margin: 0,
    });
    s.addText(row[1], {
      x: 10.6,
      y: y - 0.03,
      w: 1.45,
      h: 0.26,
      fontFace: FONTS.body,
      fontSize: 15,
      bold: true,
      color: i === 4 ? "B9FBC0" : C.white,
      align: "right",
      margin: 0,
    });
  });
  s.addText("Sources: 0403 cross-strategy summary, 0404 loss strategy report", {
    x: 0.7,
    y: 6.88,
    w: 11.8,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 8,
    color: C.gray,
    margin: 0,
  });
  finalize(s);
}

function slide2() {
  const s = pptx.addSlide();
  addFrame(s, "Question and Setup", "0403/0404 seminar");
  addCard(s, { x: 0.6, y: 1.05, w: 6.0, h: 5.35 });
  s.addText("System", {
    x: 0.85,
    y: 1.2,
    w: 1.0,
    h: 0.25,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  const labels = [
    "TX",
    "1 km turbulence",
    "150 mm telescope",
    "5-layer D2NN",
    "f = 4.5 mm lens",
    "10 um bucket",
  ];
  labels.forEach((label, i) => {
    const x = 0.98 + i * 0.96;
    s.addShape(SHAPES.ROUNDED_RECTANGLE, {
      x,
      y: 1.8,
      w: 0.72,
      h: 0.55,
      rectRadius: 0.04,
      line: { color: i === labels.length - 1 ? C.green : C.line, width: 1 },
      fill: { color: i === labels.length - 1 ? "E7F7EE" : C.white },
    });
    s.addText(label, {
      x: x + 0.05,
      y: 1.95,
        w: 0.7,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 8.7,
      color: C.ink,
      bold: i === labels.length - 1,
      align: "center",
      margin: 0,
    });
    if (i < labels.length - 1) {
      s.addShape(SHAPES.CHEVRON, {
        x: x + 0.77,
        y: 1.99,
        w: 0.11,
        h: 0.12,
        line: { color: C.cyan, transparency: 100 },
        fill: { color: C.cyan },
      });
    }
  });
  addBullets(
    s,
    [
      "Wavelength 1.55 um, distance 1 km, Cn^2 = 5e-14",
      "Detector proxy는 focal plane 10 um fiber bucket",
      "Vacuum PIB@10um = 95.98%, turbulent PIB@10um = 77.11%",
      "Vacuum RP@10um = 277.17, turbulent RP@10um = 212.04",
      "따라서 회복해야 할 absolute headroom은 65.12 units",
    ],
    { x: 0.9, y: 2.85, w: 5.2, h: 2.9, fontSize: 12 }
  );
  addCard(s, { x: 6.85, y: 1.05, w: 5.85, h: 5.35, fill: C.softBlue, line: "C6DCE8" });
  s.addText("Working Question", {
    x: 7.1,
    y: 1.2,
    w: 2.0,
    h: 0.28,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  s.addText("정적 phase-only D2NN가 sample-wise wavefront를 복원하지 못하더라도,\nensemble level에서 central bucket 안 absolute power를 조금 더 밀어 넣을 수 있는가?", {
    x: 7.1,
    y: 1.65,
    w: 5.05,
    h: 1.1,
    fontFace: FONTS.header,
    fontSize: 18,
    color: C.ink,
    bold: true,
    margin: 0,
  });
  s.addText("이번 발표는 'PIB가 올라갔다'가 아니라 'RP가 실제로 늘었는가'를 기준으로 모든 loss를 다시 읽는다.", {
    x: 7.1,
    y: 3.05,
    w: 5.0,
    h: 0.8,
    fontFace: FONTS.body,
    fontSize: 13,
    color: C.ink,
    margin: 0,
  });
  addCard(s, { x: 7.1, y: 4.25, w: 4.95, h: 1.35, fill: C.white });
  s.addText("Main Thesis", {
    x: 7.28,
    y: 4.42,
    w: 1.5,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.green,
    margin: 0,
  });
  s.addText("Normalized PIB alone can invert the ranking.\nReceived Power plus Throughput gives the physically honest ordering.", {
    x: 7.28,
    y: 4.7,
    w: 4.5,
    h: 0.65,
    fontFace: FONTS.body,
    fontSize: 12,
    color: C.ink,
    margin: 0,
  });
  addFooter(s, "Source: 0403 cross-strategy summary JSON and 0404 system-parameter section");
  finalize(s);
}

function slide3() {
  const s = pptx.addSlide();
  addFrame(s, "Why PIB Can Mislead", "metric definitions");
  const cards = [
    { x: 0.65, title: "PIB", fill: "EFF6FB", line: "C8DCE8" },
    { x: 4.45, title: "Received Power", fill: "EEF8F1", line: "CFE4D5" },
    { x: 8.25, title: "Throughput", fill: "FFF5E9", line: "E9D6B7" },
  ];
  cards.forEach((card) => addCard(s, { ...card, y: 1.2, w: 3.25, h: 2.0, titleColor: C.ink }));
  s.addText("Bucket energy / total energy\nratio metric", {
    x: 0.88,
    y: 1.7,
    w: 2.8,
    h: 0.5,
    fontFace: FONTS.header,
    fontSize: 18,
    color: C.blue,
    align: "center",
    margin: 0,
  });
  s.addText("Absolute energy inside\n10 um focal bucket", {
    x: 4.68,
    y: 1.7,
    w: 2.8,
    h: 0.5,
    fontFace: FONTS.header,
    fontSize: 18,
    color: C.green,
    align: "center",
    margin: 0,
  });
  s.addText("Output energy / input energy\npreservation ratio", {
    x: 8.48,
    y: 1.7,
    w: 2.8,
    h: 0.5,
    fontFace: FONTS.header,
    fontSize: 18,
    color: C.amber,
    align: "center",
    margin: 0,
  });
  addCard(s, { x: 1.1, y: 3.7, w: 11.1, h: 1.55, fill: C.white });
  s.addText("RP ≈ PIB × TP × E_in", {
    x: 3.75,
    y: 4.0,
    w: 4.0,
    h: 0.35,
    fontFace: FONTS.mono,
    fontSize: 24,
    bold: true,
    color: C.red,
    align: "center",
    margin: 0,
  });
  s.addText("PIB-only loss는 denominator를 줄이는 방향으로도 숫자를 올릴 수 있다. 즉 '더 잘 모았다'가 아니라 '나머지를 더 많이 버렸다'가 가능하다.", {
    x: 1.45,
    y: 4.48,
    w: 10.4,
    h: 0.42,
    fontFace: FONTS.body,
    fontSize: 13,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addCard(s, { x: 1.65, y: 5.65, w: 10.1, h: 0.9, fill: "FFF3F0", line: "E5C5BF" });
  s.addText("0404의 결론: TP를 명시적으로 묶지 않으면 PIB gain은 catastrophic throughput collapse를 숨길 수 있다.", {
    x: 1.9,
    y: 5.93,
    w: 9.6,
    h: 0.28,
    fontFace: FONTS.body,
    fontSize: 13,
    bold: true,
    color: C.red,
    align: "center",
    margin: 0,
  });
  addFooter(s, "Source: 0404 'The Fundamental Problem: PIB vs Received Power'");
  finalize(s);
}

function slide4() {
  const s = pptx.addSlide();
  addFrame(s, "0403 Loss Space", "same data, different objectives");
  const rows = [
    ["Strategy", "Core objective", "TP", "PIB10", "RP10"],
    ["AbsBucket+CO", "absolute bucket + CO", pct(SUMMARY["AbsBucket+CO"].throughput * 100, 2), pct(SUMMARY["AbsBucket+CO"].pib_10um, 2), num(SUMMARY["AbsBucket+CO"].rp_10um_mean, 2)],
    ["PIB+HardTP", "PIB + quadratic TP penalty", pct(SUMMARY["PIB+HardTP(10)"].throughput * 100, 2), pct(SUMMARY["PIB+HardTP(10)"].pib_10um, 2), num(SUMMARY["PIB+HardTP(10)"].rp_10um_mean, 2)],
    ["TP-PIB w=0.5", "PIB + weak TP penalty", pct(SUMMARY["TP-PIB w=0.5"].throughput * 100, 2), pct(SUMMARY["TP-PIB w=0.5"].pib_10um, 2), num(SUMMARY["TP-PIB w=0.5"].rp_10um_mean, 2)],
    ["TP-PIB w=2.0", "PIB + strong TP penalty", pct(SUMMARY["TP-PIB w=2.0"].throughput * 100, 2), pct(SUMMARY["TP-PIB w=2.0"].pib_10um, 2), num(SUMMARY["TP-PIB w=2.0"].rp_10um_mean, 2)],
    ["Raw RP", "-log(E_b / E_in)", pct(SUMMARY["Raw RP"].throughput * 100, 2), pct(SUMMARY["Raw RP"].pib_10um, 2), num(SUMMARY["Raw RP"].rp_10um_mean, 2)],
  ];
  s.addTable(rows, {
    x: 0.75,
    y: 1.2,
    w: 8.15,
    h: 3.5,
    colW: [1.65, 3.35, 0.95, 0.95, 1.0],
    rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4],
    fontFace: FONTS.body,
    fontSize: 10.5,
    border: { pt: 0.5, color: C.line },
    fill: C.white,
    color: C.ink,
    autoPage: false,
  });
  addCard(s, { x: 9.2, y: 1.2, w: 3.35, h: 3.5, fill: C.softBlue, line: "C7DCE8" });
  s.addText("Interpretation", {
    x: 9.42,
    y: 1.36,
    w: 1.5,
    h: 0.22,
    fontFace: FONTS.body,
    fontSize: 11.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  addBullets(
    s,
    [
      "0403에서는 optical setup과 data가 동일하고 loss만 바뀐다.",
      "따라서 성능 차이는 metric design의 효과로 직접 읽을 수 있다.",
      "모든 successful strategy는 TP를 97% 이상으로 유지한다.",
      "Raw RP가 가장 직접적인 objective이고 실제로 최고 RP10을 만든다.",
    ],
    { x: 9.42, y: 1.75, w: 2.7, h: 2.45, fontSize: 11 }
  );
  addCard(s, { x: 0.95, y: 5.2, w: 11.85, h: 0.88, fill: C.white });
  s.addText("같은 80%대 PIB라도 RP는 223.78~225.54 범위로 갈린다. 0403의 포인트는 'PIB가 아니라 RP 기준으로 다시 정렬해야 한다'는 점이다.", {
    x: 1.2,
    y: 5.5,
    w: 11.35,
    h: 0.24,
    fontFace: FONTS.body,
    fontSize: 12.5,
    bold: true,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(s, "Source: 0403 cross-strategy summary JSON");
  finalize(s);
}

function slide5() {
  const s = pptx.addSlide();
  addFrame(s, "Cross-Strategy Result", "0403 main result");
  placeImage(s, IMG.cross, { x: 0.65, y: 1.1, w: 7.3, h: 5.5 }, true);
  addCard(s, { x: 8.2, y: 1.1, w: 4.45, h: 5.5, fill: C.white });
  s.addText("Headline", {
    x: 8.45,
    y: 1.28,
    w: 1.3,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  addBullets(
    s,
    [
      `Raw RP is best: RP10 ${num(SUMMARY["Raw RP"].rp_10um_mean, 2)} (${gainVsTurbulent("Raw RP").toFixed(2)}% gain).`,
      `PIB+HardTP and TP-PIB w=2.0 are near-tied at ${gainVsTurbulent("PIB+HardTP(10)").toFixed(2)}% and ${gainVsTurbulent("TP-PIB w=2.0").toFixed(2)}%.`,
      `TP-PIB w=0.5 has the highest PIB10 (${pct(SUMMARY["TP-PIB w=0.5"].pib_10um, 2)}) but the lowest RP gain.`,
      "TP-preserving objectives all exceed the turbulent baseline.",
    ],
    { x: 8.45, y: 1.7, w: 3.7, h: 2.2, fontSize: 11.3 }
  );
  s.addChart(pptx.charts.BAR, [
    {
      name: "ΔRP vs turbulent (%)",
      labels: ["AbsBkt+CO", "PIB+HardTP", "TP-PIB 0.5", "TP-PIB 2.0", "Raw RP"],
      values: [
        Number(gainVsTurbulent("AbsBucket+CO").toFixed(2)),
        Number(gainVsTurbulent("PIB+HardTP(10)").toFixed(2)),
        Number(gainVsTurbulent("TP-PIB w=0.5").toFixed(2)),
        Number(gainVsTurbulent("TP-PIB w=2.0").toFixed(2)),
        Number(gainVsTurbulent("Raw RP").toFixed(2)),
      ],
    },
  ], {
    x: 8.38,
    y: 4.28,
    w: 3.95,
    h: 1.95,
    barDir: "col",
    catAxisLabelFontFace: FONTS.body,
    catAxisLabelFontSize: 8,
    catAxisLabelColor: C.gray,
    valAxisLabelFontFace: FONTS.body,
    valAxisLabelFontSize: 8,
    valAxisLabelColor: C.gray,
    valGridLine: { color: C.line, size: 0.5 },
    catGridLine: { style: "none" },
    chartArea: { fill: { color: C.white } },
    chartColors: [C.green],
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelColor: C.ink,
    dataLabelFontSize: 8,
    showLegend: false,
    showTitle: false,
    valAxisMinVal: 0,
    valAxisMaxVal: 7,
  });
  addFooter(s, "Figure: 19_cross_strategy_received_power.png");
  finalize(s);
}

function slide6() {
  const s = pptx.addSlide();
  addFrame(s, "Rank by RP, Not by PIB", "metric ordering");
  s.addChart(pptx.charts.BAR, [
    {
      name: "PIB@10um (%)",
      labels: ["Turbulent", "AbsBkt+CO", "PIB+HardTP", "TP-PIB 0.5", "TP-PIB 2.0", "Raw RP"],
      values: [
        Number(SUMMARY.Turbulent.pib_10um.toFixed(2)),
        Number(SUMMARY["AbsBucket+CO"].pib_10um.toFixed(2)),
        Number(SUMMARY["PIB+HardTP(10)"].pib_10um.toFixed(2)),
        Number(SUMMARY["TP-PIB w=0.5"].pib_10um.toFixed(2)),
        Number(SUMMARY["TP-PIB w=2.0"].pib_10um.toFixed(2)),
        Number(SUMMARY["Raw RP"].pib_10um.toFixed(2)),
      ],
    },
  ], {
    x: 0.65,
    y: 1.25,
    w: 5.95,
    h: 4.35,
    barDir: "col",
    chartColors: [C.blue],
    chartArea: { fill: { color: C.white } },
    showLegend: false,
    showTitle: true,
    title: "PIB@10um",
    titleFontFace: FONTS.body,
    titleFontSize: 11,
    titleColor: C.ink,
    catAxisLabelFontFace: FONTS.body,
    catAxisLabelFontSize: 8.5,
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valAxisMinVal: 74,
    valAxisMaxVal: 83,
    valGridLine: { color: C.line, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 8,
  });
  s.addChart(pptx.charts.BAR, [
    {
      name: "RP@10um",
      labels: ["Turbulent", "AbsBkt+CO", "PIB+HardTP", "TP-PIB 0.5", "TP-PIB 2.0", "Raw RP"],
      values: [
        Number(SUMMARY.Turbulent.rp_10um_mean.toFixed(2)),
        Number(SUMMARY["AbsBucket+CO"].rp_10um_mean.toFixed(2)),
        Number(SUMMARY["PIB+HardTP(10)"].rp_10um_mean.toFixed(2)),
        Number(SUMMARY["TP-PIB w=0.5"].rp_10um_mean.toFixed(2)),
        Number(SUMMARY["TP-PIB w=2.0"].rp_10um_mean.toFixed(2)),
        Number(SUMMARY["Raw RP"].rp_10um_mean.toFixed(2)),
      ],
    },
  ], {
    x: 6.78,
    y: 1.25,
    w: 5.9,
    h: 4.35,
    barDir: "col",
    chartColors: [C.green],
    chartArea: { fill: { color: C.white } },
    showLegend: false,
    showTitle: true,
    title: "RP@10um",
    titleFontFace: FONTS.body,
    titleFontSize: 11,
    titleColor: C.ink,
    catAxisLabelFontFace: FONTS.body,
    catAxisLabelFontSize: 8.5,
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valAxisMinVal: 210,
    valAxisMaxVal: 227,
    valGridLine: { color: C.line, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 8,
  });
  addCard(s, { x: 1.3, y: 6.05, w: 10.75, h: 0.62, fill: C.white });
  s.addText("왼쪽 그래프만 보면 TP-PIB w=0.5가 가장 좋아 보이지만, 오른쪽 RP ordering에서는 Raw RP가 최고다. 이것이 metric inversion이다.", {
    x: 1.55,
    y: 6.25,
    w: 10.2,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(s, "Source: 0403 summary JSON; same data, same optics, only objective changes");
  finalize(s);
}

function slide7() {
  const s = pptx.addSlide();
  addFrame(s, "What the Best Strategy Actually Does", "Raw RP focal pattern");
  placeImage(s, IMG.rawTriptych, { x: 0.62, y: 1.05, w: 8.0, h: 5.75 }, true);
  addCard(s, { x: 8.9, y: 1.05, w: 3.75, h: 5.75, fill: C.white });
  addBullets(
    s,
    [
      `Vacuum sets the ceiling: PIB10 ${pct(SUMMARY.Vacuum.pib_10um, 2)}, RP10 ${num(SUMMARY.Vacuum.rp_10um_mean, 2)}.`,
      `Turbulent baseline sits at PIB10 ${pct(SUMMARY.Turbulent.pib_10um, 2)}, RP10 ${num(SUMMARY.Turbulent.rp_10um_mean, 2)}.`,
      `Raw RP reaches PIB10 ${pct(SUMMARY["Raw RP"].pib_10um, 2)} and RP10 ${num(SUMMARY["Raw RP"].rp_10um_mean, 2)}.`,
      "핵심 변화는 vacuum restoration이 아니라 central bucket 안으로 에너지를 조금 더 밀어 넣는 focal redistribution이다.",
    ],
    { x: 9.15, y: 1.45, w: 3.1, h: 2.75, fontSize: 11.2 }
  );
  addCard(s, { x: 9.15, y: 4.55, w: 3.0, h: 1.35, fill: C.softBlue, line: "C7DCE8" });
  s.addText("Interpretation", {
    x: 9.35,
    y: 4.73,
    w: 1.15,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 10.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  s.addText("Best strategy도 vacuum에 가깝게 돌아가진 않는다. modest한 +6% gain은 오히려 물리적으로 더 정직한 결과다.", {
    x: 9.35,
    y: 5.03,
    w: 2.6,
    h: 0.55,
    fontFace: FONTS.body,
    fontSize: 11.2,
    color: C.ink,
    margin: 0,
  });
  addFooter(s, "Figure: focal_raw_received_power/06_fig1_focal_plane_vacuum_vs_turbulent_vs_d2nn.png");
  finalize(s);
}

function slide8() {
  const s = pptx.addSlide();
  addFrame(s, "Improvement Is Distributional", "Raw RP histogram and bucket-radius gains");
  placeImage(s, IMG.rawHist, { x: 0.68, y: 1.1, w: 7.25, h: 5.45 }, true);
  s.addChart(pptx.charts.BAR, [
    {
      name: "Gain vs turbulent (%)",
      labels: ["5um", "10um", "25um", "50um"],
      values: [
        Number(gainRadius("Raw RP", 5).toFixed(2)),
        Number(gainRadius("Raw RP", 10).toFixed(2)),
        Number(gainRadius("Raw RP", 25).toFixed(2)),
        Number(gainRadius("Raw RP", 50).toFixed(2)),
      ],
    },
  ], {
    x: 8.25,
    y: 1.35,
    w: 4.0,
    h: 2.75,
    barDir: "col",
    chartColors: [C.violet],
    showLegend: false,
    showTitle: true,
    title: "Raw RP gain by bucket radius",
    titleFontFace: FONTS.body,
    titleFontSize: 10.5,
    titleColor: C.ink,
    chartArea: { fill: { color: C.white } },
    catAxisLabelFontFace: FONTS.body,
    catAxisLabelFontSize: 9,
    catAxisLabelColor: C.gray,
    valAxisLabelColor: C.gray,
    valAxisMinVal: 0,
    valAxisMaxVal: 12,
    valGridLine: { color: C.line, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 8,
  });
  addCard(s, { x: 8.25, y: 4.45, w: 4.0, h: 1.65, fill: C.white });
  addBullets(
    s,
    [
      `5um improvement is largest: +${num(gainRadius("Raw RP", 5), 2)}%.`,
      `10um objective improves by +${num(gainRadius("Raw RP", 10), 2)}%.`,
      `25um and 50um gains shrink to +${num(gainRadius("Raw RP", 25), 2)}% and +${num(gainRadius("Raw RP", 50), 2)}%.`,
      "즉 total energy increase가 아니라 core tightening이 주효하다.",
    ],
    { x: 8.48, y: 4.7, w: 3.4, h: 1.1, fontSize: 11 }
  );
  addFooter(s, "Figure: focal_raw_received_power/18_received_power_histogram.png");
  finalize(s);
}

function slide9() {
  const s = pptx.addSlide();
  addFrame(s, "Why This Is Not Wavefront Correction", "WF RMS and Zernike evidence");
  placeImage(s, IMG.zernike, { x: 0.7, y: 1.25, w: 6.95, h: 4.95 }, true);
  addCard(s, { x: 7.95, y: 1.25, w: 4.55, h: 4.95, fill: C.white });
  const raw = WFRMS.raw_align_global_phase;
  const ptt = WFRMS.piston_tiptilt_removed;
  const rows = [
    ["Metric", "Turbulent", "D2NN"],
    ["WFRMS global-phase", num(raw.turbulent.mean_nm, 2), num(raw.d2nn.mean_nm, 2)],
    ["WFRMS piston-removed", num(WFRMS.piston_removed.turbulent.mean_nm, 2), num(WFRMS.piston_removed.d2nn.mean_nm, 2)],
    ["WFRMS tip/tilt removed", num(ptt.turbulent.mean_nm, 2), num(ptt.d2nn.mean_nm, 2)],
  ];
  s.addTable(rows, {
    x: 8.18,
    y: 1.45,
    w: 4.05,
    colW: [1.55, 1.15, 1.15],
    rowH: [0.38, 0.36, 0.36, 0.36],
    fontFace: FONTS.body,
    fontSize: 9.5,
    border: { pt: 0.5, color: C.line },
    fill: C.white,
    color: C.ink,
    autoPage: false,
  });
  addBullets(
    s,
    [
      "Global-phase aligned WFRMS is 368.94 nm -> 368.63 nm, essentially unchanged.",
      "Tip/tilt removed RMS even increases slightly.",
      "Zernike delta per mode stays near zero rather than showing a correction signature.",
      "따라서 이 장치는 adaptive corrector가 아니라 statistical mode filter로 읽는 편이 정확하다.",
    ],
    { x: 8.18, y: 3.35, w: 3.75, h: 2.2, fontSize: 10.8 }
  );
  addFooter(s, "Sources: focal_raw_received_power/10_piston_removed_wfrms.json and /14_zernike_delta_per_mode.png");
  finalize(s);
}

function slide10() {
  const s = pptx.addSlide();
  addFrame(s, "Physical Mechanism", "TP-preserving focal energy redistribution");
  placeImage(s, IMG.phaseMasks, { x: 0.7, y: 1.2, w: 6.25, h: 5.25 }, true);
  addCard(s, { x: 7.3, y: 1.2, w: 5.35, h: 5.25, fill: C.white });
  s.addText("What the masks suggest", {
    x: 7.58,
    y: 1.42,
    w: 2.3,
    h: 0.22,
    fontFace: FONTS.body,
    fontSize: 12,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  addBullets(
    s,
    [
      "TP-preserving strategies learn mostly smooth, low-spatial-frequency phase profiles.",
      "Sharp grating signatures that would support strong scattering are not dominant.",
      "이 패턴은 TP 98%대 유지와 잘 맞고, aggressive energy dumping과는 다르다.",
      "0404의 해석처럼 weak Fresnel-lens-like redistribution 또는 statistical mode filtering에 가깝다.",
    ],
    { x: 7.58, y: 1.8, w: 4.55, h: 2.6, fontSize: 11.3 }
  );
  addCard(s, { x: 7.58, y: 4.65, w: 4.55, h: 1.2, fill: C.softBlue, line: "C7DCE8" });
  s.addText("Operational reading", {
    x: 7.8,
    y: 4.85,
    w: 1.6,
    h: 0.16,
    fontFace: FONTS.body,
    fontSize: 10.5,
    bold: true,
    color: C.blue,
    margin: 0,
  });
  s.addText("Wavefront conjugation보다 focal core와 halo 사이의 ensemble-average energy reshaping으로 보는 편이 맞다.", {
    x: 7.8,
    y: 5.12,
    w: 4.1,
    h: 0.42,
    fontFace: FONTS.body,
    fontSize: 11,
    color: C.ink,
    margin: 0,
  });
  addFooter(s, "Figure: focal_raw_received_power/15_phase_masks_5layers.png");
  finalize(s);
}

function slide11() {
  const s = pptx.addSlide();
  addFrame(s, "0404 Interpretation", "how PIB-only objectives fail");
  const boxes = [
    { x: 0.85, y: 1.55, w: 3.4, h: 3.7, title: "PIB-only loophole", fill: "FFF3F0", line: "E5C5BF", titleColor: C.red },
    { x: 4.97, y: 1.55, w: 3.4, h: 3.7, title: "TP-aware closure", fill: "EEF8F1", line: "CFE4D5", titleColor: C.green },
    { x: 9.09, y: 1.55, w: 3.4, h: 3.7, title: "Observed attractor", fill: "EFF6FB", line: "C8DCE8", titleColor: C.blue },
  ];
  boxes.forEach((b) => addCard(s, b));
  addBullets(s, [
    "Phase-only layers are pointwise unitary, but cascaded propagation can still redistribute energy angularly.",
    "Finite computational window plus diffraction lets the model scatter aberrated energy out of the effective passband.",
    "Then PIB rises because the denominator shrinks, even if RP falls."
  ], { x: 1.08, y: 2.0, w: 2.9, h: 2.6, fontSize: 10.8 });
  addBullets(s, [
    "Absolute-bucket and TP-penalized objectives make scattering expensive.",
    "Now the optimizer must keep energy while improving concentration.",
    "That is why successful strategies all stay near TP ≈ 0.98–0.99."
  ], { x: 5.2, y: 2.0, w: 2.9, h: 2.6, fontSize: 10.8 });
  addBullets(s, [
    "Different TP-aware formulas converge to roughly the same +6% RP regime.",
    "This suggests a physical limit of the static phase-only architecture in this turbulence regime.",
    "Objective shape matters less once the TP operating point is fixed."
  ], { x: 9.32, y: 2.0, w: 2.9, h: 2.6, fontSize: 10.8 });
  addCard(s, { x: 2.05, y: 5.7, w: 9.25, h: 0.72, fill: C.white });
  s.addText("0404의 요약: PIB는 단독 metric으로 쓰면 misleading하고, direct RP objective 또는 explicit TP constraint가 필요하다.", {
    x: 2.3,
    y: 5.95,
    w: 8.75,
    h: 0.18,
    fontFace: FONTS.body,
    fontSize: 12.5,
    bold: true,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(s, "Source: 0404 sections on throughput catastrophe, TP-aware losses, and conclusions");
  finalize(s);
}

function slide12() {
  const s = pptx.addSlide();
  addFrame(s, "How To Phrase the Claim", "what to say, what to avoid");
  addCard(s, { x: 0.85, y: 1.35, w: 5.4, h: 4.95, fill: "EEF8F1", line: "CFE4D5", title: "Say This", titleColor: C.green });
  addCard(s, { x: 6.95, y: 1.35, w: 5.4, h: 4.95, fill: "FFF3F0", line: "E5C5BF", title: "Avoid This", titleColor: C.red });
  addBullets(s, [
    "Static phase-only D2NN improves absolute received power inside a 10 um focal bucket by 5.5–6.4% over the turbulent baseline.",
    "The successful regime is TP-preserving: throughput stays around 98% while RP rises.",
    "The mechanism is focal energy redistribution or statistical mode filtering, not sample-wise phase conjugation.",
    "Raw RP loss is the simplest and strongest objective among tested TP-aware losses."
  ], { x: 1.12, y: 1.85, w: 4.85, h: 3.75, fontSize: 11.6 });
  addBullets(s, [
    "Do not call this wavefront recovery, vacuum restoration, or full aberration correction.",
    "Do not use PIB gain alone as evidence of better coupling.",
    "Do not imply that static masks adapt to random time-varying turbulence sample by sample.",
    "Do not hide throughput when comparing objective functions."
  ], { x: 7.22, y: 1.85, w: 4.85, h: 3.75, fontSize: 11.6 });
  addFooter(s, "Claim wording aligned to 0404 conclusions and 0403 wavefront evidence");
  finalize(s);
}

function slide13() {
  const s = pptx.addSlide();
  addFrame(s, "Seminar Takeaways", "four numbers to remember");
  const cards = [
    { x: 0.9, y: 1.5, w: 2.75, h: 2.25, title: "Best ΔRP", value: `+${num(gainVsTurbulent("Raw RP"), 2)}%`, color: C.green, note: "Raw RP is best overall" },
    { x: 3.95, y: 1.5, w: 2.75, h: 2.25, title: "Operating TP", value: pct(SUMMARY["Raw RP"].throughput * 100, 2), color: C.blue, note: "Successful losses stay near 98%" },
    { x: 7.0, y: 1.5, w: 2.75, h: 2.25, title: "Best PIB10", value: pct(SUMMARY["Raw RP"].pib_10um, 2), color: C.violet, note: "Still far below vacuum" },
    { x: 10.05, y: 1.5, w: 2.35, h: 2.25, title: "Remaining Gap", value: `${num(SUMMARY.Vacuum.pib_10um - SUMMARY["Raw RP"].pib_10um, 2)}p`, color: C.red, note: "Static limit remains large" },
  ];
  cards.forEach((card) => {
    addCard(s, { x: card.x, y: card.y, w: card.w, h: card.h, fill: C.white });
    s.addText(card.title, {
      x: card.x + 0.18,
      y: card.y + 0.18,
      w: card.w - 0.36,
      h: 0.18,
      fontFace: FONTS.body,
      fontSize: 10.5,
      bold: true,
      color: C.gray,
      align: "center",
      margin: 0,
    });
    s.addText(card.value, {
      x: card.x + 0.18,
      y: card.y + 0.72,
      w: card.w - 0.36,
      h: 0.45,
      fontFace: FONTS.header,
      fontSize: 24,
      bold: true,
      color: card.color,
      align: "center",
      margin: 0,
    });
    s.addText(card.note, {
      x: card.x + 0.18,
      y: card.y + 1.5,
      w: card.w - 0.36,
      h: 0.35,
      fontFace: FONTS.body,
      fontSize: 10.5,
      color: C.ink,
      align: "center",
      margin: 0,
    });
  });
  addCard(s, { x: 1.15, y: 4.45, w: 11.15, h: 1.4, fill: C.softBlue, line: "C7DCE8" });
  addBullets(s, [
    "Metric choice changes the ranking: PIB alone can invert the story.",
    "TP-preserving losses converge to a modest but real +6% RP regime.",
    "The observed gain is compatible with energy redistribution, not wavefront correction.",
    "The large remaining gap to vacuum is itself a result: static phase-only D2NN has a structural ceiling here."
  ], { x: 1.45, y: 4.8, w: 10.55, h: 0.8, fontSize: 12 });
  addFooter(s, "Summary numbers from 0403 summary JSON; interpretation from 0404");
  finalize(s);
}

function slide14() {
  const s = pptx.addSlide();
  addFrame(s, "Next Experiments", "what to do after this seminar");
  const phases = [
    ["Seed robustness", "rerun Raw RP and PIB+HardTP with multiple random seeds and quantify RP10 mean±std"],
    ["Bucket sweep", "evaluate 5, 10, 25 um objectives to map where core-vs-halo redistribution helps most"],
    ["Stronger turbulence", "increase Cn^2 or deeper fades to check whether the +6% attractor survives"],
    ["Distance sweep", "connect to run_distance_sweep_after_0404.sh and map where static gain collapses"],
  ];
  phases.forEach((phase, i) => {
    const x = 1.0 + i * 3.0;
    addCard(s, { x, y: 2.0, w: 2.35, h: 3.6, fill: C.white });
    s.addShape(SHAPES.ROUNDED_RECTANGLE, {
      x: x + 0.72,
      y: 1.25,
      w: 0.9,
      h: 0.46,
      rectRadius: 0.05,
      line: { color: C.blue, transparency: 100 },
      fill: { color: C.blue },
    });
    s.addText(`Step ${i + 1}`, {
      x: x + 0.72,
      y: 1.38,
      w: 0.9,
      h: 0.16,
      fontFace: FONTS.body,
      fontSize: 9,
      bold: true,
      color: C.white,
      align: "center",
      margin: 0,
    });
    s.addText(phase[0], {
      x: x + 0.2,
      y: 2.25,
      w: 1.95,
      h: 0.3,
      fontFace: FONTS.header,
      fontSize: 15,
      bold: true,
      color: C.ink,
      align: "center",
      margin: 0,
    });
    s.addText(phase[1], {
      x: x + 0.18,
      y: 2.8,
      w: 1.98,
      h: 1.9,
      fontFace: FONTS.body,
      fontSize: 10.8,
      color: C.ink,
      margin: 0,
      align: "center",
      valign: "mid",
    });
    if (i < phases.length - 1) {
      s.addShape(SHAPES.CHEVRON, {
        x: x + 2.42,
        y: 3.65,
        w: 0.3,
        h: 0.22,
        line: { color: C.cyan, transparency: 100 },
        fill: { color: C.cyan },
      });
    }
  });
  addFooter(s, "Follow-up direction based on 0404 conclusions and existing distance-sweep script");
  finalize(s);
}

function slide15() {
  const s = pptx.addSlide();
  addFrame(s, "Appendix: Detailed Reference Metrics", "Q&A backup");
  const rows = [
    ["Case", "PIB10", "TP", "RP10", "RP5", "RP25", "RP50"],
    ["Vacuum", pct(SUMMARY.Vacuum.pib_10um, 2), pct(SUMMARY.Vacuum.throughput * 100, 2), num(SUMMARY.Vacuum.rp_10um_mean, 2), num(SUMMARY.Vacuum.rp_5um_mean, 2), num(SUMMARY.Vacuum.rp_25um_mean, 2), num(SUMMARY.Vacuum.rp_50um_mean, 2)],
    ["Turbulent", pct(SUMMARY.Turbulent.pib_10um, 2), pct(SUMMARY.Turbulent.throughput * 100, 2), num(SUMMARY.Turbulent.rp_10um_mean, 2), num(SUMMARY.Turbulent.rp_5um_mean, 2), num(SUMMARY.Turbulent.rp_25um_mean, 2), num(SUMMARY.Turbulent.rp_50um_mean, 2)],
    ["AbsBucket+CO", pct(SUMMARY["AbsBucket+CO"].pib_10um, 2), pct(SUMMARY["AbsBucket+CO"].throughput * 100, 2), num(SUMMARY["AbsBucket+CO"].rp_10um_mean, 2), num(SUMMARY["AbsBucket+CO"].rp_5um_mean, 2), num(SUMMARY["AbsBucket+CO"].rp_25um_mean, 2), num(SUMMARY["AbsBucket+CO"].rp_50um_mean, 2)],
    ["PIB+HardTP", pct(SUMMARY["PIB+HardTP(10)"].pib_10um, 2), pct(SUMMARY["PIB+HardTP(10)"].throughput * 100, 2), num(SUMMARY["PIB+HardTP(10)"].rp_10um_mean, 2), num(SUMMARY["PIB+HardTP(10)"].rp_5um_mean, 2), num(SUMMARY["PIB+HardTP(10)"].rp_25um_mean, 2), num(SUMMARY["PIB+HardTP(10)"].rp_50um_mean, 2)],
    ["TP-PIB w=2.0", pct(SUMMARY["TP-PIB w=2.0"].pib_10um, 2), pct(SUMMARY["TP-PIB w=2.0"].throughput * 100, 2), num(SUMMARY["TP-PIB w=2.0"].rp_10um_mean, 2), num(SUMMARY["TP-PIB w=2.0"].rp_5um_mean, 2), num(SUMMARY["TP-PIB w=2.0"].rp_25um_mean, 2), num(SUMMARY["TP-PIB w=2.0"].rp_50um_mean, 2)],
    ["Raw RP", pct(SUMMARY["Raw RP"].pib_10um, 2), pct(SUMMARY["Raw RP"].throughput * 100, 2), num(SUMMARY["Raw RP"].rp_10um_mean, 2), num(SUMMARY["Raw RP"].rp_5um_mean, 2), num(SUMMARY["Raw RP"].rp_25um_mean, 2), num(SUMMARY["Raw RP"].rp_50um_mean, 2)],
  ];
  s.addTable(rows, {
    x: 0.6,
    y: 1.2,
    w: 12.15,
    colW: [2.15, 1.25, 1.15, 1.35, 1.35, 1.35, 1.35],
    rowH: [0.42, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    fontFace: FONTS.body,
    fontSize: 10.2,
    border: { pt: 0.5, color: C.line },
    fill: C.white,
    color: C.ink,
    autoPage: false,
  });
  addFooter(s, "Source: 19_cross_strategy_summary.json");
  finalize(s);
}

function slide16() {
  const s = pptx.addSlide();
  addFrame(s, "Appendix: Constraint Equivalence", "two objectives, same operating point");
  placeImage(s, IMG.hardTpHist, { x: 0.75, y: 1.3, w: 5.8, h: 4.75 }, true);
  placeImage(s, IMG.w2Hist, { x: 6.78, y: 1.3, w: 5.8, h: 4.75 }, true);
  s.addText("PIB+HardTP", {
    x: 2.55,
    y: 0.98,
    w: 1.7,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.blue,
    align: "center",
    margin: 0,
  });
  s.addText("TP-PIB w=2.0", {
    x: 8.55,
    y: 0.98,
    w: 2.0,
    h: 0.2,
    fontFace: FONTS.body,
    fontSize: 11,
    bold: true,
    color: C.blue,
    align: "center",
    margin: 0,
  });
  addCard(s, { x: 1.55, y: 6.25, w: 10.25, h: 0.58, fill: C.white });
  s.addText("두 objective는 TP ≈ 0.99 근방에서 사실상 같은 constraint force를 만들고, 그래서 histogram과 mean performance가 거의 겹친다.", {
    x: 1.85,
    y: 6.45,
    w: 9.65,
    h: 0.16,
    fontFace: FONTS.body,
    fontSize: 11.8,
    color: C.ink,
    align: "center",
    margin: 0,
  });
  addFooter(s, "Figures: pib_hard_tp/18_received_power_histogram.png and focal_tp_pib_w2/18_received_power_histogram.png");
  finalize(s);
}

function build() {
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Codex";
  pptx.company = "OpenAI";
  pptx.subject = "D2NN received-power seminar deck";
  pptx.title = "Static D2NN for FSO: RP vs PIB Seminar";
  pptx.lang = "en-US";
  pptx.theme = {
    headFontFace: FONTS.header,
    bodyFontFace: FONTS.body,
    lang: "en-US",
  };

  slide1();
  slide2();
  slide3();
  slide4();
  slide5();
  slide6();
  slide7();
  slide8();
  slide9();
  slide10();
  slide11();
  slide12();
  slide13();
  slide14();
  slide15();
  slide16();

  return pptx.writeFile({ fileName: OUT });
}

build()
  .then(() => {
    console.log(`Wrote ${OUT}`);
  })
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
