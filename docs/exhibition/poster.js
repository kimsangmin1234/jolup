// 졸업작품전시회 패널 (A2 세로 42 x 59.4 cm) 두 버전 생성
//   node poster.js A   → 논문 구조 결과 중심
//   node poster.js B   → 실험 최고 성능(앙상블) 결과 중심
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const V = process.argv[2] || "A";
const OUT = process.argv[3] || `poster_${V}.pptx`;

const THEME = {
  name: "News Event Poster",
  headFontFace: "맑은 고딕",
  bodyFontFace: "맑은 고딕",
  colors: {
    dk1: "1F2633", lt1: "FFFFFF", dk2: "14233F", lt2: "EDF2F8",
    accent1: "14233F", accent2: "E3A12F", accent3: "1F7A8C", accent4: "C0392B",
    accent5: "2E8B57", accent6: "6B7686", hlink: "1F7A8C", folHlink: "6B7686",
  },
};
const HEX = { navy: "14233F", gold: "E3A12F", teal: "1F7A8C", red: "C0392B", green: "2E8B57", gray: "6B7686", tint: "EDF2F8", grid: "D9DEE6", text: "1F2633" };

const pres = new pptxgen();
pres.defineLayout({ name: "A2_PORTRAIT", width: 16.535, height: 23.386 });
pres.layout = "A2_PORTRAIT";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "뉴스 이벤트 기반 주식 가격 예측 모델";
pres.author = "김상민, 성우제";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "PANEL",
  background: { color: C.accent1 },
  objects: [],
});

const W = 16.535, H = 23.386, M = 0.6, G = 0.4;
const CW = (W - 2 * M - G) / 2, X2 = M + CW + G, FW = W - 2 * M;

pres.addSection({ title: "패널" });
const s = pres.addSlide({ masterName: "PANEL", sectionTitle: "패널" });

// ---------- 공통 도우미 ----------
let nObj = 0;
const nm = (k) => `${k}-${++nObj}`;
function text(t, o) { s.addText(t, { isTextBox: true, margin: 0, color: C.text1, valign: "top", objectName: nm("text"), ...o }); }
function card(x, y, w, h, num, title) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.18, fill: { color: C.background1 }, line: { color: C.background1 }, objectName: nm("card") });
  s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: y + 0.3, w: 0.68, h: 0.68, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: nm("badge") });
  text(String(num), { x: x + 0.35, y: y + 0.3, w: 0.68, h: 0.68, fontSize: 26, bold: true, color: C.accent1, align: "center", valign: "middle" });
  text(title, { x: x + 1.2, y: y + 0.3, w: w - 1.5, h: 0.68, fontSize: 32, bold: true, color: C.accent1, valign: "middle" });
  return { x: x + 0.4, y: y + 1.25, w: w - 0.8, h: h - 1.55 };
}
function box(x, y, w, h, t, fill, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: o.line || fill, width: 1.25 }, objectName: nm("node") });
  text(t, { x: x + 0.08, y, w: w - 0.16, h, fontSize: o.fs || 17, bold: !!o.bold, color: o.color || C.text1, align: "center", valign: "middle", lineSpacingMultiple: 0.95 });
}
function arrow(x1, y1, x2, y2, o = {}) {
  const flipV = (x2 - x1) * (y2 - y1) < 0 || (x1 === x2 && y2 < y1);
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV,
    line: { color: o.color || HEX.navy, width: o.width || 2, dashType: o.dash || "solid", endArrowType: o.noHead ? undefined : "triangle" },
    objectName: nm("arrow"),
  });
}
function bullets(items, o) {
  // items: [[굵은 머리, 설명], ...] 또는 문자열
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (typeof it === "string") runs.push({ text: it, options: { bullet: { indent: 18 }, breakLine: !last, paraSpaceAfter: o.gap ?? 8 } });
    else {
      runs.push({ text: it[0] + "  ", options: { bullet: { indent: 18 }, bold: true, color: C.accent1 } });
      runs.push({ text: it[1], options: { breakLine: !last, paraSpaceAfter: o.gap ?? 8 } });
    }
  });
  text(runs, { fontSize: 21, lineSpacingMultiple: 1.05, ...o });
}

// ---------- 머리말 ----------
text("뉴스 이벤트 기반 주식 가격 예측 모델", { x: M, y: 0.55, w: FW, h: 1.15, fontSize: 64, bold: true, color: C.background1, align: "center", valign: "middle" });
text("Stock Price Prediction Model Based on News Events", { x: M, y: 1.72, w: FW, h: 0.5, fontSize: 26, color: "C9D3E3", align: "center", valign: "middle" });
text([
  { text: "충북대학교 정보통신공학부", options: { color: C.background1 } },
  { text: "     지도교수  ", options: { color: C.accent2, bold: true } },
  { text: "유재수 교수님", options: { color: C.background1 } },
  { text: "     참여학생  ", options: { color: C.accent2, bold: true } },
  { text: "김상민 · 성우제", options: { color: C.background1 } },
], { x: M, y: 2.33, w: FW, h: 0.5, fontSize: 25, align: "center", valign: "middle" });

const Y1 = 3.2, HA = V === "B" ? 10.6 : 10.0, H1 = 4.75;

// ---------- 1. 연구 내용 ----------
let a = card(M, Y1, CW, H1, 1, "연구 내용");
bullets([
  ["필요성", "단기 주가는 기술적 지표만으로 설명하기 어려운 뉴스 이벤트에 크게 반응"],
  ["기존 연구", "LLM 감성 점수 또는 텍스트 임베딩 중 하나만 사용"],
  ["연구 내용", "뉴스의 의미·감성(주 신호)과 과거 30일 시장 상태(보조 신호)를 결합해 뉴스 발표 후 1일 초과수익률 예측"],
  ["검증", "실제 기업 보도자료 6.3만 건, 미래 정보 누수 없는 표본 외 평가"],
], { ...a, fontSize: 19, gap: 7 });

// ---------- 2. 원리 ----------
a = card(M, Y1 + H1 + 0.3, CW, HA - H1 - 0.3, 2, "원리");
const principles = [
  ["① LLM 기반 주 신호", "GPT-4o-mini 요약·감성 점수 + 요약문 임베딩(1536차원) → 완전연결층 → 512차원"],
  ["② TCN 기반 보조 벡터", "기술적 지표 9종 × 30일 → 인과적 TCN → 시점별 512차원"],
  ["③ 비대칭 교차 어텐션", "주 신호(Q)로 30개 보조 벡터(K·V)를 softmax 가중 평균"],
  ["④ 게이트 기반 잔차 결합", "h = p + σ(W[p; c]) ⊙ c, 보조 정보 반영 정도를 학습"],
  ["⑤ 등락률 예측", "MLP 512 → 256 → 64 → 1"],
];
if (V === "B") principles.push(["⑥ 확장·앙상블", "이벤트 27종·순위 손실 추가, 12개 모델 검증 가중 앙상블"]);
bullets(principles, { ...a, fontSize: V === "B" ? 17 : 18, gap: V === "B" ? 4 : 7 });

// ---------- 3. 구조 (졸업 논문 그림 1) ----------
a = card(X2, Y1, CW, HA, 3, V === "B" ? "구조 (논문 모델 + 앙상블)" : "구조");
{
  const strip = V === "B" ? 1.4 : 0;
  const Hd = a.h - strip, k = Hd / 9.45, X = (f) => a.x + f * a.w, Y = (u) => a.y + u * k;
  const W1 = "FFFFFF", GA = "EAF1FA", GB = "E8F3EA", O = "FBE8D0";
  const nb = (x1, y1, x2, y2, t, o = {}) => box(X(x1), Y(y1), X(x2) - X(x1), (y2 - y1) * k, t, o.fill || W1, { fs: 15.5, line: o.line || "8A94A6", ...o });
  const ln = (x1, y1, x2, y2, head = true) => arrow(X(x1), Y(y1), X(x2), Y(y2), { noHead: !head, width: 1.75 });
  // 그룹 바깥 틀
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(0.03), y: Y(0.75), w: X(0.58) - X(0.03), h: 3.3 * k, rectRadius: 0.08, fill: { color: GA }, line: { color: HEX.navy, width: 1.5 }, objectName: nm("group") });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: X(0.64), y: Y(1.95), w: X(0.97) - X(0.64), h: 2.1 * k, rectRadius: 0.08, fill: { color: GB }, line: { color: HEX.green, width: 1.5 }, objectName: nm("group") });
  text("1. LLM 기반 주 신호 생성", { x: X(0.03), y: Y(0.8), w: X(0.58) - X(0.03), h: 0.45 * k, fontSize: 16, bold: true, color: C.accent1, align: "center", valign: "middle" });
  text("2. TCN 기반\n보조 벡터 생성", { x: X(0.64), y: Y(2.0), w: X(0.97) - X(0.64), h: 0.7 * k, fontSize: 16, bold: true, color: C.accent1, align: "center", valign: "middle", lineSpacingMultiple: 0.95 });
  nb(0.16, 0, 0.48, 0.55, "이벤트 발생 (뉴스)", { line: HEX.navy });
  ln(0.32, 0.55, 0.32, 0.75);
  nb(0.21, 1.3, 0.41, 1.8, "LLM");
  ln(0.31, 1.8, 0.31, 1.97, false); ln(0.19, 1.97, 0.425, 1.97, false);
  ln(0.19, 1.97, 0.19, 2.15); ln(0.425, 1.97, 0.425, 2.15);
  nb(0.08, 2.15, 0.30, 2.65, "뉴스 요약"); nb(0.32, 2.15, 0.53, 2.65, "감성 점수");
  ln(0.19, 2.65, 0.19, 2.9); nb(0.08, 2.9, 0.30, 3.3, "임베딩 생성");
  ln(0.19, 3.3, 0.19, 3.5); ln(0.425, 2.65, 0.425, 3.5);
  nb(0.06, 3.5, 0.55, 3.9, "완전 연결층");
  nb(0.68, 2.75, 0.93, 3.15, "기술적 지표"); ln(0.805, 3.15, 0.805, 3.45); nb(0.68, 3.45, 0.93, 3.85, "TCN");
  ln(0.305, 4.05, 0.305, 4.3); ln(0.805, 4.05, 0.805, 4.3);
  nb(0.03, 4.3, 0.58, 4.9, "주 신호: 512차원 벡터", { line: HEX.navy, bold: true });
  nb(0.64, 4.3, 0.97, 4.9, "보조 신호:\n(T, 512)차원 벡터", { line: HEX.green, bold: true, fs: 14.5 });
  ln(0.305, 4.9, 0.305, 5.12, false); ln(0.805, 4.9, 0.805, 5.12, false); ln(0.305, 5.12, 0.805, 5.12, false);
  ln(0.55, 5.12, 0.55, 5.4);
  nb(0.19, 5.4, 0.91, 5.95, "3. 비대칭 교차 어텐션", { fill: O, line: HEX.gold, bold: true, fs: 17 });
  ln(0.55, 5.95, 0.55, 6.2); nb(0.30, 6.2, 0.80, 6.6, "보조 문맥 벡터: 512차원");
  ln(0.55, 6.6, 0.55, 6.85);
  nb(0.19, 6.85, 0.91, 7.4, "4. 게이트 기반 잔차 결합", { fill: O, line: HEX.gold, bold: true, fs: 17 });
  ln(0.03, 4.6, -0.03, 4.6, false); ln(-0.03, 4.6, -0.03, 7.125, false); ln(-0.03, 7.125, 0.19, 7.125);
  ln(0.55, 7.4, 0.55, 7.65); nb(0.30, 7.65, 0.80, 8.05, "통합 표현 벡터: 512차원");
  ln(0.55, 8.05, 0.55, 8.3);
  nb(0.19, 8.3, 0.91, 8.85, "5. 등락률 예측", { fill: O, line: HEX.gold, bold: true, fs: 17 });
  ln(0.55, 8.85, 0.55, 9.05);
  nb(0.34, 9.05, 0.76, 9.45, V === "B" ? "등락률 (개선 구조)" : "등락률", { fill: HEX.navy, line: HEX.navy, color: C.background1, bold: true });
  if (V === "B") {
    const ey = a.y + Hd + 0.15;
    text("최종 예측: 위 신경망을 포함한 12개 모델의 앙상블", { x: a.x, y: ey, w: a.w, h: 0.38, fontSize: 15, bold: true, color: C.accent1, valign: "middle" });
    const ens = [["신경망 8종", 1.5], ["LightGBM 4종", 1.55], ["Ridge", 0.95], ["가중 순위 앙상블", 1.85]];
    let ex = a.x; const by = ey + 0.48, bh2 = 0.7;
    ens.forEach(([t, w], i) => {
      box(ex, by, w, bh2, t, i === 3 ? HEX.navy : W1, { fs: 14.5, bold: i === 3, color: i === 3 ? C.background1 : C.text1, line: i === 3 ? HEX.navy : "8A94A6" });
      if (i < 2) text("+", { x: ex + w, y: by, w: 0.28, h: bh2, fontSize: 22, bold: true, color: C.accent1, align: "center", valign: "middle" });
      else if (i === 2) arrow(ex + w + 0.03, by + bh2 / 2, ex + w + 0.25, by + bh2 / 2);
      ex += w + 0.28;
    });
  }
}

// ---------- 아래 줄: 5. 성능평가 (왼쪽), 4. Source Program · 6. 결론 (오른쪽) ----------
const Y4 = Y1 + HA + 0.3, BOT = H - 0.5, H4 = 5.1;
a = card(X2, Y4, CW, H4, 4, "Source Program");
{
  const qs = 1.5;
  s.addImage({ path: "qr_github.png", x: a.x, y: a.y, w: qs, h: qs, altText: "GitHub 저장소 QR 코드", objectName: "qr" });
  text([
    { text: "GitHub 저장소", options: { bold: true, color: C.accent1, breakLine: true } },
    { text: "github.com/kimsangmin1234/jolup", options: { color: C.accent3, bold: true, breakLine: true } },
    { text: "개발 환경  ", options: { bold: true, color: C.accent1, fontSize: 15 } },
    { text: "Python · PyTorch · LightGBM · OpenAI API", options: { fontSize: 15 } },
  ], { x: a.x + qs + 0.3, y: a.y, w: a.w - qs - 0.3, h: qs, fontSize: 17, valign: "middle", paraSpaceAfter: 6 });
  const rows = [
    ["modules/news_encoder.py", "① LLM 기반 주 신호"],
    ["modules/tcn_encoder.py", "② TCN 기반 보조 벡터"],
    ["modules/cross_attention.py", "③ 비대칭 교차 어텐션"],
    ["modules/gated_fusion.py", "④ 게이트 기반 잔차 결합"],
    ["modules/return_predictor.py", "⑤ 등락률 예측"],
    V === "B" ? ["run_edt_suite3.py", "학습 · 앙상블 · 평가"] : ["run_edt_suite2.py", "학습 · 평가"],
  ];
  const ty = a.y + qs + 0.15;
  s.addTable(rows.map(([f, m]) => [
    { text: f, options: { fontFace: "Courier New", fontSize: 13, color: C.accent1, bold: true } },
    { text: m, options: { fontSize: 14.5, color: C.text1 } },
  ]), { x: a.x, y: ty, w: a.w, colW: [3.25, a.w - 3.25], rowH: 0.32, valign: "middle", margin: [0, 0.08, 0, 0.08],
        border: { type: "solid", pt: 0.75, color: HEX.grid }, fill: { color: "FFFFFF" }, objectName: "src-table" });
}

const Y6 = Y4 + H4 + 0.3;
a = card(X2, Y6, CW, BOT - Y6, 6, "결론");
bullets(V === "B" ? [
  ["성과", "이벤트 정보·순위 손실·앙상블로 순위 IC 0.060 → 0.092 (차이 유의)"],
  ["해석", "자사주 매입은 상승, 임상 승인·신규 계약 같은 홍보성 호재는 하락을 학습"],
  ["한계", "거래비용 반영 시 수익 소멸, 일반 뉴스(FNSPID)에서는 비유의"],
] : [
  ["성과", "감성 점수만으로는 예측력이 없지만(−0.006), 제안 구조는 유의한 예측력(0.060)"],
  ["한계", "최근 구간 약화(0.033), 거래비용 반영 시 수익 소멸, 어텐션은 30일에 고르게 분포"],
  ["향후", "입력 확장·앙상블로 0.092까지 개선 확인 → 비용을 고려한 전략 검증"],
], { ...a, fontSize: V === "B" ? 17 : 19, gap: V === "B" ? 4 : 8 });

// ---------- 5. 성능평가 결과 ----------
a = card(M, Y4, CW, BOT - Y4, 5, "성능평가 결과");
{
  let y = a.y;
  text([
    { text: "데이터  ", options: { bold: true, color: C.accent1 } }, { text: "EDT 보도자료(2020.03~2021.05), 표본 외 32,926건", options: { breakLine: true } },
    { text: "지표  ", options: { bold: true, color: C.accent1 } }, { text: "순위 IC = 예측 순위와 실제 1일 초과수익률 순위의 상관" },
  ], { x: a.x, y, w: a.w, h: 0.75, fontSize: 15, lineSpacingMultiple: 1.05 });
  y += 0.9;
  const stats = V === "B"
    ? [["0.092", "최종 앙상블 순위 IC\n95% CI 0.077~0.107"], ["+0.032", "논문 구조 대비 향상\n95% CI 0.015~0.048"], ["+0.37%", "예측 상위 10%\n1일 평균 (비용 전)"]]
    : [["0.060", "논문 구조 순위 IC\n95% CI 0.046~0.073"], ["3 / 3", "평가 구간 모두\n양(+)의 예측력"], ["−0.006", "감성 점수만 사용\n(예측력 없음)"]];
  const sw = (a.w - 0.4) / 3, SH = V === "B" ? 1.65 : 1.8;
  stats.forEach(([n, l], i) => {
    const sx = a.x + i * (sw + 0.2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: sx, y, w: sw, h: SH, rectRadius: 0.1, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: nm("stat") });
    text(n, { x: sx, y: y + 0.12, w: sw, h: 0.8, fontSize: 34, bold: true, color: i === 2 && V === "A" ? C.accent6 : C.accent1, align: "center", valign: "middle" });
    text(l, { x: sx + 0.08, y: y + SH - 0.88, w: sw - 0.16, h: 0.8, fontSize: 13.5, color: C.text1, align: "center", valign: "middle" });
  });
  y += SH + 0.2;
  const axis = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
    catAxisLabelColor: HEX.text, valAxisLabelColor: HEX.gray, valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 15, catAxisLabelFontSize: 15, valAxisLabelFontSize: 12,
    showTitle: true, titleFontSize: 17, titleColor: HEX.navy, valAxisLabelFormatCode: "0.00" };
  const lab = V === "B" ? ["감성 점수만", "논문 구조 (P0)", "개선 구조 (P1)", "최종 앙상블 (E6)"] : ["감성 점수만", "LLM 방향 규칙", "논문 구조 (P0)", "추가 개선 실험"];
  const val = V === "B" ? [-0.006, 0.060, 0.078, 0.092] : [-0.006, 0.000, 0.060, 0.092];
  const h1 = V === "B" ? 2.3 : 2.6;
  s.addChart(pres.charts.BAR, [{ name: "순위 IC", labels: lab, values: val }], {
    x: a.x, y, w: a.w, h: h1, barDir: "bar", ...axis, title: "모델별 순위 IC (3개 구간 합산)",
    chartColors: V === "B" ? [HEX.gray, HEX.teal, HEX.teal, HEX.gold] : [HEX.gray, HEX.gray, HEX.gold, "9AA6B5"],
    valAxisMinVal: -0.04, valAxisMaxVal: 0.1, valAxisMajorUnit: 0.02, barGapWidthPct: 45, catAxisLabelPos: "low", showLegend: false, objectName: "chart-ic",
  });
  y += h1 + 0.15;
  const h2 = a.y + a.h - y + 0.05;
  const series = V === "B"
    ? [{ name: "논문 구조 (P0)", labels: ["20.11~12", "21.01~02", "21.03~05"], values: [0.079, 0.067, 0.033] },
       { name: "최종 앙상블 (E6)", labels: ["20.11~12", "21.01~02", "21.03~05"], values: [0.079, 0.083, 0.110] }]
    : [{ name: "논문 구조 (P0)", labels: ["20.11~12", "21.01~02", "21.03~05"], values: [0.079, 0.067, 0.033] }];
  s.addChart(pres.charts.BAR, series, {
    x: a.x, y, w: a.w, h: h2, barDir: "col", ...axis, title: "평가 구간별 순위 IC",
    chartColors: V === "B" ? [HEX.teal, HEX.gold] : [HEX.gold], valAxisMinVal: 0, valAxisMaxVal: 0.12, valAxisMajorUnit: 0.03,
    barGapWidthPct: 60, showLegend: V === "B", legendPos: "b", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.text,
    dataLabelPosition: "outEnd", objectName: "chart-window",
  });
}

s.addNotes(V === "B"
  ? "버전 B: 실험 최고 성능(12개 모델 검증 비례 순위 앙상블 E6) 중심. 수치는 EDT 3개 평가 구간 32,926건, 날짜 블록 부트스트랩 95% 신뢰구간."
  : "버전 A: 논문 구조 그대로(P0) 중심. 수치는 EDT 3개 평가 구간 32,926건, 날짜 블록 부트스트랩 95% 신뢰구간. 추가 개선 실험(0.092)은 입력 확장·앙상블 결과. 참고: Ridge 선형 기준선은 0.071.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
