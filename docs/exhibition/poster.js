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

const Y1 = 3.2, H1 = 4.8;

// ---------- 1. 연구 내용 ----------
let a = card(M, Y1, CW, H1, 1, "연구 내용");
bullets([
  ["필요성", "단기 주가는 기술적 지표만으로 설명하기 어려운 뉴스 이벤트에 크게 반응"],
  ["기존 연구", "LLM 감성 점수 또는 텍스트 임베딩 중 하나만 사용"],
  ["연구 내용", "뉴스의 의미·감성(주 신호)과 과거 30일 시장 상태(보조 신호)를 결합해 뉴스 발표 후 1일 초과수익률 예측"],
  ["검증", "실제 기업 보도자료 6.3만 건, 미래 정보 누수 없는 표본 외 평가"],
], { ...a, fontSize: 19, gap: 7 });

// ---------- 2. 원리 ----------
a = card(X2, Y1, CW, H1, 2, "원리");
const principles = [
  ["① 주 신호", "GPT-4o-mini 요약·감성 + 임베딩(1536차원) → 512차원"],
  ["② 보조 신호", "지표 9종 × 30일 → 인과적 TCN(미래 차단) → 30 × 512"],
  ["③ 비대칭 교차 어텐션", "뉴스(Q)가 30일 지표(K·V) 중 관련 시점을 가중 평균"],
  ["④ 게이트 잔차 결합", "h = p + σ(W[p; c]) ⊙ c → MLP로 등락률 예측"],
];
if (V === "B") principles.push(["⑤ 확장·앙상블", "이벤트 27종·발행 시점 입력, 순위 손실, 12개 모델 앙상블"]);
bullets(principles, { ...a, fontSize: V === "B" ? 18 : 19, gap: V === "B" ? 4 : 9 });

// ---------- 3. 구조 ----------
const Y3 = Y1 + H1 + 0.3, H3 = V === "B" ? 5.4 : 4.55;
a = card(M, Y3, FW, H3, 3, V === "B" ? "구조 (논문 모델 + 확장 · 앙상블)" : "구조 (논문 제안 모델)");
{
  const x0 = a.x + 0.05, ws = [1.75, 2.15, 2.15, 2.1, 1.8, 1.5, 1.4], gap = 0.33;
  const xs = []; let cx = x0; ws.forEach((w) => { xs.push(cx); cx += w + gap; });
  const top = a.y + 0.05, bot = a.y + 1.85, bh = 1.05, mid = a.y + 0.85, mh = 1.25;
  const N = "DCE8F6", T = "E1F0E3", O = "FBE8D0";
  box(xs[0], top, ws[0], bh, "뉴스 본문\n(발행 시각)", N);
  box(xs[1], top, ws[1], bh, V === "B" ? "GPT-4o-mini\n요약·감성·이벤트" : "GPT-4o-mini\n요약 · 감성 점수", N);
  box(xs[2], top, ws[2], bh, V === "B" ? "주 신호 512\n임베딩·감성·이벤트" : "주 신호 512\n임베딩 + 감성", N, { bold: true, line: HEX.navy });
  box(xs[0], bot, ws[0], bh, "기술적 지표\n9종 × 30일", T);
  box(xs[1], bot, ws[1], bh, "TCN\n(인과적 합성곱)", T);
  box(xs[2], bot, ws[2], bh, "보조 신호\n30 × 512", T, { bold: true, line: HEX.green });
  box(xs[3], mid, ws[3], mh, "비대칭\n교차 어텐션\nQ=뉴스 K·V=지표", O, { bold: true, line: HEX.gold });
  box(xs[4], mid, ws[4], mh, "게이트 기반\n잔차 결합", O, { bold: true, line: HEX.gold });
  box(xs[5], mid, ws[5], mh, V === "B" ? "MLP\n순위 손실" : "MLP\n512→256\n→64→1", O);
  box(xs[6], mid, ws[6], mh, V === "B" ? "개선 구조\n예측" : "1일 등락률\n예측", HEX.navy, { color: C.background1, bold: true });
  for (const y of [top, bot]) for (let i = 0; i < 2; i++) arrow(xs[i] + ws[i], y + bh / 2, xs[i + 1], y + bh / 2);
  arrow(xs[2] + ws[2], top + bh / 2, xs[3], mid + 0.35);
  arrow(xs[2] + ws[2], bot + bh / 2, xs[3], mid + mh - 0.35);
  for (let i = 3; i < 6; i++) arrow(xs[i] + ws[i], mid + mh / 2, xs[i + 1], mid + mh / 2);
  // 잔차: 주 신호 → 게이트
  const ry = top + 0.22, gx = xs[4] + ws[4] / 2;
  arrow(xs[2] + ws[2], ry, gx, ry, { dash: "dash", color: HEX.gray, width: 1.75, noHead: true });
  arrow(gx, ry, gx, mid, { dash: "dash", color: HEX.gray, width: 1.75 });
  text("잔차: 주 신호 보존", { x: xs[3] + 0.1, y: ry - 0.42, w: 2.6, h: 0.36, fontSize: 15, color: HEX.gray });
  text("Q = 뉴스 1개, K·V = 30개 시점 (비대칭)", { x: xs[3], y: mid + mh + 0.12, w: 4.6, h: 0.4, fontSize: 15, color: HEX.gray });
  if (V === "B") {
    const ey = bot + bh + 0.3, eh = 0.95;
    text("최종 예측", { x: xs[0], y: ey, w: ws[0], h: eh, fontSize: 19, bold: true, color: C.accent1, valign: "middle" });
    const ens = [["신경망 8종\n(논문 구조 포함)", N, 2.6], ["LightGBM 4종\n(임베딩 PCA 등)", T, 2.2], ["Ridge\n선형 기준선", T, 1.6], ["검증 순위 IC 비례\n가중 순위 평균", O, 2.55], ["최종 예측 E6", HEX.navy, 1.85]];
    let ex = a.x + 1.85;
    ens.forEach(([t, f, w], i) => {
      box(ex, ey, w, eh, t, f, { fs: 16, bold: i >= 3, color: i === 4 ? C.background1 : C.text1, line: i === 3 ? HEX.gold : f });
      if (i < 2) text("+", { x: ex + w, y: ey, w: 0.4, h: eh, fontSize: 26, bold: true, color: C.accent1, align: "center", valign: "middle" });
      else if (i < 4) arrow(ex + w + 0.05, ey + eh / 2, ex + w + 0.35, ey + eh / 2);
      ex += w + 0.4;
    });
  }
}

// ---------- 4. Source Program / 6. 결론 (왼쪽), 5. 성능평가 (오른쪽) ----------
const Y4 = Y3 + H3 + 0.3, BOT = H - 0.5, H4 = V === "B" ? 4.95 : 5.75;
a = card(M, Y4, CW, H4, 4, "Source Program");
{
  const code = V === "B" ? [
    "# module 3: asymmetric cross-attention",
    "q, k, v = Wq(news), Wk(market), Wv(market)",
    "w = softmax(einsum('bd,btd->bt', q, k) * scale)",
    "c = einsum('bt,btd->bd', w, v)   # (B, 512)",
    "# module 4: gated residual fusion",
    "h = news + sigmoid(gate([news, c])) * c",
    "# final: validation-IC weighted rank ensemble",
    "wt = [max(ric(p_val[m], y_val), 0) for m in M]",
    "final = sum(w * rank(p_test[m]) for w, m in zip(wt, M))",
  ] : [
    "class AsymmetricCrossAttention(nn.Module):",
    "  def forward(self, news, market):",
    "    q = self.query_proj(news)     # (B, 512)",
    "    k = self.key_proj(market)     # (B, 30, 512)",
    "    v = self.value_proj(market)",
    "    s = einsum('bd,btd->bt', q, k) * self.scale",
    "    w = softmax(s, dim=-1)        # 30-day weights",
    "    return einsum('bt,btd->bd', w, v), w",
    "class GatedResidualFusion(nn.Module):",
    "  def forward(self, p, c):",
    "    g = sigmoid(self.gate_net(cat([p, c], -1)))",
    "    return p + g * c, g",
  ];
  const ch = V === "B" ? 2.25 : 3.0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: a.x, y: a.y, w: a.w, h: ch, rectRadius: 0.08, fill: { color: "1E2533" }, line: { color: "1E2533" }, objectName: nm("code-bg") });
  text(code.map((l, i) => ({ text: l, options: { color: l.trim().startsWith("#") || l.startsWith("class") ? (l.startsWith("class") ? "F2C46D" : "8FA3BF") : "E8EDF4", breakLine: i < code.length - 1 } })),
    { x: a.x + 0.22, y: a.y + 0.15, w: a.w - 0.4, h: ch - 0.3, fontFace: "Courier New", fontSize: V === "B" ? 13.5 : 13.5, lineSpacingMultiple: 1.0, valign: "middle" });
  text([
    { text: "GitHub  ", options: { bold: true, color: C.accent1 } },
    { text: "github.com/kimsangmin1234/jolup", options: { color: C.accent3, breakLine: true } },
    { text: "modules/  ", options: { bold: true, color: C.accent1, fontFace: "Courier New" } },
    { text: "논문 모듈 1~5 (news_encoder, tcn_encoder, cross_attention, gated_fusion, return_predictor)", options: { breakLine: true } },
    { text: V === "B" ? "run_edt_suite3.py  " : "run_edt_suite2.py  ", options: { bold: true, color: C.accent1, fontFace: "Courier New" } },
    { text: V === "B" ? "개선 구조 학습 · 앙상블 · 평가" : "EDT 표본 외 학습 · 평가" },
  ], { x: a.x, y: a.y + ch + 0.2, w: a.w, h: a.h - ch - 0.2, fontSize: 15, lineSpacingMultiple: 1.05, paraSpaceAfter: 4 });
}

const Y6 = Y4 + H4 + 0.3;
a = card(M, Y6, CW, BOT - Y6, 6, "결론");
bullets(V === "B" ? [
  ["성과", "이벤트 정보·순위 손실·앙상블로 순위 IC 0.060 → 0.092 (차이 유의)"],
  ["해석", "자사주 매입은 상승, 임상 승인·신규 계약 같은 홍보성 호재는 하락을 학습"],
  ["한계", "거래비용 반영 시 수익 소멸, 일반 뉴스(FNSPID)에서는 비유의"],
] : [
  ["성과", "감성 점수만으로는 예측력이 없지만(−0.006), 제안 구조는 유의한 예측력(0.060)"],
  ["한계", "최근 구간 약화(0.033), 거래비용 반영 시 수익 소멸, 어텐션은 30일에 고르게 분포"],
  ["향후", "입력 확장·앙상블로 0.092까지 개선 확인 → 비용을 고려한 전략 검증"],
], { ...a, fontSize: 17.5, gap: 5 });

// ---------- 5. 성능평가 결과 ----------
a = card(X2, Y4, CW, BOT - Y4, 5, "성능평가 결과");
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
  const h1 = V === "B" ? 2.2 : 2.85;
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
