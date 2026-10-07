"""EDT(Trade the Event, Zhou et al. 2021) 거래 벤치마크를 이벤트 표로 만든다.

EDT 는 PRNewswire·Businesswire 보도자료 303,893건(2020-03 ~ 2021-05)에 분 단위
발행 시각과 발행 시점 이후 1·2·3일 가격을 붙여 공개한 데이터다. 보도자료는 회사가
직접 낸 1차 공시이므로 '이미 일어난 주가 움직임 재보도'가 섞이지 않는다.

거르는 기준
    - 가격 라벨과 종목 코드가 있다.
    - 본문 앞부분에 '(NYSE: XXX)', '(Nasdaq: XXX)' 처럼 해당 종목 코드가 거래소와 함께 나온다
      (EDT 의 종목 연결에는 다른 회사가 붙은 경우가 있어 발행 회사 기사만 남긴다).

라벨 (발행 분의 종가에서 진입: 발행 1분 안의 반응은 넣지 않는다)
    r_hd = end_price_{h}day / start_price_close − 1   (h = 1, 2, 3)
    EDT 의 '1일'은 발행일의 마지막 체결(시간외 포함)까지다.

원문은 재배포하지 않는다. 깃에는 id·종목·시각·가격 라벨만 저장하고, 본문은 작업 폴더에만 둔다.

    python -I prepare_edt.py --edt <EDT 폴더> --out data/edt/edt_events.jsonl.gz --text-out <작업폴더>/edt_text.jsonl.gz
"""
import argparse
import ast
import gzip
import json
import re
from pathlib import Path

EXCH = r"(?:NYSE(?:\s+American|\s+MKT|\s+Arca)?|NASDAQ|Nasdaq(?:GS|GM|CM)?|AMEX|Cboe\s+BZX|OTC\w*|TSX(?:V)?)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edt", required=True)
    ap.add_argument("--out", default="data/edt/edt_events.jsonl.gz")
    ap.add_argument("--text-out", required=True)
    a = ap.parse_args()
    data = json.load(open(Path(a.edt) / "Trading_benchmark" / "evaluate_news.json"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as fo, gzip.open(a.text_out, "wt", encoding="utf-8") as ft:
        for i, r in enumerate(data):
            lab = r.get("labels")
            if not lab:
                continue
            lab = ast.literal_eval(lab) if isinstance(lab, str) else lab
            t = lab.get("ticker")
            if not t or not lab.get("start_price_close") or not lab.get("end_price_1day"):
                continue
            head = r["title"] + " " + r["text"][:3000]
            if not re.search(r"%s\s*[:：]\s*%s\b" % (EXCH, re.escape(t)), head):
                continue
            s = float(lab["start_price_close"])
            if s <= 0:
                continue
            rec = {"edt_id": i, "ticker": t, "pub_time": r["pub_time"], "start_time": lab["start_time"],
                   "start_price": s}
            for h in (1, 2, 3):
                e = lab.get(f"end_price_{h}day")
                rec[f"end_time_{h}d"] = lab.get(f"end_time_{h}day")
                rec[f"r_{h}d"] = float(e) / s - 1 if e else None
            fo.write(json.dumps(rec) + "\n")
            ft.write(json.dumps({"edt_id": i, "ticker": t, "title": r["title"], "text": r["text"][:1500]}) + "\n")
            kept += 1
    print("저장", kept)


if __name__ == "__main__":
    main()
