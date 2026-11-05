"""검색 / 답변 품질 평가.

  python -m eval.run_eval                 # 검색 평가 (임베딩만 사용, 빠름)
  python -m eval.run_eval --e2e           # + 답변 정확도, 근거 없음 거절율 (LLM 호출)

지표
  hit@k   정답 섹션이 상위 k개 안에 있는 비율
  MRR     정답 섹션의 평균 역순위
  answer  답변에 정답 키워드가 포함된 비율 (answerable 질문)
  refusal 근거 없는 질문에 "근거 없음"으로 답한 비율
"""
import argparse
import json
import time
from pathlib import Path

from rag import config, pipeline
from rag.index import Index

GOLDEN = Path(__file__).with_name("golden.jsonl")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def eval_retrieval(index: Index, golden: list[dict], mode: str, k: int) -> dict:
    items = [g for g in golden if g["answerable"]]
    hits_at_k, rr, lat = 0, 0.0, []
    for g in items:
        t = time.perf_counter()
        res = index.search(g["q"], k=k, mode=mode)
        lat.append(time.perf_counter() - t)
        rank = next(
            (i for i, h in enumerate(res, 1) if h.chunk.source == g["source"] and g["section"] in h.chunk.section),
            None,
        )
        if rank:
            hits_at_k += 1
            rr += 1 / rank
    n = len(items)
    return {"mode": mode, f"hit@{k}": hits_at_k / n, "MRR": rr / n, "avg_ms": 1000 * sum(lat) / n}


def eval_e2e(index: Index, golden: list[dict]) -> dict:
    ok_ans = tot_ans = ok_ref = tot_ref = 0
    for g in golden:
        a = pipeline.ask(index, g["q"])
        if g["answerable"]:
            tot_ans += 1
            ok_ans += int(a.grounded and g["keyword"] in a.answer)
        else:
            tot_ref += 1
            ok_ref += int(not a.grounded)
    return {"answer": ok_ans / tot_ans, "refusal": ok_ref / tot_ref}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--e2e", action="store_true")
    args = ap.parse_args()

    index = Index.build()
    golden = load_golden()
    print(f"chunks={len(index.chunks)}  questions={len(golden)}  embed={config.EMBED_MODEL}\n")
    print(f"{'mode':8} {'hit@' + str(args.k):>7} {'MRR':>6} {'ms/query':>9}")
    for mode in ("bm25", "dense", "hybrid"):
        r = eval_retrieval(index, golden, mode, args.k)
        print(f"{r['mode']:8} {r[f'hit@{args.k}']:7.2f} {r['MRR']:6.2f} {r['avg_ms']:9.1f}")
    if args.e2e:
        r = eval_e2e(index, golden)
        print(f"\nE2E  answer={r['answer']:.2f}  refusal={r['refusal']:.2f}  (chat={config.CHAT_MODEL})")


if __name__ == "__main__":
    main()
