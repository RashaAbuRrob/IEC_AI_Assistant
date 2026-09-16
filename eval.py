# -*- coding: utf-8 -*-
"""
Evaluation harness for the IEC Arabic RAG system.

Expects a JSON file with a list of test items shaped like:

[
  {
    "question": "ما هي شروط الترشح لعضوية مجلس النواب؟",
    "in_scope": true,
    "expected_source_file": "قانون_الانتخاب.pdf",
    "expected_page": 7,
    "expected_article": "12"
  },
  {
    "question": "ما هي عاصمة فرنسا؟",
    "in_scope": false
  }
]

`expected_page` and `expected_article` are optional -- provide whichever
you can verify against the source document; a match on either counts
toward retrieval hit-rate / citation accuracy for that item.

Usage:
    python eval.py --db-path ./chroma_db --qa-file eval_qa.json --k 5 \
        --report-out eval_report.md
"""

import argparse
import json
import re
from pathlib import Path

from chroma_utils import get_collection
from ollama_utils import prewarm_embedder
from rag_service import REFUSAL, answer_question

CITATION_RE = re.compile(
    r"المصدر:\s*([^،\n]+)،\s*(?:صفحة\s*(\d+)|المادة\s*(\d+))"
)


def parse_citations(answer_text: str):
    """Extract (source_file, page_or_None, article_or_None) tuples from the
    generated answer's citation markers."""
    citations = []
    for m in CITATION_RE.finditer(answer_text):
        source_file = m.group(1).strip()
        page = m.group(2)
        article = m.group(3)
        citations.append((source_file, page, article))
    return citations


def retrieval_hit(item: dict, retrieved: list) -> bool:
    exp_file = item.get("expected_source_file")
    if not exp_file:
        return None  # not checkable
    exp_page = str(item.get("expected_page")) if item.get("expected_page") is not None else None
    exp_article = str(item.get("expected_article")) if item.get("expected_article") is not None else None

    for r in retrieved:
        if r["source_file"] != exp_file:
            continue
        if exp_page and str(r["page"]) == exp_page:
            return True
        if exp_article and r["article_number"] and str(r["article_number"]) == exp_article:
            return True
        if not exp_page and not exp_article:
            return True  # only file specified, and file matched
    return False


def citation_correct(item: dict, answer_text: str) -> bool:
    exp_file = item.get("expected_source_file")
    if not exp_file:
        return None  # not checkable
    exp_page = str(item.get("expected_page")) if item.get("expected_page") is not None else None
    exp_article = str(item.get("expected_article")) if item.get("expected_article") is not None else None

    for source_file, page, article in parse_citations(answer_text):
        if source_file != exp_file:
            continue
        if exp_page and page == exp_page:
            return True
        if exp_article and article == exp_article:
            return True
        if not exp_page and not exp_article:
            return True
    return False


def is_refusal(answer_text: str) -> bool:
    return REFUSAL in answer_text


def run_eval(collection, qa_items: list, k: int):
    rows = []
    for item in qa_items:
        question = item["question"]
        in_scope = item.get("in_scope", True)
        result = answer_question(collection, question, k=k)

        row = {
            "question": question,
            "in_scope": in_scope,
            "answer": result["answer"],
            "retrieved": result["retrieved"],
        }

        if in_scope:
            row["retrieval_hit"] = retrieval_hit(item, result["retrieved"])
            row["citation_correct"] = citation_correct(item, result["answer"])
            row["refused"] = is_refusal(result["answer"])
        else:
            row["correct_refusal"] = is_refusal(result["answer"])

        rows.append(row)
    return rows


def summarize(rows: list):
    in_scope_rows = [r for r in rows if r["in_scope"]]
    out_scope_rows = [r for r in rows if not r["in_scope"]]

    checkable_retrieval = [r for r in in_scope_rows if r.get("retrieval_hit") is not None]
    checkable_citation = [r for r in in_scope_rows if r.get("citation_correct") is not None]

    retrieval_hit_rate = (
        sum(1 for r in checkable_retrieval if r["retrieval_hit"]) / len(checkable_retrieval)
        if checkable_retrieval else None
    )
    citation_accuracy = (
        sum(1 for r in checkable_citation if r["citation_correct"]) / len(checkable_citation)
        if checkable_citation else None
    )
    false_refusal_rate = (
        sum(1 for r in in_scope_rows if r.get("refused")) / len(in_scope_rows)
        if in_scope_rows else None
    )
    correct_refusal_rate = (
        sum(1 for r in out_scope_rows if r["correct_refusal"]) / len(out_scope_rows)
        if out_scope_rows else None
    )

    return {
        "n_total": len(rows),
        "n_in_scope": len(in_scope_rows),
        "n_out_of_scope": len(out_scope_rows),
        "retrieval_hit_rate": retrieval_hit_rate,
        "retrieval_hit_rate_n": len(checkable_retrieval),
        "citation_accuracy": citation_accuracy,
        "citation_accuracy_n": len(checkable_citation),
        "false_refusal_rate_on_in_scope": false_refusal_rate,
        "correct_refusal_rate_on_out_of_scope": correct_refusal_rate,
    }


def format_pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def write_markdown_report(summary: dict, rows: list, out_path: Path):
    lines = []
    lines.append("# IEC RAG Evaluation Report\n")
    lines.append(f"- Total questions: {summary['n_total']} "
                 f"({summary['n_in_scope']} in-scope, {summary['n_out_of_scope']} out-of-scope)\n")
    lines.append("## Headline metrics (reported separately, per spec)\n")
    lines.append(f"- **Retrieval hit-rate** (in-scope, checkable questions only, n={summary['retrieval_hit_rate_n']}): "
                 f"{format_pct(summary['retrieval_hit_rate'])}")
    lines.append(f"- **Citation accuracy** (in-scope, checkable questions only, n={summary['citation_accuracy_n']}): "
                 f"{format_pct(summary['citation_accuracy'])}")
    lines.append(f"- **Correct-refusal rate** (out-of-scope questions, n={summary['n_out_of_scope']}): "
                 f"{format_pct(summary['correct_refusal_rate_on_out_of_scope'])}")
    lines.append(f"- False-refusal rate on in-scope questions (should be low): "
                 f"{format_pct(summary['false_refusal_rate_on_in_scope'])}\n")

    lines.append("## Per-question detail\n")
    for i, r in enumerate(rows, start=1):
        lines.append(f"### {i}. {r['question']}")
        lines.append(f"- in_scope: {r['in_scope']}")
        if r["in_scope"]:
            lines.append(f"- retrieval_hit: {r.get('retrieval_hit')}")
            lines.append(f"- citation_correct: {r.get('citation_correct')}")
            lines.append(f"- refused: {r.get('refused')}")
        else:
            lines.append(f"- correct_refusal: {r.get('correct_refusal')}")
        lines.append(f"- answer: {r['answer']}")
        retrieved_str = "; ".join(
            f"{x['source_file']} p{x['page']}" + (f" art{x['article_number']}" if x['article_number'] else "")
            for x in r["retrieved"]
        )
        lines.append(f"- retrieved: {retrieved_str}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the IEC Arabic RAG system.")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--qa-file", required=True, help="JSON file with test Q&A items.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report-out", default="eval_report.md")
    parser.add_argument("--json-out", default="eval_report.json")
    args = parser.parse_args()

    qa_items = json.loads(Path(args.qa_file).read_text(encoding="utf-8"))

    print("Pre-warming bge-m3 embedder via Ollama...")
    prewarm_embedder()
    collection = get_collection(args.db_path)

    rows = run_eval(collection, qa_items, args.k)
    summary = summarize(rows)

    Path(args.json_out).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summary, rows, Path(args.report_out))

    print("\n=== SUMMARY ===")
    print(f"Retrieval hit-rate:      {format_pct(summary['retrieval_hit_rate'])} "
          f"(n={summary['retrieval_hit_rate_n']})")
    print(f"Citation accuracy:       {format_pct(summary['citation_accuracy'])} "
          f"(n={summary['citation_accuracy_n']})")
    print(f"Correct-refusal rate:    {format_pct(summary['correct_refusal_rate_on_out_of_scope'])} "
          f"(n={summary['n_out_of_scope']})")
    print(f"False-refusal (in-scope):{format_pct(summary['false_refusal_rate_on_in_scope'])}")
    print(f"\nFull report: {args.report_out}")
    print(f"Raw data: {args.json_out}")


if __name__ == "__main__":
    main()
