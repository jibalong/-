#!/usr/bin/env python3
"""调用本地 8002 服务，对统一题集做状态、来源、引用和语言的端到端验收。"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag_chat import answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "end_to_end_report.json"
CASES_PATH = ROOT / "tests" / "questions.jsonl"


def main() -> None:
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = []
    for case in cases:
        result = answer(case["question"], top_k=3, scope_policy=case["scope_policy"])
        text = result.get("answer", "")
        citations = [int(n) for n in re.findall(r"\[资料\s*(\d+)\]", text)]
        valid_citations = bool(citations) and all(1 <= n <= len(result.get("results", [])) for n in citations)
        sources = [item["source"] for item in result.get("results", [])]
        source_ok = case["expected_source"] is None or case["expected_source"] in sources
        citation_ok = result["status"] != "ok" or valid_citations
        language_ok = case["answer_language"] != "English" or result["status"] != "ok" or bool(re.search(r"[A-Za-z]{3}", text))
        safety_ok = "expected_safety_note" not in case or bool(result.get("safety_note")) == case["expected_safety_note"]
        passed = result["status"] == case["expected_status"] and source_ok and citation_ok and language_ok and safety_ok
        report.append({"id": case["id"], "question": case["question"], "policy": case["scope_policy"], "expected_status": case["expected_status"], "status": result["status"], "top3_sources": sources, "citations": citations, "citation_numbers_valid": valid_citations, "language_ok": language_ok, "has_safety_note": bool(result.get("safety_note")), "passed": passed, "answer": text})
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"端到端通过 {sum(item['passed'] for item in report)}/{len(report)}；报告：{OUT}")
    for item in report:
        if not item["passed"]:
            print(f"失败 {item['id']}：status={item['status']} citations={item['citations']}")


if __name__ == "__main__":
    main()
