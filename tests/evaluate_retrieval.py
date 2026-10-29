#!/usr/bin/env python3
"""不调用回答生成，评估当前统一题集的范围判断和 Top-3 来源。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

from rag_chat import answer

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests" / "questions.jsonl"
OUT = ROOT / "logs" / "retrieval_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="对指定检索模式运行回归测试，不调用 Qwen 生成模型")
    parser.add_argument("--retrieval", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--understanding", choices=("llm", "rules"), default="rules", help="默认 rules，保证离线检索回归可复现")
    parser.add_argument("--scope-policy", choices=("adaptive", "strict"), help="可选：覆盖题集内每题的范围策略")
    parser.add_argument("--cases", default=str(TESTS), help="题集 JSONL 路径；默认 tests/questions.jsonl")
    parser.add_argument("--output", help="报告路径；默认按检索模式写入 logs/")
    args = parser.parse_args()
    cases_path = Path(args.cases)
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = []
    for case in cases:
        # log=False：批量回归测试不混入人工使用日志，保持 logs/retrieval_events.jsonl 可读。
        policy = args.scope_policy or case["scope_policy"]
        result = answer(case["question"], top_k=3, retrieve_only=True, retrieval=args.retrieval, log=False, understanding=args.understanding, scope_policy=policy)
        actual = [item["source"] for item in result.get("results", [])]
        passed_status = result["status"] == case["expected_status"]
        passed_top3 = case["expected_source"] is None or case["expected_source"] in actual
        actual_scope = result.get("scope", {})
        passed_scope = all(actual_scope.get(key) == value for key, value in case["expected_scope"].items())
        report.append({**case, "actual_policy": policy, "actual_status": result["status"], "actual_scope": actual_scope, "top3_sources": actual, "passed_status": passed_status, "passed_top3": passed_top3, "passed_scope": passed_scope, "passed": passed_status and passed_top3 and passed_scope})
    output = Path(args.output) if args.output else ROOT / "logs" / f"retrieval_report_{args.retrieval}_current.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    good = sum(item["passed"] for item in report)
    top3_cases = [item for item in report if item["expected_source"]]
    top3_good = sum(item["passed_top3"] for item in top3_cases)
    scope_good = sum(item["passed_scope"] for item in report)
    print(f"模式 {args.retrieval}；题内范围策略：总计 {len(report)}；整体通过 {good}/{len(report)}；范围识别 {scope_good}/{len(report)}；有期望资料的问题 Top-3 命中 {top3_good}/{len(top3_cases)}。")
    print(f"报告：{output}")
    for item in report:
        if not item["passed"]:
            print(f"失败 {item['id']}：期望状态={item['expected_status']}，实际={item['actual_status']}，Top-3={item['top3_sources']}")


if __name__ == "__main__":
    main()
