#!/usr/bin/env python3
"""验证统一题集的作物、地区、生育期识别，不调用检索或回答生成。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from query_understanding import understand

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests" / "questions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="评估统一题集的范围字段识别")
    parser.add_argument("--understanding", choices=("llm", "rules"), default="rules")
    parser.add_argument("--scope-policy", choices=("adaptive", "strict"), help="可选：覆盖题集内每题的范围策略")
    args = parser.parse_args()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = 0
    for case in cases:
        policy = args.scope_policy or case["scope_policy"]
        result = understand(case["question"], mode=args.understanding, scope_policy=policy)
        expected = case["expected_scope"]
        good = all((result["crop"] == expected["crop"], result["region"] == expected["region"], result["growth_stage"] == expected["stage"]))
        if policy == "strict" and case["expected_status"] == "need_info":
            good = good and not result["search_queries"]
        elif policy == "adaptive":
            good = good and bool(result["search_queries"])
        passed += good
        print(f"{case['id']}：{'通过' if good else '失败'}；策略={policy}；范围={result['crop']}/{result['region']}/{result['growth_stage']}；rewrite={result['rewrite_mode']}")
    print(f"查询理解通过 {passed}/{len(cases)}（模式：{args.understanding}）。")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
