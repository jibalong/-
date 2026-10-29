#!/usr/bin/env python3
"""逐题运行当前统一题集，生成含答案、证据和人工核对重点的 Markdown 报告。"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from rag_chat import answer

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "questions.jsonl"
DEFAULT_OUTPUT = ROOT / "logs" / "question_set_review.md"


def preview(text: str, limit: int = 360) -> str:
    """报告展示证据摘要；需要复核时仍可按 URL 或 documents/ 查看全文。"""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def citation_check(text: str, result_count: int) -> str:
    """自动检查引用编号范围，不把这个格式检查冒充为农业事实核验。"""
    citations = [int(value) for value in re.findall(r"\[资料\s*(\d+)\]", text)]
    if not text:
        return "不适用（未生成回答）"
    if citations and all(1 <= value <= result_count for value in citations):
        return f"通过（引用：{', '.join(map(str, citations))}）"
    return "失败（没有引用或编号超出证据范围）"


def language_check(text: str, expected_language: str) -> str:
    """英文仅做轻量形式检查；翻译质量仍需人工阅读判断。"""
    if expected_language != "English":
        return "不适用（预期中文）"
    return "通过（检测到英文字符）" if re.search(r"[A-Za-z]{3}", text) else "失败（未检测到英文内容）"


def main() -> None:
    parser = argparse.ArgumentParser(description="运行统一题集，并输出给人工核对的答案报告")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown 报告输出路径")
    parser.add_argument("--retrieval", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--understanding", choices=("llm", "rules"), default="llm", help="默认使用当前 Qwen 查询理解与改写")
    parser.add_argument("--ids", help="可选：逗号分隔的题号，例如 w01,m04,r12")
    args = parser.parse_args()

    selected_ids = set(args.ids.split(",")) if args.ids else None
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if selected_ids:
        cases = [case for case in cases if case["id"] in selected_ids]
    if not cases:
        raise SystemExit("没有匹配的题目；请检查 --ids。")

    lines = [
        "# 粮食作物 RAG 当前题集人工核对报告", "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"题目数：{len(cases)}；检索模式：{args.retrieval}；查询理解：{args.understanding}",
        "", "自动项只检查状态、Top-3 来源、范围字段、引用编号和英文形式。",
        "农业结论是否充分、是否遗漏原文条件、措施是否真的适用于提问情形，必须按每题人工核对重点复核。",
    ]
    status_passed = source_passed = scope_passed = citation_passed = language_passed = 0

    for number, case in enumerate(cases, start=1):
        result = answer(
            case["question"], top_k=3, retrieval=args.retrieval, log=False,
            understanding=args.understanding, scope_policy=case["scope_policy"],
        )
        sources = [item["source"] for item in result.get("results", [])]
        actual_scope = result.get("scope", {})
        status_ok = result["status"] == case["expected_status"]
        source_ok = case["expected_source"] is None or case["expected_source"] in sources
        scope_ok = all(actual_scope.get(key) == value for key, value in case["expected_scope"].items())
        citations = citation_check(result.get("answer", ""), len(result.get("results", [])))
        citation_ok = result["status"] != "ok" or citations.startswith("通过")
        language = language_check(result.get("answer", ""), case["answer_language"])
        language_ok = case["expected_status"] != "ok" or case["answer_language"] != "English" or language.startswith("通过")
        safety_expected = case.get("expected_safety_note")
        safety_ok = safety_expected is None or bool(result.get("safety_note")) == safety_expected

        status_passed += status_ok
        source_passed += source_ok
        scope_passed += scope_ok
        citation_passed += citation_ok
        language_passed += language_ok
        lines.extend([
            "", f"## {number}. {case['id']}｜{case['crop_group']}｜{case['scenario']}", "",
            f"- 问题：{case['question']}",
            f"- 策略：{case['scope_policy']}；预期状态：{case['expected_status']}；实际状态：{result['status']}；状态检查：{'通过' if status_ok else '失败'}",
            f"- 预期来源：{case['expected_source'] or '无'}；Top-3 来源：{', '.join(sources) or '无'}；来源检查：{'通过' if source_ok else '失败'}",
            f"- 预期范围：{json.dumps(case['expected_scope'], ensure_ascii=False)}；实际范围：{json.dumps(actual_scope, ensure_ascii=False)}；范围检查：{'通过' if scope_ok else '失败'}",
            f"- 引用检查：{citations}；英文检查：{language}；安全提示检查：{'通过' if safety_ok else '失败'}",
            f"- 人工核对重点：{case['review_focus']}",
        ])
        if result.get("scope_notice"):
            lines.extend(["", "**范围提示**", "", result["scope_notice"]])
        if result.get("message"):
            lines.extend(["", "**系统提示**", "", result["message"]])
        data = result.get("query_understanding")
        if data:
            lines.extend(["", f"**查询理解**：模式={data['understanding_mode']}；意图={data.get('intent') or '未标注'}；改写={data.get('rewrite_mode')}"])
            if data.get("search_queries"):
                lines.extend(["", "**检索表达**", "", *[f"- {query}" for query in data["search_queries"]]])
        if result.get("results"):
            lines.extend(["", "**检索证据**", ""])
            for index, item in enumerate(result["results"], start=1):
                lines.extend([
                    f"- [资料 {index}] `{item['source']}`：{item['title']} > {item['section']}",
                    f"  - 范围：{item['region']}；{item['season_or_growth_stage']}",
                    f"  - URL：{item['source_url']}",
                    f"  - 原文预览：{preview(item['text'])}",
                ])
        if result.get("answer"):
            lines.extend(["", "**模型回答**", "", result["answer"]])
        if result.get("safety_note"):
            lines.extend(["", "**安全提示**", "", result["safety_note"]])

    lines[7:7] = [
        "", f"自动汇总：状态 {status_passed}/{len(cases)}；来源 {source_passed}/{len(cases)}；范围 {scope_passed}/{len(cases)}；引用 {citation_passed}/{len(cases)}；英文 {language_passed}/{len(cases)}。",
    ]
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已完成 {len(cases)} 题；状态 {status_passed}/{len(cases)}；来源 {source_passed}/{len(cases)}；范围 {scope_passed}/{len(cases)}；引用 {citation_passed}/{len(cases)}；英文 {language_passed}/{len(cases)}。")
    print(f"人工核对报告：{output}")


if __name__ == "__main__":
    main()
