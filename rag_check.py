#!/usr/bin/env python3
"""人工检查 RAG 检索结果：不调用 Qwen，只显示范围、分数和证据预览。"""
from __future__ import annotations

import argparse

from rag_chat import CROPS, answer


def preview(text: str, limit: int) -> str:
    """终端检查只需看开头，避免一条资料块刷满屏幕。"""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 BM25、向量或混合检索证据，不调用回答生成")
    parser.add_argument("question", help="可直接输入口语问题；范围不完整时会显示范围提示")
    parser.add_argument("--crop", choices=CROPS)
    parser.add_argument("--region")
    parser.add_argument("--stage")
    parser.add_argument("--retrieval", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--understanding", choices=("llm", "rules"), default="llm", help="llm 使用 Qwen 抽取与改写；rules 不调用查询理解模型")
    parser.add_argument("--scope-policy", choices=("adaptive", "strict"), default="adaptive", help="adaptive 默认先检索；strict 条件不完整即停止")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--preview", type=int, default=260, help="每条证据的最大预览字符数")
    parser.add_argument("--no-log", action="store_true", help="不写入 logs/retrieval_events.jsonl")
    args = parser.parse_args()

    # retrieve_only=True 是这个工具的核心：它只运行范围过滤和检索，绝不生成回答。
    result = answer(
        args.question, args.crop, args.region, args.stage, args.top_k,
        retrieve_only=True, retrieval=args.retrieval, log=not args.no_log,
        understanding=args.understanding,
        scope_policy=args.scope_policy,
    )
    print(f"状态：{result['status']}")
    if result.get("scope"):
        print("识别范围：" + "；".join(f"{key}={value or '未提供'}" for key, value in result["scope"].items()))
    if result.get("message"):
        print("说明：" + result["message"])
    if result.get("scope_notice"):
        print(result["scope_notice"])
    if result.get("diagnostics"):
        print("检索过程：" + "；".join(f"{key}={value}" for key, value in result["diagnostics"].items()))
    for number, item in enumerate(result.get("results", []), start=1):
        score_names = ("score", "bm25_score", "vector_score", "hybrid_score", "reranker_score")
        scores = "，".join(f"{name}={item[name]}" for name in score_names if name in item)
        print(f"\n[资料 {number}] {scores}")
        print(f"{item['title']} > {item['section']}")
        print(f"范围：{item['region']}；{item['season_or_growth_stage']}")
        print(f"来源：{item['source_url']}")
        print("预览：" + preview(item["text"], args.preview))


if __name__ == "__main__":
    main()
