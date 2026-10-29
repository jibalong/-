#!/usr/bin/env python3
"""粮食作物 RAG：受控查询理解、多路检索、证据重排与可选生成。"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from build_index import tokenize
from query_understanding import CROPS, RISK_WORDS, STAGES, understand

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8002/v1/chat/completions"
LOG_PATH = ROOT / "logs" / "retrieval_events.jsonl"
EMBEDDING_MODEL_PATH = ROOT / "models" / "bge-m3"
RERANKER_MODEL_PATH = ROOT / "models" / "bge-reranker-v2-m3"
RRF_K = 60
_embedding_model = None
_reranker_model = None

# 这些是“必须在证据里直接看到”的现象组，而非农业知识推理规则。
# 例如用户问“太冷”时，不能让 bge-m3 因“温度”语义接近而拿“高温热害”段落作答。
EVIDENCE_REQUIREMENTS = {
    "low_temperature": {
        "question_terms": ("低温", "太冷", "降温", "冻害", "冷害", "寒潮", "倒春寒", "受冻", "霜冻", "早霜"),
        "evidence_terms": ("低温", "冻害", "冷害", "寒潮", "倒春寒", "受冻", "抗寒", "霜冻", "早霜", "保温", "增温"),
    },
    "high_temperature": {
        "question_terms": ("高温", "太热", "热害", "热浪", "晒伤"),
        "evidence_terms": ("高温", "热害", "耐热", "降温", "遮阴"),
    },
}


def stage_matches(stage: str, chunk: dict) -> bool:
    """资料块必须直接提及用户阶段；来源级宽泛阶段不能替代块级证据。"""
    if stage in chunk.get("growth_stages", []):
        return True
    aliases = {
        "育秧": ("育秧", "育苗", "苗床", "苗棚", "秧苗"),
        "插秧": ("插秧", "移栽", "栽插"),
        "返青": ("返青", "活棵", "缓苗"),
        "灌浆": ("灌浆", "结实", "乳熟"),
        "开花": ("开花", "扬花", "吐丝"),
    }
    stage_words = aliases.get(stage, (stage,))
    haystack = chunk["section"] + "\n" + chunk["text"]
    if any(word in haystack for word in stage_words):
        return True
    # 正文没有明确阶段标签的块可使用已登记的资料范围；一旦块已有其他阶段标签，
    # 则不回退，避免“灌浆块”仅因整份资料覆盖全生育期而被当成“播种块”。
    return not chunk.get("growth_stages") and any(word in chunk["season_or_growth_stage"] for word in stage_words)


def evidence_requirement(question: str) -> tuple[str, tuple[str, ...]] | None:
    """识别不能由语义相似度替代的明确现象，例如低温与高温方向相反。"""
    for name, definition in EVIDENCE_REQUIREMENTS.items():
        if any(term in question for term in definition["question_terms"]):
            return name, definition["evidence_terms"]
    return None


def _sentence_has_conflicting_region(sentence: str, requested_region: str | None) -> bool:
    """资料若在同一句明确写了别的区域，不把它冒充成用户所在区域的证据。"""
    if not requested_region:
        return False
    mentioned = [region for region in ("东北", "黄淮海", "长江中下游", "西北", "华北", "西南", "江淮", "江汉", "江南", "华南") if region in sentence]
    return bool(mentioned) and requested_region not in mentioned


def has_direct_evidence(chunk: dict, terms: tuple[str, ...], requested_region: str | None) -> bool:
    """现象词要在同一条原文句子中出现；区域冲突句不作为本地证据。"""
    sentences = re.split(r"[。！？；\n]", chunk["text"])
    for sentence in sentences:
        if any(term in sentence for term in terms) and not _sentence_has_conflicting_region(sentence, requested_region):
            return True
    return False


def vector_input(chunk: dict) -> str:
    """向量与重排都看到 metadata，防止正文相似却跨区域或跨阶段。"""
    return "\n".join(
        [
            f"作物：{chunk['crop']}", f"地区：{chunk['region']}",
            f"生育期：{chunk['season_or_growth_stage']}",
            f"块级阶段：{'、'.join(chunk.get('growth_stages', [])) or '未标注'}",
            f"主题：{chunk['topic']}", f"标题：{chunk['title']}",
            f"章节：{chunk['section']}", f"正文：{chunk['text']}",
        ]
    )


def scope_from_understanding(data: dict, scope_policy: str) -> tuple[dict, str | None]:
    """将查询理解结果转换为检索条件；strict 才会因范围不完整而停止。"""
    scope = {
        "crop": data["crop"],
        "region": data["region"],
        "stage": data["growth_stage"],
        "planting_method": data["planting_method"],
    }
    return scope, data.get("missing_message") if scope_policy == "strict" else None


def filter_chunks(index: dict, scope_data: dict, scope_policy: str) -> tuple[list[dict], dict]:
    """对已确认范围强过滤；未知范围不伪造条件，交由检索器按问题文本排序。"""
    all_chunks = index["chunks"]
    crop_filtered = [chunk for chunk in all_chunks if chunk["crop"] == scope_data["crop"]] if scope_data["crop"] else all_chunks

    if scope_data["region"]:
        region_filtered = []
        for chunk in crop_filtered:
            requested, covered = scope_data["region"], chunk["region"]
            if requested == "全国" or requested in covered or "全国" in covered:
                region_filtered.append(chunk)
    else:
        region_filtered = crop_filtered

    if scope_data["stage"]:
        # 自适应只放宽“未提供”的字段。用户明确给出阶段时仍须硬过滤，
        # 否则“冬小麦育秧”会错误回退为一般小麦资料。
        stage_filtered = [chunk for chunk in region_filtered if stage_matches(scope_data["stage"], chunk)]
    else:
        stage_filtered = region_filtered

    return stage_filtered, {
        "all_chunks": len(index["chunks"]),
        "after_crop": len(crop_filtered),
        "after_region": len(region_filtered),
        "after_stage": len(stage_filtered),
    }


def query_context(question: str, scope_data: dict) -> str:
    """只把已确认 metadata 放入向量和重排查询，绝不拼入字符串 None。"""
    fields = [("作物", scope_data["crop"]), ("地区", scope_data["region"]), ("生育期", scope_data["stage"])]
    return "\n".join([f"{name}：{value}" for name, value in fields if value] + [f"问题：{question}"])


def bm25_scores(index: dict, question: str, chunks: list[dict]) -> list[dict]:
    """为已过滤的块计算可解释的关键词 BM25 分数。"""
    query_terms = tokenize(question)
    n, avgdl = len(index["chunks"]), index["average_length"]
    selected = []
    for chunk in chunks:
        score = 0.0
        for term in query_terms:
            tf = chunk["term_freq"].get(term, 0)
            if not tf:
                continue
            df = index["document_frequency"].get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * chunk["length"] / avgdl))
        if score > 0:
            selected.append({**chunk, "bm25_score": round(score, 6)})
    return sorted(selected, key=lambda item: item["bm25_score"], reverse=True)


def load_embedding_model():
    """仅在向量检索时加载 bge-m3，纯 BM25 不占用额外显存。"""
    global _embedding_model
    if _embedding_model is None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到 bge-m3 模型：{EMBEDDING_MODEL_PATH}")
        from FlagEmbedding import BGEM3FlagModel
        _embedding_model = BGEM3FlagModel(str(EMBEDDING_MODEL_PATH), use_fp16=True)
    return _embedding_model


def vector_scores(question: str, scope_data: dict, chunks: list[dict]) -> list[dict]:
    """查询向量和索引内归一化向量点积，即余弦相似度。"""
    if not chunks or any("dense_vector" not in chunk for chunk in chunks):
        return []
    model = load_embedding_model()
    query = query_context(question, scope_data)
    encoded = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)
    query_vector = np.asarray(encoded["dense_vecs"][0], dtype=np.float32)
    matrix = np.asarray([chunk["dense_vector"] for chunk in chunks], dtype=np.float32)
    scores = matrix @ query_vector
    return sorted(
        [{**chunk, "vector_score": round(float(score), 6)} for chunk, score in zip(chunks, scores, strict=True)],
        key=lambda item: item["vector_score"], reverse=True,
    )


def chunk_key(item: dict) -> str:
    return item["source"] + "\0" + item["section"] + "\0" + item["text"]


def rrf_fuse(rankings: list[list[dict]]) -> list[dict]:
    """按排名融合原问题、改写问题、BM25 和向量结果，避免强比原始分数。"""
    merged: dict[str, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            key = chunk_key(item)
            current = merged.setdefault(key, {**item, "rrf_score": 0.0, "retrieval_hits": 0})
            current["rrf_score"] += 1 / (RRF_K + rank)
            current["retrieval_hits"] += 1
            for score_name in ("bm25_score", "vector_score"):
                if score_name in item:
                    current[score_name] = max(current.get(score_name, float("-inf")), item[score_name])
    return sorted(merged.values(), key=lambda item: (item["rrf_score"], item["retrieval_hits"]), reverse=True)


def load_reranker():
    """仅重排融合后的少量候选。"""
    global _reranker_model
    if _reranker_model is None:
        if not RERANKER_MODEL_PATH.exists():
            raise FileNotFoundError(f"未找到 reranker 模型：{RERANKER_MODEL_PATH}")
        from FlagEmbedding import FlagReranker
        _reranker_model = FlagReranker(str(RERANKER_MODEL_PATH), use_fp16=True)
    return _reranker_model


def rerank(question: str, scope_data: dict, candidates: list[dict]) -> list[dict]:
    """将用户范围与问题一起交给 reranker，进一步降低阶段误排。"""
    if not candidates:
        return []
    model = load_reranker()
    query = query_context(question, scope_data)
    scores = model.compute_score([[query, vector_input(item)] for item in candidates], normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    raw_scores = [round(float(score), 6) for score in scores]
    specific_scores = [score for item, score in zip(candidates, raw_scores, strict=True) if item["section"] != "正文"]
    best_specific_score = max(specific_scores, default=float("-inf"))
    ranked = []
    for item, raw_score in zip(candidates, raw_scores, strict=True):
        # 综合正文可能覆盖多个环节；有明确章节时仅作补充证据，不能压过具体章节。
        # 但若综合正文显著更相关（如整份病虫害技术意见），则不能机械降级。
        section_penalty = 0.03 if item["section"] == "正文" and best_specific_score >= raw_score - 0.01 else 0.0
        ranked.append({**item, "reranker_score": raw_score, "section_penalty": section_penalty, "final_rank_score": round(raw_score - section_penalty, 6)})
    return sorted(ranked, key=lambda item: item["final_rank_score"], reverse=True)


def retrieve(index: dict, queries: list[str], scope_data: dict, top_k: int, mode: str, scope_policy: str) -> tuple[list[dict], dict]:
    """原问题和改写问题分别召回，RRF 融合后再重排。"""
    chunks, diagnostics = filter_chunks(index, scope_data, scope_policy)
    requirement = evidence_requirement(queries[0]) if queries else None
    if requirement:
        requirement_name, requirement_terms = requirement
        chunks = [chunk for chunk in chunks if has_direct_evidence(chunk, requirement_terms, scope_data["region"])]
        diagnostics["evidence_requirement"] = requirement_name
        diagnostics["after_direct_evidence"] = len(chunks)
    # 仅当同一份资料已有具体章节块时，才排除它的“正文”总览块。
    # 某些官方网页没有章节结构，整份资料只能作为“正文”块，必须保留。
    sources_with_sections = {chunk["source"] for chunk in chunks if chunk["section"] != "正文"}
    chunks = [chunk for chunk in chunks if chunk["section"] != "正文" or chunk["source"] not in sources_with_sections]
    diagnostics["after_specific_section"] = len(chunks)
    if any("安全间隔期" in query for query in queries):
        chunks = [chunk for chunk in chunks if "安全间隔期" in chunk["text"]]
        diagnostics["after_safety_evidence"] = len(chunks)
    diagnostics.update({"requested_mode": mode, "query_count": len(queries)})
    if not chunks:
        diagnostics["used_mode"] = mode
        return [], diagnostics

    bm25_rankings = [bm25_scores(index, query, chunks) for query in queries]
    if mode == "bm25":
        ranked = rrf_fuse(bm25_rankings)
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
        diagnostics["used_mode"] = "bm25_rrf"
        return ranked[:top_k], diagnostics

    vector_enabled = index.get("vector", {}).get("enabled") and all("dense_vector" in item for item in chunks)
    if not vector_enabled:
        ranked = rrf_fuse(bm25_rankings)
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
        diagnostics.update({"used_mode": "bm25_fallback", "warning": index.get("vector", {}).get("reason", "index.json 中没有可用的 dense_vector")})
        return ranked[:top_k], diagnostics

    try:
        vector_rankings = [vector_scores(query, scope_data, chunks) for query in queries]
    except Exception as exc:
        ranked = rrf_fuse(bm25_rankings)
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
        diagnostics.update({"used_mode": "bm25_fallback", "warning": f"bge-m3 加载或编码失败：{type(exc).__name__}: {exc}"})
        return ranked[:top_k], diagnostics

    if mode == "vector":
        ranked = rrf_fuse(vector_rankings)
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
        diagnostics["used_mode"] = "vector_rrf"
        return ranked[:top_k], diagnostics

    candidates = rrf_fuse([*bm25_rankings, *vector_rankings])[:12]
    diagnostics["hybrid_candidates"] = len(candidates)
    try:
        ranked = rerank(queries[0], scope_data, candidates)
        for item in ranked:
            item["score"] = item["final_rank_score"]
        diagnostics["used_mode"] = "hybrid_rrf_reranked"
    except Exception as exc:
        ranked = candidates
        for item in ranked:
            item["score"] = round(item["rrf_score"], 6)
        diagnostics.update({"used_mode": "hybrid_rrf_without_reranker", "warning": f"reranker 加载或重排失败：{type(exc).__name__}: {exc}"})
    return ranked[:top_k], diagnostics


def append_retrieval_log(question: str, result: dict, diagnostics: dict, elapsed_ms: int) -> None:
    """一行 JSON 保存抽取、改写、过滤与排序，支持之后的误差复盘。"""
    event = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": question,
        "status": result["status"],
        "scope": result.get("scope"),
        "query_understanding": result.get("query_understanding"),
        "diagnostics": diagnostics,
        "elapsed_ms": elapsed_ms,
        "results": [
            {
                "source": item["source"], "title": item["title"], "section": item["section"],
                "score": item.get("score"), "rrf_score": item.get("rrf_score"),
                "bm25_score": item.get("bm25_score"), "vector_score": item.get("vector_score"),
                "reranker_score": item.get("reranker_score"),
            }
            for item in result.get("results", [])
        ],
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def evidence_text(results: list[dict]) -> str:
    return "\n\n".join(
        f"[资料 {i}] 标题：{item['title']}\n章节：{item['section']}\n适用范围：{item['region']}；{item['season_or_growth_stage']}\n原文：{item['text']}"
        for i, item in enumerate(results, 1)
    )


def _request_answer(system: str, question: str, scope_data: dict, results: list[dict], retry: bool = False) -> str:
    """向本地 Qwen 请求回答；retry 只用于修复无引用的无效生成。"""
    retry_rule = ""
    if retry:
        retry_rule = "\n上一次输出无引用，不能接受。检索证据已经与问题范围匹配：必须从证据中提取直接措施并在每句后标注 [资料 N]；不要只回复“现有资料未提及”。"
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "system", "content": system + retry_rule},
            {"role": "user", "content": f"问题：{question}\n\n检索证据：\n{evidence_text(results)}"},
        ],
        "temperature": 0.1,
        "max_tokens": 700,
    }
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"].strip()


def _translate_cited_answer(chinese_answer: str, retry: bool = False) -> str:
    """英文回答从已引用的中文答案翻译而来，避免翻译时重新选择或丢失证据。"""
    retry_rule = "\n上一次翻译遗漏或改动了引用。每一条都必须保留原有 [资料 N] 并放在行末。" if retry else ""
    system = """你是技术资料翻译器，不补充事实。把输入的 Markdown 短横线列表逐行翻译成英文。
必须保留相同的行数、顺序和每行末尾的 [资料 N]，不得添加或删除引用、标题、解释。""" + retry_rule
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": chinese_answer}],
        "temperature": 0.0,
        "max_tokens": 700,
    }
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"].strip()


def has_line_level_citations(content: str) -> bool:
    """要求每个实际回答行既有内容又在行末引用，不能只输出范围标签。"""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    answer_lines = [line for line in lines if not re.fullmatch(r"(?:[-*]\s*)?(?:资料\s*\d+)?\s*\[资料\s*\d+\]", line)]
    if not answer_lines or not all(re.search(r"\[资料\s*\d+\](?:\s*)$", line) for line in answer_lines):
        return False
    # 去掉列表符号、引用和“【范围】”前缀后，仍必须有一段实际陈述。
    return all(len(re.sub(r"(?:[-*]\s*|\[资料\s*\d+\]|【[^】]+】|\s)", "", line)) >= 8 for line in answer_lines)


def call_model(question: str, scope_data: dict, query_data: dict, results: list[dict]) -> str:
    """生成只组织证据；范围不完整时，要求模型逐条标出证据自己的适用条件。"""
    confirmed = "；".join(
        f"{label}={value}" for label, value in (("作物", scope_data["crop"]), ("地区", scope_data["region"]), ("生育期", scope_data["stage"])) if value
    ) or "无"
    missing = query_data.get("missing_message")
    incomplete_rule = ""
    if missing:
        incomplete_rule = f"""
用户问题的范围尚未完全确认：{missing}
这不是停止回答的理由，但每一行必须在措施前明确写出该条证据实际适用的作物、地区和阶段，例如“【东北水稻育秧】采取…… [资料 1]”。范围标签后必须有直接措施或事实，不能只输出范围标签。不得把不同作物、地区或阶段的措施合并成对用户的统一处方。"""
    system = f"""你是农业资料问答助手。只可依据给出的检索证据回答，不能用模型记忆补充。
本次已确认范围：{confirmed}。只回答证据直接支持的措施，不能加入证据外的阶段、地区或作物内容。{incomplete_rule}
回答必须使用中文。引用格式始终使用 [资料 N]。
输出格式严格为 2 至 5 行 Markdown 短横线列表：每一行只写一项直接措施，并且行末必须是 [资料 N]。不要写标题、前言、结语或“资料 N”单独一行。
不得改变证据中的条件、对象、时间或因果关系；尤其不能把“遇有冻害”改成“温度过高”等不同情形。只要证据中有直接措施，就必须据此回答；不得因为用户说法口语化而只回复“现有资料未提及”。若所有证据确实没有相关内容，每一行也必须说明依据并在行末引用。不得把一个编号的内容归到另一个编号。若涉及农药、施用剂量或安全间隔期，必须提醒：以当地登记、产品标签和植保人员意见为准。"""
    try:
        content = _request_answer(system, question, scope_data, results)
        content = re.sub(r"\[资料\s*(\d+)\s+资料\s*(\d+)\]", r"[资料 \1][资料 \2]", content)
        if not has_line_level_citations(content):
            content = _request_answer(system, question, scope_data, results, retry=True)
            content = re.sub(r"\[资料\s*(\d+)\s+资料\s*(\d+)\]", r"[资料 \1][资料 \2]", content)
        if not has_line_level_citations(content):
            return "模型两次输出均未满足逐条引用格式，已拒绝展示；请先核对上方检索证据。"
        if query_data.get("answer_language") == "English":
            translated = _translate_cited_answer(content)
            if not has_line_level_citations(translated):
                translated = _translate_cited_answer(content, retry=True)
            if has_line_level_citations(translated):
                return translated
            return "英文转换未能保留逐条引用，以下展示已验证引用的中文回答：\n" + content
        return content
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return f"模型服务不可用（{exc}）。已完成检索；可先核对下方证据，或启动 model_server.py。"


def answer(question: str, crop: str | None = None, region: str | None = None, stage: str | None = None, top_k: int = 3, retrieve_only: bool = False, retrieval: str = "hybrid", log: bool = True, understanding: str = "llm", scope_policy: str = "adaptive") -> dict:
    """完整入口：adaptive 默认检索部分范围，strict 保留原来的条件不足即停止行为。"""
    started = time.perf_counter()
    diagnostics: dict = {"requested_mode": retrieval, "requested_understanding": understanding}

    def finish(result: dict) -> dict:
        if log:
            append_retrieval_log(question, result, diagnostics, round((time.perf_counter() - started) * 1000))
        return result

    index_path = ROOT / "index.json"
    if not index_path.exists():
        return finish({"status": "error", "message": "索引不存在，请先运行 python build_index.py"})
    query_data = understand(question, crop, region, stage, understanding, scope_policy)
    scope_data, refusal = scope_from_understanding(query_data, scope_policy)
    diagnostics["understanding_mode"] = query_data["understanding_mode"]
    if query_data.get("warning"):
        diagnostics["understanding_warning"] = query_data["warning"]
    if refusal:
        return finish({"status": "need_info", "message": refusal + "。严格模式不会进入检索。", "scope": scope_data, "query_understanding": query_data, "results": []})

    index = json.loads(index_path.read_text(encoding="utf-8"))
    results, retrieval_diagnostics = retrieve(index, query_data["search_queries"], scope_data, top_k, retrieval, scope_policy)
    diagnostics.update(retrieval_diagnostics)
    if not results:
        if retrieval_diagnostics.get("evidence_requirement"):
            message = "现有资料中没有与问题现象直接对应的原文证据，不能把语义相近但条件相反的段落当作答案。请补充该作物、地区和情形的权威资料。"
        else:
            message = "现有资料没有与已确认条件相匹配的证据，不能据此作答。请补充相应范围的权威资料或核实条件。"
        return finish({"status": "out_of_scope", "message": message, "scope": scope_data, "query_understanding": query_data, "results": []})
    result = {"status": "ok", "scope": scope_data, "query_understanding": query_data, "results": results, "diagnostics": retrieval_diagnostics}
    if query_data.get("missing_message"):
        result["scope_notice"] = "范围提示：" + query_data["missing_message"] + "。系统已按已确认条件检索；每条回答会标明资料自身适用范围。"
    result["answer"] = "" if retrieve_only else call_model(question, scope_data, query_data, results)
    if any(word in question for word in RISK_WORDS):
        result["safety_note"] = "涉及农药、施用剂量或安全间隔期时，请以当地登记、产品标签和植保人员意见为准。"
    return finish(result)


def print_result(result: dict, retrieve_only: bool) -> None:
    print(f"状态：{result['status']}")
    if result.get("scope"):
        print("识别范围：" + json.dumps(result["scope"], ensure_ascii=False))
    if result.get("query_understanding"):
        data = result["query_understanding"]
        print(f"查询理解：模式={data['understanding_mode']}；意图={data.get('intent') or '未标注'}；现象={'、'.join(data.get('issues', [])) or '未标注'}")
        if data.get("search_queries"):
            print("检索表达：" + " | ".join(data["search_queries"]))
    if result.get("message"):
        print(result["message"])
    if result.get("scope_notice"):
        print(result["scope_notice"])
    if result.get("diagnostics", {}).get("warning"):
        print("检索提示：" + result["diagnostics"]["warning"])
    for i, item in enumerate(result.get("results", []), 1):
        scores = "，".join(f"{name}={item[name]}" for name in ("score", "rrf_score", "bm25_score", "vector_score", "reranker_score") if name in item)
        print(f"\n[资料 {i}] {scores}\n{item['title']}｜{item['section']}\n范围：{item['region']}；{item['season_or_growth_stage']}\nURL：{item['source_url']}\n{item['text']}")
    if not retrieve_only and result.get("answer"):
        print("\n回答：\n" + result["answer"])
    if result.get("safety_note"):
        print("\n安全提示：" + result["safety_note"])


def main() -> None:
    parser = argparse.ArgumentParser(description="粮食作物 RAG 终端问答")
    parser.add_argument("question", nargs="?", help="问题；省略则进入交互模式")
    parser.add_argument("--crop", choices=CROPS)
    parser.add_argument("--region", help="如：东北、黄淮海、长江中下游、全国")
    parser.add_argument("--stage", choices=STAGES, help="如：播种、育秧、返青、灌浆")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--retrieval", choices=("bm25", "vector", "hybrid"), default="hybrid")
    parser.add_argument("--understanding", choices=("llm", "rules"), default="llm", help="llm 默认用 Qwen 抽取和改写；rules 为离线可复现后备模式")
    parser.add_argument("--scope-policy", choices=("adaptive", "strict"), default="adaptive", help="adaptive 默认：条件不完整也检索；strict：沿用旧版条件不足即停止")
    parser.add_argument("--retrieve-only", action="store_true", help="只检索，不调用回答生成；llm 查询理解仍可能调用 8002")
    parser.add_argument("--no-log", action="store_true", help="不写入 logs/retrieval_events.jsonl")
    args = parser.parse_args()
    if args.question:
        print_result(answer(args.question, args.crop, args.region, args.stage, args.top_k, args.retrieve_only, args.retrieval, not args.no_log, args.understanding, args.scope_policy), args.retrieve_only)
        return
    print("粮食作物 RAG。默认自适应检索：条件不完整也会先找证据；加 --scope-policy strict 可启用旧版严格停止。输入 quit 退出。")
    while True:
        question = input("\n问题> ").strip()
        if question.lower() in {"quit", "exit", "退出"}:
            return
        if question:
            print_result(answer(question, top_k=args.top_k, retrieve_only=args.retrieve_only, retrieval=args.retrieval, log=not args.no_log, understanding=args.understanding, scope_policy=args.scope_policy), args.retrieve_only)


if __name__ == "__main__":
    main()
