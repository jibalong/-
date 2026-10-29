#!/usr/bin/env python3
"""将口语化农业问题转为受校验的检索条件，不生成农业技术结论。"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8002/v1/chat/completions"
VOCAB = json.loads((ROOT / "domain_vocab.json").read_text(encoding="utf-8"))
CROPS = tuple(VOCAB["crops"])
REGIONS = tuple(VOCAB["regions"])
STAGES = tuple(VOCAB["growth_stages"])
METHODS = tuple(VOCAB["planting_methods"])
RISK_WORDS = tuple(VOCAB["risk_words"])


def first_alias_match(text: str, groups: dict[str, list[str]]) -> str | None:
    """从可审计的词表中找第一个标准字段，规则模式和 LLM 校验都复用它。"""
    return next((name for name, aliases in groups.items() if any(alias in text for alias in aliases)), None)


def rule_understanding(question: str, crop: str | None = None, region: str | None = None, stage: str | None = None) -> dict[str, Any]:
    """无模型后备模式：只使用本地词表，不猜测未在问题中出现的关键条件。"""
    found_crops = [name for name, aliases in VOCAB["crops"].items() if any(alias in question for alias in aliases)]
    found_regions = [name for name in REGIONS if name in question]
    matched_stages = [name for name, aliases in VOCAB["growth_stages"].items() if any(alias in question for alias in aliases)]
    # “秋播苗期”“夏玉米播种期”同时含季节和具体环节时，具体环节优先。
    specific_order = ("播种", "育秧", "插秧", "返青", "苗期", "分蘖", "拔节", "孕穗", "抽穗", "开花", "灌浆", "成熟", "中后期")
    resolved_stage = stage or next((item for item in specific_order if item in matched_stages), matched_stages[0] if matched_stages else None)
    ambiguous = any(phrase in question for phrase in VOCAB["ambiguous_stage_phrases"])
    if ambiguous and not stage:
        resolved_stage = None
    return {
        "crop": crop or (found_crops[0] if len(found_crops) == 1 else None),
        "region": region or (found_regions[0] if len(found_regions) == 1 else None),
        "growth_stage": resolved_stage,
        "planting_method": first_alias_match(question, VOCAB["planting_methods"]),
        "issues": [],
        "intent": "",
        "ambiguous_stage": ambiguous or len(found_crops) > 1 or len(found_regions) > 1,
        "search_queries": [],
        "understanding_mode": "rules",
    }


def _extract_json(text: str) -> dict[str, Any]:
    """允许模型偶尔包上 Markdown 代码块，但只接受其中的 JSON 对象。"""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("模型输出中没有 JSON 对象")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return data


def _call_llm(question: str) -> dict[str, Any]:
    schema = {
        "crop": "小麦|玉米|水稻|null",
        "region": "东北|黄淮海|长江中下游|西北|华北|西南|江淮|江汉|江南|华南|全国|null",
        "growth_stage": "春播|夏播|秋冬种|冬前|春季|播种|育秧|插秧|返青|苗期|分蘖|拔节|孕穗|抽穗|开花|灌浆|中后期|成熟|null",
        "planting_method": "直播|移栽|机插|机播|水肥一体化|滴灌|null",
        "issues": ["从问题原文提取的现象，不补写"],
        "intent": "简短检索意图",
        "ambiguous_stage": "true|false",
        "search_queries": ["最多3条，不含答案的标准农业检索语句"]
    }
    system = f"""你是农业 RAG 的查询理解器，不是农业顾问。只从用户原话抽取条件，绝不补充技术措施、日期、剂量或未说出的地区/阶段。\n
返回严格 JSON，不能使用 Markdown 或解释。字段格式：{json.dumps(schema, ensure_ascii=False)}。\n
规则：\n1. crop、region、growth_stage、planting_method 只能使用枚举值或 null。\n2. "刚种"、"种的时候"等无法区分播种、育秧、插秧的表达，growth_stage 必须为 null，ambiguous_stage 必须为 true。\n3. 当作物、地区或生育期没有在原话中明确出现或可由口语别名唯一确定时，填 null，不能猜。\n4. search_queries 只用于检索，不能输出农业建议；若字段不明确，可以只保留原问题和已明确条件，不能补猜缺失条件。"""
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "temperature": 0.0,
        "max_tokens": 360,
    }
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return _extract_json(json.loads(response.read())["choices"][0]["message"]["content"])


def _confirmed_scope(data: dict[str, Any]) -> list[str]:
    """只返回用户已明确说出的范围；缺失字段绝不能由改写器补造。"""
    labels = (("作物", "crop"), ("地区", "region"), ("生育期", "growth_stage"))
    return [f"{label}={data[key]}" for label, key in labels if data.get(key)]


def _rewrite_with_llm(question: str, data: dict[str, Any]) -> list[str]:
    """将口语改成检索表达，即使范围不完整也只保留已确认的信息。"""
    confirmed = "；".join(_confirmed_scope(data)) or "无（只能保留原问题，不能猜作物、地区或生育期）"
    system = f"""你是农业资料检索的 Query Rewrite 模块，不是农业顾问。返回严格 JSON：{{"queries":["..."]}}，不能使用 Markdown 或解释。
已确认且不得修改的范围：{confirmed}。
根据原问题生成 2 条不同措辞的标准农业检索语句。每条必须保留所有已确认范围字段和原问题中的现象/目标；缺失字段保持缺失，不得猜测作物、地区、生育期、技术措施、剂量、药剂、时间或其他事实。"""
    payload = {
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "temperature": 0.0,
        "max_tokens": 220,
    }
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        parsed = _extract_json(json.loads(response.read())["choices"][0]["message"]["content"])
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError("改写输出缺少 queries 数组")
    return [query for query in queries if isinstance(query, str)]


def _template_rewrites(question: str, data: dict[str, Any]) -> list[str]:
    """模型不可用或漏字段时的保底改写；只组合已确认条件，不制造农业事实。"""
    issue = "、".join(data["issues"]) or data["intent"] or question
    prefix = "".join(value for value in (data.get("region"), data.get("crop"), data.get("growth_stage")) if value)
    if not prefix:
        return [question, f"粮食作物田间管理 {issue}"]
    return [f"{prefix}{issue}", f"{prefix}田间管理 {issue}"]


def _valid(value: Any, allowed: tuple[str, ...]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _missing_message(data: dict[str, Any], question: str) -> str | None:
    missing = []
    if not data["crop"]:
        missing.append("作物（小麦、玉米或水稻）")
    if not data["region"]:
        missing.append("种植地区")
    if not data["growth_stage"] or data["ambiguous_stage"]:
        missing.append("明确生育期或种植环节（如播种、育秧、插秧、返青、分蘖、灌浆）")
    if not missing:
        return None
    return "识别范围不完整，尚未确认：" + "；".join(missing) + f"。原问题：{question}"


def answer_language(question: str) -> str:
    """语言是展示要求，不参与范围识别和检索，避免英文词干扰农业条件。"""
    return "English" if re.search(r"英语|英文|english", question, flags=re.I) else "中文"


def understand(question: str, crop: str | None = None, region: str | None = None, stage: str | None = None, mode: str = "llm", scope_policy: str = "adaptive") -> dict[str, Any]:
    """优先 LLM 理解，失败时退回规则；adaptive 允许不完整范围继续检索。"""
    if scope_policy not in {"adaptive", "strict"}:
        raise ValueError(f"不支持的范围策略：{scope_policy}")
    base = rule_understanding(question, crop, region, stage)
    raw: dict[str, Any] = {}
    warning = None
    if mode == "llm":
        try:
            raw = _call_llm(question)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            warning = f"查询理解模型不可用，已退回规则模式：{type(exc).__name__}: {exc}"
    elif mode != "rules":
        raise ValueError(f"不支持的查询理解模式：{mode}")

    # 命令行参数优先；LLM 只能填充合法枚举，不能覆盖用户显式指定的范围。
    explicit_stage = base["growth_stage"]
    forced_ambiguity = any(phrase in question for phrase in VOCAB["ambiguous_stage_phrases"])
    data = {
        # 作物、地区和种植方式都必须由用户原话或命令行参数确认，不能让 LLM 猜测。
        "crop": crop or base["crop"],
        # 地区直接决定资料边界，禁止 LLM 在原话没有地区时补成“全国”等值。
        "region": region or base["region"],
        # 精确词表命中（如“苗棚”）比 LLM 的宽泛归类（如“苗期”）可信。
        "growth_stage": stage or explicit_stage or _valid(raw.get("growth_stage"), STAGES),
        "planting_method": base["planting_method"],
        "issues": [item for item in raw.get("issues", []) if isinstance(item, str)][:5],
        "intent": raw.get("intent", "") if isinstance(raw.get("intent", ""), str) else "",
        "ambiguous_stage": forced_ambiguity or (bool(raw.get("ambiguous_stage")) if not explicit_stage and not stage else False),
        "understanding_mode": "llm" if raw else "rules",
        "answer_language": answer_language(question),
    }
    # 可审计词表中的歧义词优先于模型的单一猜测，除非用户通过 --stage 明确指定。
    if not stage and forced_ambiguity:
        data["growth_stage"] = None
        data["ambiguous_stage"] = True
    data["missing_message"] = _missing_message(data, question)
    raw_queries = []
    rewrite_mode = "none"
    # 严格模式会在主程序停止；自适应模式则用已确认字段做检索增强。
    if scope_policy == "adaptive" or not data["missing_message"]:
        if mode == "llm":
            try:
                raw_queries = _rewrite_with_llm(question, data)
                rewrite_mode = "llm"
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                warning = (warning + "；" if warning else "") + f"Query Rewrite 不可用，已使用模板改写：{type(exc).__name__}: {exc}"
        if not raw_queries:
            raw_queries = _template_rewrites(question, data)
            rewrite_mode = "template"
    cleaned_queries = []
    for query in raw_queries:
        confirmed_values = [data[key] for key in ("crop", "region", "growth_stage") if data.get(key)]
        has_confirmed_scope = all(value in query for value in confirmed_values) if isinstance(query, str) else False
        if isinstance(query, str) and has_confirmed_scope and 12 <= len(query) <= 120 and query not in cleaned_queries:
            cleaned_queries.append(query)
    can_retrieve = scope_policy == "adaptive" or not data["missing_message"]
    if can_retrieve and not cleaned_queries:
        # LLM 可能漏掉已确认字段；模板只组合已确认字段，因而不扩大资料适用范围。
        cleaned_queries = _template_rewrites(question, data)
        rewrite_mode = "template" if rewrite_mode == "none" else f"{rewrite_mode}_fallback"
        warning = (warning + "；" if warning else "") + "Query Rewrite 输出未保留已确认范围，已使用模板改写。"
    # 原问题永远参与；改写只补充术语匹配与语义召回，不能替代用户原话。
    data["search_queries"] = [question, *cleaned_queries[:2]] if can_retrieve else []
    data["rewrite_mode"] = rewrite_mode
    if warning:
        data["warning"] = warning
    return data
