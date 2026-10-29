#!/usr/bin/env python3
"""从 raw/ 的农业农村部网页快照生成带可追溯元数据的 Markdown。"""
from __future__ import annotations

import json
import re
from datetime import date
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def clean_html(fragment: str) -> str:
    """保留文本和段落边界；网页导航、样式与脚本不会进入 documents。"""
    fragment = re.sub(r"<(script|style|noscript)\b[^>]*>.*?</\1>", "", fragment, flags=re.I | re.S)
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"</(?:p|div|li|tr|h[1-6])\s*>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<t[dh][^>]*>", " | ", fragment, flags=re.I)
    text = unescape(re.sub(r"<[^>]+>", "", fragment))
    text = text.replace("\u3000", " ").replace("&nbsp;", " ")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def editor_fragment(html: str) -> str:
    """取得官方页面正文容器，按 div 嵌套层级匹配结束标签。"""
    start = re.search(
        r"<div\b[^>]*\bclass\s*=\s*(?:TRS_Editor\b|['\"][^'\"]*\barc_body\b[^'\"]*['\"])[^>]*>",
        html,
        re.I,
    )
    if not start:
        raise ValueError("未找到受支持的官方正文容器（TRS_Editor 或 arc_body）")
    pos, depth = start.end(), 1
    tag_re = re.compile(r"</?div\b[^>]*>", re.I)
    for tag in tag_re.finditer(html, pos):
        if tag.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                return html[pos:tag.start()]
        else:
            depth += 1
    raise ValueError("正文容器未闭合")


def main() -> None:
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        raw_path = ROOT / source["raw_file"]
        html = raw_path.read_text(encoding="utf-8", errors="replace")
        title = re.search(r"<title>\s*(.*?)\s*</title>", html, re.I | re.S)
        if not title:
            raise ValueError(f"{raw_path}: 未找到标题")
        page_title = clean_html(title.group(1))
        if page_title != source["title"]:
            raise ValueError(f"{raw_path}: 网页标题与登记册不一致：{page_title}")
        body = clean_html(editor_fragment(html))
        if len(body) < 300:
            raise ValueError(f"{raw_path}: 正文过短，停止生成以避免错误入库")
        metadata = [
            f"# {source['title']}",
            "",
            f"- 来源机构：{source['publisher']}",
            f"- 发布日期：{source['published_at']}",
            f"- 原始 URL：{source['source_url']}",
            f"- 抓取日期：{source['fetched_at']}",
            f"- 作物：{source['crop']}",
            f"- 适用地区：{source['region']}",
            f"- 种植季节或生育期：{source['season_or_growth_stage']}",
            f"- 主题：{source['topic']}",
            f"- 文件格式：{source['file_format']}",
            "",
            "## 正文（由官方网页快照提取，未补写技术内容）",
            "",
            body,
            "",
        ]
        (ROOT / source["document_file"]).write_text("\n".join(metadata), encoding="utf-8")
        print(f"已生成 {source['document_file']}（正文 {len(body)} 字符）")


if __name__ == "__main__":
    main()
