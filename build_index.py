#!/usr/bin/env python3
"""为粮食作物资料构建可审计、无外部依赖的中文 BM25 索引。"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL_PATH = ROOT / "models" / "bge-m3"
VOCAB = json.loads((ROOT / "domain_vocab.json").read_text(encoding="utf-8"))
LEVEL_ONE_RE = re.compile(r"^([一二三四五六七八九十]+、[^\n]{0,50})$")
LEVEL_TWO_RE = re.compile(r"^(（[一二三四五六七八九十]+）[^。；：\n]{0,50})[。；：]\s*(.*)$")


def tokenize(text: str) -> list[str]:
    """中文以连续汉字二元词为主，同时保留字母/数字词，便于可解释检索。"""
    words: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9.%]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", run):
            words.extend(run[i : i + 2] for i in range(max(0, len(run) - 1)))
            if len(run) == 1:
                words.append(run)
        else:
            words.append(run)
    return words


def detect_growth_stages(section: str, text: str) -> list[str]:
    """为块补充词表驱动的阶段标签；原文不变，标签只帮助范围过滤。"""
    haystack = section + "\n" + text
    return [stage for stage, aliases in VOCAB["growth_stages"].items() if any(alias in haystack for alias in aliases)]


def split_document(path: Path, source: dict) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        begin = lines.index("## 正文（由官方网页快照提取，未补写技术内容）") + 1
    except ValueError as exc:
        raise ValueError(f"{path} 缺少受支持正文标记") from exc
    paragraphs = [line.strip() for line in lines[begin:] if line.strip()]
    # 用两个层级保存章节路径。这样“病虫害防控 > 北方春玉米区”不会丢失上级语境。
    level_one = "正文"
    level_two = ""
    chunks: list[dict] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        section = " > ".join(part for part in (level_one, level_two) if part)
        text = "\n".join(buffer)
        if len(text) >= 30:
            chunks.append(
                {
                    "source": source["id"], "title": source["title"], "section": section,
                    "text": text, "crop": source["crop"], "region": source["region"],
                    "growth_stages": detect_growth_stages(section, text),
                    "season_or_growth_stage": source["season_or_growth_stage"],
                    "topic": source["topic"], "published_at": source["published_at"],
                    "source_url": source["source_url"], "publisher": source["publisher"],
                }
            )
        buffer.clear()

    for para in paragraphs:
        # 一级标题必须是一整段短文本，例如“二、防控策略”。
        # 这避免把正文中“第一、第二”等普通表述误判为标题。
        if LEVEL_ONE_RE.match(para):
            flush()
            level_one = para
            level_two = ""
            continue
        # 网页常将“（一）标题。正文”放进同一个 HTML 段落。
        # 这里仅剥离明显的短标题，正文仍作为一个完整块保留。
        level_two_match = LEVEL_TWO_RE.match(para)
        if level_two_match:
            flush()
            level_two = level_two_match.group(1)
            para = level_two_match.group(2)
            if not para:
                continue
        # 约 600 汉字为一块；不跨越章节，也不拆开段落/剂量语句。
        if buffer and sum(len(x) for x in buffer) + len(para) > 650:
            flush()
        buffer.append(para)
    flush()
    return chunks


def vector_input(chunk: dict) -> str:
    """将会影响语义的元数据与正文一起送入 embedding，增强区域和阶段区分。"""
    return "\n".join(
        [
            f"作物：{chunk['crop']}",
            f"地区：{chunk['region']}",
            f"生育期：{chunk['season_or_growth_stage']}",
            f"主题：{chunk['topic']}",
            f"标题：{chunk['title']}",
            f"章节：{chunk['section']}",
            f"正文：{chunk['text']}",
        ]
    )


def add_dense_vectors(chunks: list[dict]) -> dict:
    """用本项目本地的 bge-m3 为全部资料块编码，不上传农业资料。"""
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到 bge-m3 模型：{EMBEDDING_MODEL_PATH}")
    from FlagEmbedding import BGEM3FlagModel

    # bge-m3 输出已归一化的稠密向量，因此后续点积可作为余弦相似度。
    model = BGEM3FlagModel(str(EMBEDDING_MODEL_PATH), use_fp16=True)
    vectors = model.encode(
        [vector_input(chunk) for chunk in chunks],
        batch_size=8,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    vectors = np.asarray(vectors, dtype=np.float32)
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["dense_vector"] = vector.tolist()
    return {"model": "BAAI/bge-m3", "local_path": str(EMBEDDING_MODEL_PATH), "dimension": int(vectors.shape[1]), "normalized": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="从 documents/ 和 sources.json 重建 BM25 索引")
    parser.add_argument("--output", default="index.json", help="输出索引文件（默认 index.json）")
    parser.add_argument("--skip-vectors", action="store_true", help="只建 BM25；默认会在本地 bge-m3 存在时建立向量")
    args = parser.parse_args()
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    chunks = []
    for source in sources:
        chunks.extend(split_document(ROOT / source["document_file"], source))
    if not chunks:
        raise SystemExit("没有可索引分块；请先运行 prepare_documents.py")

    doc_freq: Counter[str] = Counter()
    total_length = 0
    for chunk in chunks:
        tokens = tokenize(" ".join([chunk["title"], chunk["section"], chunk["text"], chunk["topic"]]))
        chunk["term_freq"] = dict(Counter(tokens))
        chunk["length"] = len(tokens)
        total_length += len(tokens)
        doc_freq.update(chunk["term_freq"].keys())
    index = {
        "schema_version": 1,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "algorithm": {"name": "BM25", "tokenizer": "中文连续字串二元词 + 字母/数字词", "k1": 1.5, "b": 0.75},
        "documents": len(sources), "chunks": chunks, "document_frequency": dict(doc_freq),
        "average_length": total_length / len(chunks),
    }
    if args.skip_vectors:
        index["vector"] = {"enabled": False, "reason": "用户使用 --skip-vectors"}
    elif EMBEDDING_MODEL_PATH.exists():
        print("正在使用本地 bge-m3 构建稠密向量...")
        index["vector"] = {"enabled": True, **add_dense_vectors(chunks)}
    else:
        index["vector"] = {"enabled": False, "reason": f"模型未下载：{EMBEDDING_MODEL_PATH}"}
        print("未找到本地 bge-m3，本次仅建立 BM25；下载模型后重新运行即可加入向量。")
    output = ROOT / args.output
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    mode = "BM25 + bge-m3 向量" if index["vector"]["enabled"] else "仅 BM25"
    print(f"已建立 {output.name}：{len(sources)} 份资料、{len(chunks)} 个分块，模式：{mode}。")


if __name__ == "__main__":
    main()
