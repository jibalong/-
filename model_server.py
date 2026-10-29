#!/usr/bin/env python3
"""仅加载本地 Qwen2.5-3B-Instruct 基座模型的 OpenAI 兼容服务（端口 8002）。"""
from __future__ import annotations

import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/ai-workspace/models/Qwen2.5-3B-Instruct"
app = FastAPI(title="rag_grains Qwen base-model server")
tokenizer = None
model = None


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str = "Qwen2.5-3B-Instruct"
    messages: list[Message]
    temperature: float = 0.1
    max_tokens: int = 700


def load_model() -> None:
    global tokenizer, model
    if model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype, device_map="auto", trust_remote_code=False)
    model.eval()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH, "adapter_loaded": False}


@app.post("/v1/chat/completions")
def chat(request: CompletionRequest) -> dict:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    messages = [{"role": item.role, "content": item.content} for item in request.messages]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=request.max_tokens, do_sample=request.temperature > 0, temperature=max(request.temperature, 0.01), pad_token_id=tokenizer.eos_token_id)
    new_tokens = generated[0][inputs.input_ids.shape[1]:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": "Qwen2.5-3B-Instruct", "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    load_model()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
