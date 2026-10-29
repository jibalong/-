# Codex 协作说明：粮食作物 RAG

本文件适用于 `/root/ai-workspace/rag_grains/` 及子目录，同时遵守上级 `../AGENTS.md`。本项目实际名称是 `rag_grains`，不要另建 `rag_grain` 目录。

## 用户与协作方式

用户通过 Codex 边做项目边学习，暂时没有开发基础。默认用中文交流、写文档与注释，命令和代码标识保留原样。

- 接任务后，先用几句话说明要解决的问题、涉及文件和验证方式，然后在授权范围内完成工作。
- 普通可逆修改可直接推进；不要把每个实现细节都变成确认问题。
- 讲解时先给整体流程，再解释本次涉及的少量概念。首次出现术语时，配一句中文解释。
- 给命令时写清运行目录、是否需要激活环境、在哪个终端执行，以及成功时看什么。
- 用户要求“带我学、一步一步来”时，一次只解释一小部分，等待用户理解后再继续；用户要求直接实现时，完成实现和必要验证。
- 交付说明：改了什么、为什么、怎样运行、实际验证结果、仍有什么限制。
- 不用“测试通过”代替证据；说明测的是检索、查询理解、HTTP 服务还是完整回答。

## 开始前阅读

1. `README.md`：面向初学者的运行、文件和命令参考。
2. `ARCHITECTURE.md`：实现细节和技术取舍。
3. 按需阅读 `PROJECT_EXPLAINER.md` 与相关脚本。
4. 如果目录属于 Git 仓库，查看状态和差异。当前项目尚不是 Git 仓库，不要假称已提交或能用 Git 回滚。

文档中的历史指标、包版本和服务状态都需要按任务重新核实。

## 环境与模型边界

使用主虚拟环境：

```bash
cd /root/ai-workspace/rag_grains
source /root/ai-workspace/llm-dev-env/bin/activate
```

| 组件 | 路径或约定 |
| --- | --- |
| 生成与查询理解 | `/root/ai-workspace/models/Qwen2.5-3B-Instruct` |
| 向量模型 | `models/bge-m3/` |
| 重排模型 | `models/bge-reranker-v2-m3/` |
| Qwen 服务 | `model_server.py`，`127.0.0.1:8002` |
| 健康检查 | `GET /health` |
| 模型调用 | `POST /v1/chat/completions` |

本项目使用 Qwen 基座，不加载猫咪 LoRA。不要把 `rag_test` 的封闭虚构资料提示词、猫咪口吻或 8000 端口搬进来。8001 属于另一个农业示例。

主要依赖是 `torch`、`transformers`、`accelerate`、`numpy`、`FlagEmbedding`、`fastapi`、`uvicorn`、`pydantic`。使用现有环境，避免无关升级。需要安装依赖时，用 `/root/.local/bin/uv pip install --python /root/ai-workspace/llm-dev-env/bin/python ...` 明确指定环境；不得向系统 Python 安装。

## 数据与回答约定

- `sources.json` 登记来源；`raw/` 保存原始网页证据；`documents/` 保存程序清洗的正文；`index.json` 为生成的检索索引。
- 不改写原始快照，不用模型补写农业技术正文。当前清洗脚本只处理支持的 HTML 结构；新增 PDF/Word 需要单独实现和核对提取步骤。
- 新资料须保留机构、标题、日期、URL、地区、生育期等来源信息，并补针对性测试。
- 修改正文、分块或索引所用词表后，按需运行清洗/索引脚本。它们会覆盖生成文件；日常提问不要重建索引。
- 默认 `adaptive`：按已确认范围继续检索并提示缺失字段；`strict`：条件不足时停止。不得擅自更改默认策略。
- 保留作物、地区、生育期过滤及现象直接证据检查；不能用高温证据回答冻害问题。
- 回答依托检索证据，保留引用与适用范围。不用模型常识补农业处方、药剂剂量或安全间隔期。
- 引用格式通过不代表事实正确，仍需逐句核对原文。
- 统一题集是 `tests/questions.jsonl`。不要为了提高分数删难题、修改预期或只报告有利指标。

## 修改与验证

先运行与改动直接相关的最小检查。纯文档修改核对路径、命令和 CLI 参数即可，不必启动 GPU 服务或跑整套题。

轻量检索检查，无需启动 Qwen，也不加载向量/重排模型：

```bash
python rag_check.py '东北水稻育秧低温如何防范？' --understanding rules --retrieval bm25 --no-log
```

检索回归示例（会写报告，默认规则理解，不调用 Qwen）：

```bash
PYTHONPATH=. python tests/evaluate_retrieval.py --retrieval bm25
```

混合检索相关改动按需使用 `--retrieval hybrid`；这会加载 bge-m3 和重排模型，可能占用显存。用 `diagnostics.used_mode` 核对实际模式，不能把退回 BM25 的结果称为混合检索成功。

`--retrieve-only` 和 `rag_check.py` 只关闭最终回答，默认查询理解仍可能调用 Qwen；完全不调用 Qwen 时加 `--understanding rules`。

涉及生成或查询理解改动时，检查显存与端口后再启动：

```bash
python model_server.py --host 127.0.0.1 --port 8002
```

另一个终端激活主环境，执行：

```bash
curl http://127.0.0.1:8002/health
python rag_chat.py '东北水稻育秧低温如何防范？'
```

需要批量验证时再运行相关 `tests/evaluate_*.py`；完整人工审查报告由 `PYTHONPATH=. python tests/run_question_review.py` 生成，会覆盖默认报告。不要把旧报告说成本次实测结果。

## 文件与资源保护

- 本任务通常只应影响本项目。保护共享的基座模型、虚拟环境、其他 RAG 项目和已有 LoRA 输出。
- 保护用户改动、原始资料与历史记录。删除或覆盖前明确目标和任务授权。
- 用 `apply_patch` 编辑文件；不用破坏性 Git 操作或大范围删除。
- 启动 GPU 任务前检查 `nvidia-smi`；不要随意终止其他服务。自己为测试启动的服务用完后说明是否仍在运行。
- README 是用户的操作入口。改变命令、模型、端口、默认行为或文件职责时，同步更新它。
