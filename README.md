# Minimal RAG

一个从零开始、可本地运行的 RAG 小项目。

流程是：

```text
documents -> chunk -> embed -> store
query -> embed -> retrieve -> prompt -> answer
```

这个版本刻意保持简单：

- 不需要安装依赖也能跑检索。
- 有 `OPENAI_API_KEY` 时会调用 OpenAI 生成最终回答。
- 没有 API key 时会返回检索到的上下文摘要和来源，方便你先理解 RAG 的骨架。

## Quick Start

```bash
python3 -m raglite ingest docs
python3 -m raglite ask "RAG 的索引阶段做什么？"
```

索引会写到 `.rag_index/index.json`。

## Learn It Module By Module

这个项目按 RAG 的真实流水线拆成几个模块。建议你按下面顺序跑：

### 1. Chunking

先看文档如何被切成 chunks：

```bash
python3 -m raglite chunks docs --chunk-size 80 --overlap 20
```

对应模块：`raglite/chunking.py`

你要观察两件事：

- 每个 chunk 是否有足够上下文。
- overlap 是否让相邻 chunk 保留了重复内容。

### 2. Embedding + Store

把 chunks 变成 vectors，然后写入本地 index：

```bash
python3 -m raglite ingest docs --chunk-size 80 --overlap 20
```

对应模块：

- `raglite/embeddings.py`
- `raglite/store.py`

这个学习版使用本地 `HashingEmbedder`，目的是先理解流程。真实项目里可以换成 OpenAI embedding。

### 3. Retrieval

只看检索结果，不让 LLM 生成：

```bash
python3 -m raglite search "What happens during indexing?"
```

对应模块：`raglite/store.py`

如果这一步找错了 chunk，后面的答案通常也会错。

### 4. Prompt

看最终会发给 LLM 的 prompt：

```bash
python3 -m raglite prompt "What happens during indexing?"
```

对应模块：`raglite/llm.py`

RAG 的 prompt 通常包含：系统约束、retrieved context、用户问题。

### 5. Answer

最后再生成回答：

```bash
python3 -m raglite ask "What happens during indexing?"
```

没有 `OPENAI_API_KEY` 时，它会显示检索到的 chunks。有 key 时，它会调用 LLM 基于这些 chunks 回答。

## Add Your Own Docs

把 `.txt` 或 `.md` 文件放进 `docs/`，然后重新索引：

```bash
python3 -m raglite ingest docs
python3 -m raglite ask "你的问题"
```

## Use OpenAI For Generation

```bash
export OPENAI_API_KEY="你的 key"
python3 -m raglite ask "RAG 检索到内容以后怎么回答？"
```

默认模型是 `gpt-4.1-mini`，可以改：

```bash
python3 -m raglite ask "你的问题" --model gpt-4.1
```

## Project Layout

```text
raglite/
  chunking.py    文档切块
  embeddings.py  本地哈希 embedding
  store.py       JSON 向量索引和 cosine 检索
  llm.py         OpenAI 生成或本地 fallback
  cli.py         命令行入口和学习用 debug 命令
docs/
  sample.md      示例知识库
```

## Next Upgrades

这个最小版本跑通以后，常见升级路线是：

1. 把 `HashingEmbedder` 换成 `text-embedding-3-small`。
2. 把 JSON store 换成 Chroma、FAISS、LanceDB 或 pgvector。
3. 增加 PDF/网页加载器。
4. 做评测：准备一组问题，检查 retrieved chunks 和答案引用是否正确。
