# AI Agent 开发学习笔记

记录 AI Agent 开发的学习过程，每个知识模块均提供 **TypeScript（NestJS）** 和 **Python** 双语实现，方便对比学习。

---

## 项目结构

```
.
├── <module>/          # TypeScript 实现（NestJS + LangChain.js）
└── <module>-py/       # Python 实现（LangChain / LangGraph）
```

---

## 学习模块

### 1. Hello LangChain — 入门集成

| 目录                      | 说明                                                        |
| ------------------------- | ----------------------------------------------------------- |
| `hello-nest-langchain`    | NestJS 中接入 LangChain，调用 OpenAI 模型的最简示例         |
| `hello-nest-langchain-py` | Python：FastAPI + LangChain，含 Book CRUD、AI 聊天（SSE）   |

---

### 2. Prompt Template — 提示词模板

| 目录                      | 说明                                                      |
| ------------------------- | --------------------------------------------------------- |
| `prompt-template-test`    | JS：基础模板、聊天模板、Few-shot、示例选择器、Prompt 管道 |
| `prompt-template-test-py` | Python：同上，使用 LangChain 1.0                          |

涵盖内容：动态变量注入、Few-shot 示例构造、基于向量相似度的示例选择（Milvus）、Prompt 链式组合。

---

### 3. Output Parser — 输出解析

| 目录                    | 说明                                                        |
| ----------------------- | ----------------------------------------------------------- |
| `output-parser-test`    | JS：JSON、结构化输出、Tool Call 解析、XML、流式解析等 13 种 |
| `output-parser-test-py` | Python：结构化解析 + MySQL 数据库集成                       |

涵盖内容：Zod / Pydantic 类型约束、StructuredOutputParser、工具调用解析、流式响应处理。

---

### 4. Runnable — 链式调用

| 目录               | 说明                                          |
| ------------------ | --------------------------------------------- |
| `runnable-test`    | JS：Runnable 序列、并行、分支、MCP 适配器集成 |
| `runnable-test-py` | Python：同上，LangChain Core 1.0 实现         |

涵盖内容：`RunnableSequence`、`RunnableParallel`、`RunnableBranch`、异步执行、Model Context Protocol（MCP）接入。

---

### 5. Tool — 工具调用与 MCP

| 目录           | 说明                                                        |
| -------------- | ----------------------------------------------------------- |
| `tool-test`    | JS：自定义工具、文件系统工具、MCP Server 实现与调用         |
| `tool-test-py` | Python：工具定义、MCP Client 集成（含 React Todo App 示例） |

涵盖内容：`@tool` 装饰器、工具组合、MCP 协议服务端与客户端、Agent 调用工具链。

---

### 6. RAG — 检索增强生成

| 目录          | 说明                                               |
| ------------- | -------------------------------------------------- |
| `rag-test`    | JS：文档加载（网页/本地）、文本分割、RAG 管道构建  |
| `rag-test-py` | Python：BeautifulSoup 网页抓取、文本分割、RAG 实现 |

涵盖内容：Document Loader、TextSplitter（按 token/字符/递归分割）、Embedding + 向量检索、问答链。

---

### 7. Milvus — 向量数据库

| 目录             | 说明                                          |
| ---------------- | --------------------------------------------- |
| `milvus-test`    | JS：向量的增删改查、RAG 实战、EPUB 电子书处理 |
| `milvus-test-py` | Python：pymilvus 操作、电子书向量化入库与检索 |

涵盖内容：Collection 管理、向量相似度搜索、Embedding 生成、结合 RAG 实现电子书语义问答。

---

### 8. Memory — 对话记忆

| 目录             | 说明                                         |
| ---------------- | -------------------------------------------- |
| `memory-test`    | JS：内存对话历史、基于 Milvus 的持久化记忆   |
| `memory-test-py` | Python：LangChain 记忆模块 + Milvus 持久存储 |

涵盖内容：`InMemoryChatMessageHistory`、多轮对话上下文管理、向量数据库持久化记忆、记忆检索注入。

---

### 9. LangGraph — 工作流与 Agent 编排

| 目录                | 说明                                                             |
| ------------------- | ---------------------------------------------------------------- |
| `langgraph-test`    | JS：基础图、条件路由、Checkpoint 持久化、中断恢复、多 Agent 监督 |
| `langgraph-test-py` | Python：同上，LangGraph 1.0 Python 实现                          |

涵盖内容：State Graph 定义、节点与边、条件分支、SQLite Checkpoint、Human-in-the-loop 中断、Supervisor 多 Agent 协作、预置 ReAct Agent。

---

### 10. Cron Job Tool — 定时任务 Agent

| 目录            | 说明                                                                    |
| --------------- | ----------------------------------------------------------------------- |
| `cron-job-tool` | NestJS 定时任务 + LangChain Agent，集成网络搜索、邮件发送、MySQL 数据库 |

涵盖内容：`@nestjs/schedule` 定时调度、Agent 自主调用工具执行任务、TypeORM 数据持久化。

---

### 11. TTS / ASR — 语音合成与识别

| 目录                       | 说明                                                     |
| -------------------------- | -------------------------------------------------------- |
| `tts-stt-test`             | 腾讯云 TTS/ASR API 基础调用，生成音频文件，流式语音合成  |
| `asr-and-tts-nest-service` | NestJS + WebSocket 实时语音服务，集成 LangChain 对话能力 |

涵盖内容：文字转语音（TTS）、语音转文字（ASR）、WebSocket 实时双工通信、流式音频输出。

---

### 12. AG-UI — Agent 可视化交互界面

| 目录            | 说明                                                   |
| --------------- | ------------------------------------------------------ |
| `agui-backend`  | NestJS 后端，提供 AI Agent 接口，支持流式响应          |
| `agui-frontend` | React 19 前端，流式 Markdown 渲染，Tool 调用面板可视化 |

涵盖内容：AG-UI 协议、AI SDK 流式接口、前后端联调、实时工具调用状态展示。

---

### 13. Advanced RAG — 高级检索增强生成

| 目录              | 说明                                                             |
| ----------------- | ---------------------------------------------------------------- |
| `advanced-rag`    | JS：四种高级 RAG 策略，LangChain.js + LangGraph + Milvus 实现   |
| `advanced-rag-py` | Python：同上，LangChain + LangGraph Python 实现，支持 Web 搜索  |

涵盖内容：

- **Naive RAG**：基础检索-生成管道
- **Query Router RAG**：根据问题复杂度智能路由，简单问题直接由 LLM 回答，复杂问题走向量检索
- **Multi-hop RAG**：将复杂问题拆解为子问题，迭代检索与规划，逐步聚合答案
- **Web Fallback RAG**：本地 Milvus 检索为主，结合 Bocha 网络搜索兜底，提升知识覆盖率

---

## 技术栈

| 分类       | 技术                                              |
| ---------- | ------------------------------------------------- |
| AI 框架    | LangChain.js / LangChain Python（1.0）、LangGraph |
| LLM        | OpenAI API                                        |
| 向量数据库 | Milvus 2.x                                        |
| 后端框架   | NestJS（Node.js）、FastAPI（Python）              |
| 前端框架   | React 19 + Vite + TailwindCSS                     |
| 语音服务   | 腾讯云 TTS / ASR                                  |
| 工具协议   | Model Context Protocol（MCP）                     |
| 数据库     | MySQL（TypeORM）、SQLite（LangGraph Checkpoint）  |
| 语言       | TypeScript、Python 3.x                            |

---

## 快速开始

### TypeScript 项目

```bash
cd <module>
npm install
cp .env.example .env   # 填写 API Key 等配置
npm run start:dev
```

### Python 项目

```bash
cd <module>-py
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env         # 填写 API Key 等配置
python src/main.py
```
