# 掌柜智库 (Shopkeeper Brain)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-FF6B6B.svg)](https://langchain-ai.github.io/langgraph/)
[![MinerU](https://img.shields.io/badge/MinerU-3.4.5-8B5CF6.svg)](https://github.com/opendatalab/MinerU)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

基于 LangGraph 的企业级知识库 RAG（检索增强生成）系统，面向零售/快消行业场景，支持商品资料文档的智能导入和自然语言查询。

---

## 目录

- [项目定位](#项目定位)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [架构设计](#架构设计)
- [目录结构](#目录结构)
- [工作流详解](#工作流详解)
  - [导入工作流](#导入工作流)
  - [查询工作流](#查询工作流)
- [数据流](#数据流)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装依赖](#安装依赖)
  - [应用补丁](#应用补丁-重要)
  - [下载模型](#下载模型)
  - [配置环境变量](#配置环境变量)
  - [启动中间件](#启动中间件)
  - [启动服务](#启动服务)
- [API 接口](#api-接口)
- [配置说明](#配置说明)
- [常见问题排查](#常见问题排查)
- [许可证](#许可证)

---

## 项目定位

面向零售店员的垂直领域 RAG 问答系统。把商品手册、使用说明等 PDF 文档导入后，店员自然语言提问即可获得基于文档的回答。

```
                    ┌─────────────┐      ┌──────────────┐
                    │  前端网页    │      │  CentOS 虚拟机 │
     上传 PDF ───►  │ 导入页       │ ───► │              │
                    │ :8000        │      │ Milvus       │
     提问 ───────►  │ 聊天页       │ ───► │ MongoDB      │
                    │ :8001        │      │ MinIO        │
                    └─────────────┘      └──────────────┘
                            │
                    ┌───────▼────────┐
                    │  本地 Windows    │
                    │  BGE-M3 (GPU)   │
                    │  MinerU CPU     │
                    │  DashScope API  │
                    └─────────────────┘
```

---

## 核心特性

- **双通道架构**：文档导入流水线 + 查询回答流水线，由 LangGraph 状态机编排
- **混合检索**：Dense（向量）+ Sparse（BM25 关键词）+ HyDE 三路召回，RRF 融合排序
- **商品名智能识别**：专有的商品名实体识别节点，支持多候选确认
- **BGE Reranker 重排**：召回后二次精排，显著提升答案相关性
- **ReAct Agent 模式**：可选的多轮工具调用，支持 MCP DashScope 联网搜索
- **SSE 流式输出**：查询结果和导入进度实时推送
- **子进程隔离**：MinerU 解析在独立 OS 子进程中运行，C 扩展崩溃不影响主服务
- **全链路可观测**：任务状态追踪、耗时统计、历史记录持久化

---

## 技术栈

| 类别 | 技术 | 版本 | 作用 |
|---|---|---|---|
| 语言 | Python | 3.12 | 开发语言 |
| Web 框架 | FastAPI | 0.141 | HTTP 接口 |
| ASGI 服务器 | Uvicorn | 0.52 | 跑 FastAPI |
| 工作流编排 | **LangGraph** | 1.2 | 各节点串成 DAG |
| 文档解析 | **MinerU** | 3.4.5 | PDF → Markdown（OCR + 版面分析） |
| 向量数据库 | **Milvus** | 3.0 | 存文档 embedding，做向量检索 |
| 文档数据库 | MongoDB | 4.18 | 存元数据、对话历史 |
| 对象存储 | MinIO | 7.2 | 存 MinerU 抽出的图片 |
| Embedding | **BGE-M3** | — | 混合 embedding（稠密 + 稀疏 BM25） |
| 重排序 | BGE-Reranker-Large | — | 召回后精排 |
| LLM | 阿里云 DashScope qwen-flash | — | 改写、商品识别、生成答案 |
| 深度学习后端 | PyTorch | 2.5.1+cu121 | BGE + MinerU 推理 |
| 模型加载器 | Transformers | 5.17 | HuggingFace 加载 |
| OCR 推理 | ONNX Runtime | 1.30 | MinerU OCR 模型 |
| 数据校验 | Pydantic | 2.13 | API 请求/响应体 |

---

## 架构设计

### 导入流水线 (Import Graph)

```
文件上传 → entry_node → pdf_to_md_node(MinerU 子进程)
                        → md_to_img_node(表格 HTML → 图片)
                        → document_split_node(按段落切 chunk)
                        → item_name_recognition_node(qwen-flash 识别商品名)
                        → embedding_chunks_node(BGE-M3 GPU 混合 embedding)
                        → import_milvus_node(写入 Milvus + MongoDB)
```

### 查询流水线 (Query Graph)

```
用户查询
  → item_name_confirmed_node(qwen-flash 提取商品名，模糊时反问)
  → multi_search（并行分发）
    ├── hybrid_vector_search_node(Milvus 混合检索)
    ├── hyde_vector_search_node(LLM 假设性答案 → 向量检索)
    └── web_mcp_search_node(DashScope Web Search MCP 联网)
  → join → rrf_merge_node(RRF 倒数排名融合)
  → reranker_node(BGE-Reranker-Large GPU 精排)
  → answer_output_node(qwen-flash 生成最终回答)
  → SSE 流式返回前端
```

### 核心设计模式

- **双重检查锁单例**（`main_graph.py`）：LangGraph 编译图懒加载，避免模块 import 时连数据库崩溃
- **TypedDict 状态传递**（`state.py`）：`total=False`，节点按需往 state 里加字段
- **BaseNode 统一封装**（`base.py`）：日志、状态校验、SSE 进度推送
- **子进程隔离**（`pdf_to_md_node.py`）：MinerU 在 `subprocess.run()` 里跑，C 扩展崩溃不影响 FastAPI
- **分层 GPU/CPU 策略**：BGE-M3 主进程用 GPU，MinerU 子进程禁 GPU（避免重复加载 cublas64 耗尽虚拟内存）

---

## 目录结构

```
shopkeeper_brain_1/
├── .env.example                        # 环境变量模板
├── .gitignore
├── README.md
│
├── knowledge/                          # ★ 主代码包
│   ├── api/                            # HTTP 接口层
│   │   ├── import_router.py            # 导入服务 FastAPI 应用（端口 8000）
│   │   └── query_router.py             # 查询服务 FastAPI 应用（端口 8001）
│   │
│   ├── core/                           # 依赖注入 & 路径配置
│   │   ├── deps.py                     # 公共依赖（DB 客户端等）
│   │   └── paths.py                    # 目录路径常量
│   │
│   ├── front/                          # 纯 HTML 前端（无前端框架）
│   │   ├── import.html                 # 导入页面（上传 PDF、看 SSE 进度）
│   │   └── chat.html                   # 聊天页面（输入问题、显示流式答案）
│   │
│   ├── schema/                         # Pydantic 数据校验模型
│   │   ├── upload_schema.py
│   │   └── query_schema.py
│   │
│   ├── service/                        # 业务服务层（薄封装）
│   │   ├── upload_service.py
│   │   └── query_service.py
│   │
│   ├── processor/                      # ★ LangGraph 工作流核心
│   │   ├── import_processor/           #   【导入工作流】
│   │   │   ├── main_graph.py           #   图定义 + 懒加载单例
│   │   │   ├── state.py                #   TypedDict 状态
│   │   │   ├── base.py                 #   BaseNode 基类
│   │   │   ├── config.py               #   导入配置
│   │   │   ├── exceptions.py           #   导入异常
│   │   │   └── nodes/                  #   7 个处理节点（详见工作流章节）
│   │   │
│   │   └── query_processor/            #   【查询工作流】
│   │       ├── main_graph.py
│   │       ├── state.py
│   │       ├── base.py
│   │       ├── config.py
│   │       ├── exceptions.py
│   │       ├── agent/                  #   ReAct Agent 逻辑
│   │       │   └── react_agent.py
│   │       ├── tools/                  #   Agent 工具定义
│   │       │   ├── base.py
│   │       │   └── query_tools.py
│   │       └── nodes/                  #   7 个处理节点
│   │
│   ├── prompts/                        # LLM Prompt 模板
│   │   ├── import_prompt.py
│   │   ├── query_prompt.py
│   │   └── agent_prompt.py
│   │
│   ├── utils/                          # 工具层
│   │   ├── client/                     #   外部客户端
│   │   │   ├── ai_clients.py           #     LLM + BGE 统一入口
│   │   │   ├── storage_clients.py      #     Milvus + MongoDB + MinIO
│   │   │   └── base.py
│   │   ├── milvus_util.py
│   │   ├── embedding_util.py
│   │   ├── mongo_history_util.py
│   │   ├── sse_util.py
│   │   ├── markdown_util.py
│   │   └── task_util.py
│   │
│   └── requirements.txt
│
├── patches/                            # ★ 兼容补丁（见"应用补丁"章节）
│   ├── sitecustomize.py                #   Transformers + FastText 全局补丁
│   └── mineru_patch/                   #   MinerU ONNX 强制 CPU
│       └── README.md
│
├── models/                             # 本地模型文件（需自行下载，已在 gitignore）
│   └── models/
│       ├── BAAI--bge-m3/
│       └── BAAI--bge-reranker-large/
│
├── docs/                               # 示例 PDF
└── data/tmp/                           # MinerU 临时输出（已在 gitignore）
```

---

## 工作流详解

### 导入工作流

```
entry_node ──► [PDF?] ──► pdf_to_md_node ──► md_to_img_node
                 │ 是                         │
                 │                            ▼
                 └── [MD?] ────────────── md_to_img_node
                                              │
                                              ▼
                                    document_split_node
                                              │
                                              ▼
                              item_name_recognition_node
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                          Milvus(kb_item_names_v1)  embedding_chunks_node
                                                        │
                                                        ▼
                                                  import_milvus_node
                                                        │
                                                        ▼
                                                      END
```

| 节点 | 文件 | 职责 |
|---|---|---|
| entry_node | `import_processor/nodes/entry_node.py` | 文件类型检查，往 state 写 flag |
| pdf_to_md_node | `import_processor/nodes/pdf_to_md_node.py` | **MinerU 子进程**解析 PDF → Markdown + 配图 |
| md_to_img_node | `import_processor/nodes/md_to_img_node.py` | 表格 HTML → 图片，存 MinIO |
| document_split_node | `import_processor/nodes/document_split_node.py` | 长文档按段落/标题切 chunk |
| item_name_recognition_node | `import_processor/nodes/item_name_recognition_node.py` | qwen-flash 识别商品名 |
| embedding_chunks_node | `import_processor/nodes/embedding_chunks_node.py` | BGE-M3 GPU 混合 embedding |
| import_milvus_node | `import_processor/nodes/import_milvus_node.py` | 写入 Milvus + MongoDB |

### 查询工作流

```
item_name_confirmed_node
       │
       ├─ (LLM 认为问题已有答案) ──► answer_output_node ──► END
       │
       └─ (需要检索) ──► multi_search（并行）
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
           hybrid_vec    hyde_vec    web_mcp
           _search       _search     _search
                 │          │          │
                 └──────────┼──────────┘
                            │ join
                            ▼
                       rrf_merge_node     ← RRF 倒数排名融合
                            │
                            ▼
                        reranker_node     ← BGE-Reranker-Large 精排
                            │
                            ▼
                       answer_output_node ← qwen-flash 生成答案
                            │
                            ▼
                          SSE 返回
```

| 节点 | 文件 | 职责 |
|---|---|---|
| item_name_confirmed_node | `query_processor/nodes/item_name_confirmed_node.py` | qwen-flash 提取/确认商品名 |
| hybrid_vector_search_node | `query_processor/nodes/hybrid_vector_search_node.py` | Milvus 混合检索（稠密 + BM25） |
| hyde_vector_search_node | `query_processor/nodes/hyde_vector_search_node.py` | HyDE：LLM 写假设答案 → 向量检索 |
| web_mcp_search_node | `query_processor/nodes/web_mcp_search_node.py` | DashScope Web Search MCP |
| rrf_merge_node | `query_processor/nodes/rrf_merge_node.py` | RRF `1/(k + rank)` 求和融合 |
| reranker_node | `query_processor/nodes/reranker_node.py` | BGE-Reranker-Large GPU 精排 |
| answer_output_node | `query_processor/nodes/answer_output_node.py` | qwen-flash 生成最终答案 |

---

## 数据流

### 导入链路

```
PDF 文件
  │
  ▼ MinerU 子进程 (CUDA_VISIBLE_DEVICES='')
Markdown + 配图 JSON
  │
  ▼ md_to_img_node
表格 HTML → 图片，存 MinIO
  │
  ▼ document_split_node
N 个文本 chunk（~500 字/块）
  │
  ├─► item_name_recognition_node ──► Milvus(kb_item_names_v1)
  │
  └─► embedding_chunks_node (BGE-M3 GPU)
          │
          ▼
     dense vectors + sparse vectors
          │
          ▼ import_milvus_node
     Milvus(kb_chunks_v1) + MongoDB(元数据)
```

### 查询链路

```
用户问题
  │
  ▼ item_name_confirmed_node
商品名提取 / 模糊反问
  │
  ▼ multi_search
  ├── hybrid_vector_search_node ──────────┐
  ├── hyde_vector_search_node (LLM) ───────┼──► RRF 融合
  └── web_mcp_search_node (DashScope) ─────┘       │
                                                   ▼ Reranker
                                              BGE-Reranker-Large
                                                   │
                                                   ▼
                                              qwen-flash 生成答案
                                                   │
                                                   ▼
                                             SSE 流式返回前端
```

---

## 快速开始

### 环境要求

| 组件 | 最低要求 | 推荐 |
|---|---|---|
| Python | 3.11 | **3.12** |
| GPU | 可选 | NVIDIA GPU（BGE-M3 / Reranker 用，8GB+ 显存） |
| 操作系统 | Linux / macOS | **Windows 10/11**（需应用补丁） |
| CUDA | 12.1（PyTorch 用） | **12.1**（ONNX Runtime 用 CPU 所以不强制） |
| 虚拟内存 | 8GB | **16GB+**（Windows 上多进程加载 cublas DLL 需要） |

### 安装依赖

```bash
cd shopkeeper_brain_1
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

cd knowledge
pip install -r requirements.txt
```

如果 `pip` 启动器因中文路径损坏报错，用：
```bash
.venv\Scripts\python.exe -m pip install -r knowledge\requirements.txt
```

### 应用补丁（重要！）

Windows 中文路径 + transformers 5.x + MinerU 3.4.5 存在多个兼容性问题，**必须应用补丁才能正常运行**。

#### 1. 全局补丁（`patches/sitecustomize.py`）

把这个文件复制到虚拟环境：

```bash
# Windows PowerShell:
copy patches\sitecustomize.py .venv\Lib\site-packages\sitecustomize.py

# Linux/macOS:
cp patches/sitecustomize.py .venv/lib/python3.12/site-packages/sitecustomize.py
```

**作用**（Python 启动自动加载，全局 monkey-patch）：
- transformers 5.x RTDetr `class_embed` 路径兼容
- transformers 5.17 CVE-2025-32434 torch >= 2.6 检查绕过
- FastText 中文路径 → copy 到 `C:\Temp\ftlang\`
- transformers 5.x pytorch_utils prune 函数兼容

#### 2. MinerU ONNX Runtime 补丁（`patches/mineru_patch/`）

详细改动见 `patches/mineru_patch/README.md`，需要修改两个文件：

| 文件 | 改动 |
|---|---|
| `.venv/Lib/site-packages/mineru/model/table/rec/onnxruntime_provider.py` | `build_table_onnx_providers()` 只返回 `CPUExecutionProvider` |
| `.venv/Lib/site-packages/mineru/model/table/cls/paddle_table_cls.py` | `InferenceSession(..., providers=['CPUExecutionProvider'])` |

**原因**：ONNX Runtime 1.30.0 要求 CUDA 13.x + cuDNN 9.x，强制 CPU 避免版本不匹配和 CUDA OOM。

### 下载模型

需要两个 BGE 模型（~3.5GB 总计）：

```bash
pip install modelscope
modelscope download --model BAAI/bge-m3 --local_dir models/BAAI--bge-m3
modelscope download --model BAAI/bge-reranker-large --local_dir models/BAAI--bge-reranker-large
```

最终目录结构：
```
models/
└── models/
    ├── BAAI--bge-m3/              # Embedding 模型 (~2.2GB)
    └── BAAI--bge-reranker-large/  # Reranker 模型 (~1.3GB)
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key、数据库连接串、模型路径等
```

**必须配置的关键项**：

| 变量 | 说明 | 示例 |
|---|---|---|
| `OPENAI_API_KEY` | LLM API Key（阿里云 DashScope） | `sk-xxxxxxxxxxxx` |
| `OPENAI_API_BASE` | LLM API Endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `MILVUS_URL` | Milvus 地址 | `http://localhost:19530` |
| `MONGO_URL` | MongoDB 连接串 | `mongodb://admin:pass@localhost:27017` |
| `MINIO_ENDPOINT` | MinIO 地址 | `localhost:9000` |
| `BGE_M3_PATH` | BGE-M3 模型目录 | `D:\xxx\models\models\BAAI--bge-m3` |
| `BGE_RERANKER_LARGE` | Reranker 模型目录 | `D:\xxx\models\models\BAAI--bge-reranker-large` |

**可选项（有默认值）**：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_DEFAULT_MODEL` | `qwen-flash` | 主 LLM |
| `BGE_DEVICE` | `cuda:0` | Embedding 用 GPU |
| `MINERU_DEVICE_MODE` | `cpu` | MinerU 用 CPU 推理 |
| `MINERU_MODEL_DIR` | — | MinerU 版面模型目录 |

> **注意**：不要在 `.env` 里设全局 `CUDA_VISIBLE_DEVICES`。BGE-M3 需要 GPU。GPU 隔离只在 MinerU 子进程里做（`pdf_to_md_node.py` 会自动设）。

### 启动中间件

确保 Milvus、MongoDB、MinIO 已启动。三个服务可以放在一台机器上（CentOS 虚拟机或本地）。

#### 方案 A：Docker Compose 一键启动（推荐）

```bash
# 创建 docker-compose.yml 后
docker compose up -d
```

#### 方案 B：单独 Docker 命令

```bash
# MongoDB
docker run -d --name mongo --restart always -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=123456 \
  mongo:7

# MinIO
docker run -d --name minio --restart always -p 9000:9000 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data

# Milvus Standalone（依赖 etcd + MinIO）
docker run -d --name milvus-etcd -p 2379:2379 quay.io/coreos/etcd etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
docker run -d --name milvus-minio -p 9001:9000 minio/minio server /data
docker run -d --name milvus-standalone -p 19530:19530 \
  -e ETCD_ENDPOINTS=127.0.0.1:2379 -e MINIO_ADDRESS=127.0.0.1:9000 \
  milvusdb/milvus:v3.0.5 milvus run standalone
```

#### 验证中间件

```bash
# 所有容器 Up
docker ps --format "table {{.Names}}\t{{.Status}}"

# Milvus 健康检查
curl -s http://localhost:19530/healthz

# MongoDB ping
docker exec my-mongo mongosh --eval "db.runCommand({ping:1})"
```

### 启动服务

项目有两个独立的 FastAPI 服务，**必须同时运行**。

#### Windows PowerShell：

```powershell
# 终端 1：导入服务（端口 8000）
cd D:\PyCharm项目\shopkeeper_brain_1
.venv\Scripts\python.exe -m knowledge.api.import_router

# 终端 2：查询服务（端口 8001）
cd D:\PyCharm项目\shopkeeper_brain_1
.venv\Scripts\python.exe -m knowledge.api.query_router
```

> **重要**：必须在**项目根目录**（`shopkeeper_brain_1/`）下启动，不能在 `knowledge/` 子目录里跑。否则会报 `ModuleNotFoundError: No module named 'knowledge'`。

#### Linux/macOS：

```bash
# 终端 1
cd shopkeeper_brain_1
.venv/bin/python -m knowledge.api.import_router

# 终端 2
cd shopkeeper_brain_1
.venv/bin/python -m knowledge.api.query_router
```

#### 启动成功标志

两个终端都看到：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

#### 访问前端

| 页面 | 地址 |
|---|---|
| 导入页面 | `http://localhost:8000/front/import.html` |
| 聊天页面 | `http://localhost:8001/front/chat.html` |

上传 PDF → 等 1-2 分钟 → 导入成功后去聊天页提问。

---

## API 接口

### 导入服务（Port 8000）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/import/upload` | 上传文档（PDF/MD），返回 task_id |
| GET | `/import/task/{task_id}` | **SSE 流**，推送节点进度 + 最终结果 |

### 查询服务（Port 8001）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query/ask` | **SSE 流**，流式返回答案 |
| GET | `/history/{session_id}` | 获取对话历史 |
| DELETE | `/history/{session_id}` | 清空对话历史 |

### 请求示例

```bash
# 查询（流式）
curl -N -X POST http://localhost:8001/query/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "万用表怎么测电压？", "session_id": "test-session-001"}'

# 查对话历史
curl http://localhost:8001/history/test-session-001
```

---

## 配置说明

所有配置通过 `.env` 管理。分为三大类：

### 必须配置

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | LLM API Key |
| `OPENAI_API_BASE` | LLM API Base URL |
| `MILVUS_URL` | Milvus 地址 |
| `MONGO_URL` | MongoDB 连接串 |
| `MINIO_ENDPOINT` | MinIO 地址（host:port） |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | MinIO 凭据 |
| `MINIO_BUCKET_NAME` | MinIO 桶名 |
| `BGE_M3_PATH` | BGE-M3 模型路径 |
| `BGE_RERANKER_LARGE` | BGE-Reranker-Large 模型路径 |

### 可选配置（有默认值）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_DEFAULT_MODEL` | `qwen-flash` | 主 LLM |
| `VL_MODEL` | `qwen3-vl-flash` | 视觉模型（VLM） |
| `BGE_DEVICE` | `cuda:0` | BGE 推理设备 |
| `BGE_FP16` | `True` | FP16 加速 |
| `MINERU_DEVICE_MODE` | `cpu` | MinerU 推理设备 |
| `MINERU_MODEL_DIR` | — | MinerU 版面分析模型目录 |
| `MILVUS_MIN_COSINE_SCORE` | `0.75` | 向量相似度阈值 |
| `RRF_K` | `60` | RRF 融合参数 |
| `RERANK_MAX_TOP_K` | `10` | 重排序保留条数 |
| `ENABLE_AGENT_MODE` | `false` | 是否启用 ReAct Agent |
| `CHUNKS_COLLECTION` | `kb_chunks_v1` | Milvus chunks collection 名 |
| `ITEM_NAME_COLLECTION` | `kb_item_names_v1` | Milvus 商品名 collection 名 |
| `OPENBLAS_NUM_THREADS` | `1` | MinerU 子进程线程限制（防内存爆） |
| `MINERU_VIRTUAL_VRAM_SIZE` | `4` | MinerU 虚拟显存大小（GB） |

---

## 常见问题排查

### Q1：ModuleNotFoundError: No module named 'knowledge'

**原因**：在 `knowledge/` 子目录下启动服务，Python `-m` 找不到 `knowledge` 包。

**解决**：在**项目根目录**下启动：
```powershell
cd D:\PyCharm项目\shopkeeper_brain_1
.venv\Scripts\python.exe -m knowledge.api.import_router
```

---

### Q2：MinerU 上传 PDF 后 PyCharm / 整个 Python 进程崩溃

**原因**：MinerU 内部 multiprocessing 的 C 扩展（OpenBLAS、fasttext）在 Windows 中文路径下 abort，连带主进程组被杀。

**解决**：
1. 确保已应用 `patches/sitecustomize.py` 和 `patches/mineru_patch/` 补丁
2. 确保 `pdf_to_md_node.py` 使用 `subprocess.run()` 调用 MinerU 子进程
3. 子进程环境变量里已设 `CUDA_VISIBLE_DEVICES=''`

---

### Q3：子进程报 WinError 1455 页面文件太小

**完整报错**：`OSError: [WinError 1455] 页面文件太小，无法完成操作。Error loading "cublas64_12.dll"`

**原因**：主 FastAPI 进程已加载 cublas64_12.dll 到虚拟内存，MinerU 子进程 import torch 又加载一份 → Windows 虚拟内存爆。

**解决**：
1. 在 MinerU 子进程里强制 `CUDA_VISIBLE_DEVICES=''`（`pdf_to_md_node.py` 已设置）
2. 增加 Windows 虚拟内存：`sysdm.cpl` → 高级 → 虚拟内存 → 自定义大小（16384~32768 MB）→ 重启电脑

**不要**在 `.env` 全局禁 GPU，BGE-M3 需要 GPU。

---

### Q4：ValueError: Due to a serious vulnerability issue in torch.load...

**完整报错**：`transformers 5.17 强制 torch >= 2.6，但我们是 2.5.1`

**原因**：transformers 5.17.0 因 CVE-2025-32434 加了硬检查，torch < 2.6 一律不准 `torch.load`。BGE-M3 权重不是 safetensors 格式必须用 torch.load。

**解决**：应用 `patches/sitecustomize.py`。必须同时 patch **两处**（`import_utils` 和 `modeling_utils`），后者在模块 load 时直接 import 了函数存独立引用。

---

### Q5：FastText cannot be opened for loading

**完整报错**：`DetectError: Failed to load FastText model: lid.176.ftz cannot be opened for loading!`

**原因**：fasttext_pybind 的 C++ binding 无法处理包含中文的路径（如 `D:\PyCharm项目\...`）。

**解决**：应用 `patches/sitecustomize.py`，启动时自动 copy 模型到 `C:\Temp\ftlang\lid.176.bin` 并 override 路径。

---

### Q6：pip 报 Unable to create process using ...PyCharm??shopkeeper_brain_1

**原因**：venv 创建时 pip launcher 硬编码的路径在中文 `项目` 字符处编码错乱。

**解决**：绕过坏掉的 launcher，用：
```powershell
.venv\Scripts\python.exe -m pip install <包名>
```

不影响项目运行，只是装包时换个写法。

---

### Q7：ONNX Runtime 报 Require cuDNN 9.* and CUDA 13.*

**原因**：ONNX Runtime 1.30.0 和本机 CUDA 12.x 不兼容。

**解决**：应用 `patches/mineru_patch/README.md` 里的两个 MinerU 源码补丁，强制 CPUExecutionProvider。

---

### Q8：Milvus / MongoDB / MinIO 连不上

**排查步骤**：
```bash
# 1. 确认容器在跑
docker ps

# 2. 测试端口通不通
telnet 192.168.40.140 19530
telnet 192.168.40.140 27017
telnet 192.168.40.140 9000

# 3. 验证 Milvus 健康
curl http://192.168.40.140:19530/healthz
```

如果是远程虚拟机，还要检查 VMware/VirtualBox 网络模式（桥接/NAT）。

---

### Q9：导入成功但查询时 embedding 节点报错

**最常见**：之前 transformer CVE 检查没 patch 好（Q4），或者 GPU 没识别。

```powershell
# 快速检查 GPU 可见性
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，检查 `.env` 有没有设错误的 `CUDA_VISIBLE_DEVICES`。

---

### Q10：前端 SSE 进度不更新

**排查**：打开浏览器 F12 Network 面板，看 `/import/task/{id}` 请求是否在持续接收事件。如果断了，看后端终端有没有 traceback。

---

## 许可证

MIT License