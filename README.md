# 掌柜智库 (Shopkeeper Brain)

基于 LangGraph 的企业级知识库 RAG（检索增强生成）系统，面向零售/快消行业场景，支持商品资料文档的智能导入和自然语言查询。

## ✨ 核心特性

- **双通道架构**：文档导入流水线 + 查询回答流水线，由 LangGraph 状态机编排
- **混合检索**：Dense（向量）+ Sparse（关键词）+ HyDE 三路召回，RRF 融合排序
- **商品名智能识别**：专有的商品名实体识别节点，支持多候选确认
- **BGE Reranker 重排**：召回后二次精排，显著提升答案相关性
- **ReAct Agent 模式**：可选的多轮工具调用，支持 MCP DashScope 联网搜索
- **SSE 流式输出**：查询结果实时流式返回
- **全链路可观测**：任务状态追踪、耗时统计、历史记录持久化

## 🛠️ 技术栈

| 类别 | 技术 |
|---|---|
| 语言 | Python 3.13 |
| 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph + LangChain |
| 向量模型 | BAAI/bge-m3 (Embedding) + BAAI/bge-reranker-large (Rerank) |
| 大模型 | OpenAI 兼容 API（支持任何 OpenAI 协议的模型服务） |
| 向量数据库 | Milvus |
| 文档存储 | MinIO |
| 对话历史 | MongoDB |
| 文档解析 | MinerU |

## 📐 架构设计

### 导入流水线 (Import Graph)

```
文件上传 → PDF解析(MinerU) → 文档切片 → BGE Embedding → 商品名识别 → 写入Milvus
```

### 查询流水线 (Query Graph)

```
用户查询 → HyDE生成 → 混合向量检索 → 商品名确认 → Rerank精排 → RRF融合 → LLM生成答案
                ↓
         (可选) Agent模式 + MCP联网搜索
```

## 📁 目录结构

```
shopkeeper_brain/
├── knowledge/
│   ├── api/                          # FastAPI 路由层
│   │   ├── import_router.py          # 导入服务（端口 8000）
│   │   └── query_router.py           # 查询服务（端口 8001）
│   ├── core/                         # 核心基础设施
│   │   ├── deps.py                   # 依赖注入
│   │   └── paths.py                  # 路径管理
│   ├── processor/                    # LangGraph 流水线
│   │   ├── import_processor/         # 导入流水线
│   │   │   ├── main_graph.py         # 图定义
│   │   │   ├── state.py              # 状态结构
│   │   │   ├── config.py             # 导入配置
│   │   │   └── nodes/                # 各处理节点
│   │   └── query_processor/          # 查询流水线
│   │       ├── main_graph.py
│   │       ├── state.py
│   │       ├── config.py
│   │       ├── agent/                # ReAct Agent
│   │       ├── tools/                # Agent 工具
│   │       └── nodes/
│   ├── prompts/                      # Prompt 模板
│   ├── schema/                       # Pydantic 数据模型
│   ├── service/                      # 业务服务层
│   ├── utils/                        # 工具类
│   │   ├── client/                   # AI / 存储客户端
│   │   └── ...
│   ├── test/                         # 测试脚本（手动）
│   └── requirements.txt
├── models/                           # 本地模型文件（需自行下载）
├── .env.example                      # 环境变量模板
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- NVIDIA GPU（推荐，用于 Embedding / Reranker 推理）
- 以下中间件（按需部署）：
  - Milvus（向量数据库）
  - MongoDB（对话历史）
  - MinIO（文件存储）

### 2. 安装依赖

```bash
cd knowledge
pip install -r requirements.txt
```

### 3. 下载模型文件

需要两个 BGE 模型，放入 `models/` 目录：

```
models/
├── BAAI--bge-m3/          # Embedding 模型 (~2.2GB)
└── BAAI--bge-reranker-large/  # Reranker 模型 (~1.3GB)
```

**推荐使用 ModelScope 下载：**

```bash
# 安装 modelscope
pip install modelscope

# 下载 bge-m3
modelscope download --model BAAI/bge-m3 --local_dir models/BAAI--bge-m3

# 下载 bge-reranker-large
modelscope download --model BAAI/bge-reranker-large --local_dir models/BAAI--bge-reranker-large
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的实际配置：

```bash
cp .env.example .env
# 编辑 .env，填入 API Key、数据库连接串等
```

详细变量说明见 [.env.example](./.env.example)。

### 5. 启动中间件

确保 Milvus、MongoDB、MinIO 已启动。如果用 Docker Compose 快速启动：

```bash
# MongoDB
docker run -d --name mongo -p 27017:27017 mongo:7

# MinIO
docker run -d --name minio -p 9000:9000 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin minio/minio server /data

# Milvus Lite（开发调试，数据存本地文件，生产环境建议用 Milvus Standalone/Cluster）
pip install pymilvus[model]
```

### 6. 启动服务

项目分为两个独立的 FastAPI 服务：

```bash
# 终端 1：导入服务（端口 8000）
cd knowledge
python -m knowledge.api.import_router

# 终端 2：查询服务（端口 8001）
cd knowledge
python -m knowledge.api.query_router
```

## 📡 API 接口

### 导入服务 (Port 8000)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 上传文档（PDF），后台解析导入 |
| GET | `/status/{task_id}` | 查询导入任务状态（前端轮询） |

### 查询服务 (Port 8001)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 自然语言查询（支持流式/非流式） |
| GET | `/stream/{task_id}` | SSE 流式接收查询结果 |
| GET | `/history/{session_id}` | 获取对话历史 |
| DELETE | `/history/{session_id}` | 清空对话历史 |

### 查询请求示例

**非流式：**
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "万用表怎么使用？", "is_stream": false}'
```

**流式：**
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "万用表怎么使用？", "is_stream": true, "session_id": "optional-session-id"}'
# 返回 task_id 后
curl http://localhost:8001/stream/{task_id}
```

## ⚙️ 配置说明

所有配置通过 `.env` 环境变量管理，分为三大类：

### 必须配置

- **LLM**: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `LLM_DEFAULT_MODEL`
- **Milvus**: `MILVUS_URL`, `CHUNKS_COLLECTION`, `ITEM_NAME_COLLECTION`
- **MongoDB**: `MONGO_URL`, `MONGO_DB_NAME`
- **MinIO**: `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET_NAME`
- **模型路径**: `BGE_M3_PATH`, `BGE_RERANKER_LARGE`

### 可选配置（有默认值）

- 检索参数：`RRF_K`, `RERANK_MAX_TOP_K`, `EMBEDDING_SEARCH_LIMIT` 等
- 阈值调优：`MILVUS_MIN_COSINE_SCORE`, `ITEM_NAME_HIGH_CONFIDENCE` 等
- Agent 模式：`ENABLE_AGENT_MODE`（默认 false）

详见 `.env.example`。

## 📝 License

MIT