# 企业内部智能客服助手

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **BERT 意图分类 × 轻量 RAG × Ollama LLM 生成**
> 本地部署 · 8 大业务领域 + 其他闲聊 · 18 万训练数据 · 局域网 / 公网可访问

## ⚙️ 首次准备（新成员必读）

仓库**不包含**模型权重、训练产物和向量索引（体积超 GitHub 限制，已在 `.gitignore` 中排除）。首次拉取代码后按顺序执行：

```powershell
# 0. 先激活你的 Python 环境（conda / venv 均可）
conda activate <你的环境名>

# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 下载全部模型文件（BERT + Embedding + 训练好的分类器，合计约 870MB）
python scripts\download_models.py

# 3. 构建向量索引 -> 生成 data/chroma/
python scripts\build_index.py
```

另需本地安装 [Ollama](https://ollama.com) 并拉取生成模型：

```powershell
ollama pull qwen3.5:9b
```

完成以上三步后，才能用下面的一键启动。**跳过任一步都会在启动时报「文件不存在」。**

> **关于分类器权重 `best.pt`（390MB）**：它由维护者发布在本仓库的
> [Releases](https://github.com/oom-alchemists/Business-Assistant/releases) 页面，
> `download_models.py` 会自动拉取（支持断点续传，中断后重跑即可）。
> 想自己训练（需 GPU，约 20 分钟）：`python scripts\train_intent.py --epochs 3 --bs 64`
>
> 💡 若从其他成员处直接拿到了模型文件，可手动放进 `models/` 与 `checkpoints/`，省去第 2 步。

### 🌐 网络问题（国内常见，必读）

`git push/pull` 或 `download_models.py` 都可能报 `Connection was reset`。
这**不是配置错误**，而是直连 GitHub 被切断——**浏览器能打开 GitHub 不代表 git 能**，
因为 git 不读取 Windows 的系统代理设置。

如果你本机有代理（Clash 默认端口是 `7890`），配置后即可：

```powershell
# 只对 GitHub 生效，不影响公司内网等其他仓库
git config --global "http.https://github.com.proxy" "http://127.0.0.1:7890"
```

下载脚本会自动读取 `HTTPS_PROXY` 环境变量：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
python scripts\download_models.py
```

不再需要时撤销：

```powershell
git config --global --unset "http.https://github.com.proxy"
```

> HuggingFace 下载慢的话，可设置国内镜像：`set HF_ENDPOINT=https://hf-mirror.com`

## 🚀 一键启动

双击项目根目录的 **`start_server.bat`** 即可自动启动全部服务（会自动检查 Python 与 Ollama）。

### 三种访问方式

| 方式 | 脚本 | 端口 | 说明 |
|------|------|------|------|
| 一键全起 | `start_server.bat` | 8501+8001 | 同时启动 Streamlit 和 FastAPI 2 worker |
| 本机 | `start_server.bat` | 8501 | Streamlit 界面，http://localhost:8501 |
| 局域网并发 | `start_server.bat` | 8001 | FastAPI 并发界面，http://<本机IP>:8001 |
| **公网** | `start_public.bat` | 8001 | FastAPI + cpolar 内网穿透，生成公网链接 |
| **并发 API** | `start_fastapi.bat` | 8001 | FastAPI 2 worker 并发，支持多用户同时访问 |

> `start_server.bat` 现在会同时启动两个服务：8501（Streamlit 单用户界面）和 8001（FastAPI 2 worker 并发界面）。**局域网多人同时使用请访问 8001 端口**，并发能力实测 5 个同时提问全部成功互不阻塞。

> 公网方式：先安装 [cpolar](https://www.cpolar.com) 并 `cpolar authtoken <你的token>`，然后双击 `start_public.bat`，复制输出的 `https://xxxxxx.cpolar.top` 链接即可分享给任何人打开聊天页面（FastAPI 版微信风格界面，支持流式回答）。

### 两种界面

| 界面 | 端口 | 特点 |
|------|------|------|
| Streamlit（原有） | 8501 | 双主题、侧边栏分类、流式气泡 |
| FastAPI 网页（新增） | 8001 | 微信风格、流式回答、Light/Dark 切换、多用户并发 |

## 📂 支持的 9 大意图分类（8 业务 + 其他闲聊）

| 分类标识 | 中文 | 覆盖问题 |
|---------|------|---------|
| `policy_attendance` | 考勤制度 | 上下班、打卡、请假、年假、加班、远程办公 |
| `it_vpn` | IT 支持 | VPN、密码、电脑、打印机、邮箱 |
| `finance_expense` | 财务报销 | 报销、发票、工资、差旅、借支 |
| `hr_onboarding` | 人力资源 | 入职、离职、社保、福利、内推 |
| `admin_logistics` | 行政后勤 | 食堂、班车、宿舍、会议室、活动 |
| `engineering_eq` | 工程设备 | 产线、维修、点检、备件、计量校准 |
| `legal_contract` | 法务合同 | 合同、公章、保密、知识产权、合规 |
| `sales_marketing` | 销售市场 | 报价、订单、客户、展会、预测 |
| `other_chitchat` | 其他/闲聊 | 问候、闲聊、与业务无关的问题 |

## 🏗️ 系统架构

```
用户提问
  │
  ▼
① BERT 意图分类（bert-base-chinese 微调，<10ms）
  │
  ▼
② bge-small-zh 向量化用户问题
  │
  ▼
③ ChromaDB 在对应类别下检索 Top-3 片段
  │
  ▼
④ Ollama qwen3.5:9b 基于片段生成回答（think=false）
  │
  ▼
回答 + 分类标签 + Top3 置信度
```

## 📁 目录结构

```
assistant/
├── start_server.bat          # 一键启动全部（Streamlit 8501 + FastAPI 8001）
├── start_fastapi.bat         # 只启动 FastAPI 8001
├── start_public.bat          # FastAPI + cpolar 公网穿透
├── README.md                 # 本文件
├── CONTRIBUTING.md           # 协作规范（分支 / 提交 / PR 流程）
├── TECH_STACK.md             # 技术栈详解与设计决策
├── LICENSE                   # MIT 开源协议
├── requirements.txt          # Python 依赖
├── .gitignore                # 忽略规则（模型/索引/日志）
├── .gitattributes            # 换行符规范（.bat 强制 CRLF）
│
├── app/                      # 应用代码（全部实现）
│   ├── webui.py              # Streamlit 界面（8501）
│   ├── api.py                # FastAPI 并发接口（8001）
│   ├── chat_engine.py        # ⭐ 问答核心（两界面共用）+ 全部可调常量
│   └── static/index.html     # FastAPI 网页前端
│
├── scripts/
│   ├── download_models.py    # 下载预训练模型
│   ├── gen_large_dataset.py  # 生成训练数据
│   ├── train_intent.py       # 训练 BERT 分类器
│   ├── build_index.py        # 构建向量索引（→ data/chroma/）
│   └── start_all.py          # 一键启动的 Python 实现
│
├── data/
│   ├── class.txt             # 9 个类别名
│   ├── train.txt / test.txt / dev.txt  # 18万/1万/1万（已入库）
│   ├── intent_train.csv      # 早期小数据集（备用）
│   ├── eval/test_queries.jsonl
│   ├── knowledge_base/       # 8 部门 28 个 .md
│   └── chroma/               # 向量索引（生成物，不入库）
│
├── models/                   # 预训练模型（生成物，不入库）
│   ├── bert-base-chinese/
│   └── bge-small-zh-v1.5/
│
└── checkpoints/intent_bert/best.pt  # 训练产物（不入库）
```

### ⚙️ 可调参数在哪

路径、模型名、阈值**全部集中在 `app/chat_engine.py` 顶部常量**（项目没有外部配置文件）：

| 常量 | 作用 | 默认值 |
|---|---|---|
| `MODEL_DIR` / `EMBED_DIR` | BERT 与 Embedding 模型目录 | `models/...` |
| `CKPT` | 分类器权重路径 | `checkpoints/intent_bert/best.pt` |
| `CHROMA_DIR` | 向量索引目录 | `data/chroma` |
| `OLLAMA_URL` / `OLLAMA_MODEL` | 生成模型接口与模型名 | `127.0.0.1:11434` / `qwen3.5:9b` |
| `OTHER_THRESHOLD` | 最高置信度低于此值判为「其他/闲聊」 | `0.55` |
| `CLASSIFY_DEVICE` | 分类与 Embedding 推理设备 | `"cpu"` |
| `retrieve(..., top_k=3)` | 检索片段数 | `3` |

## 🔧 训练流程

```powershell
# 生成数据
python scripts\gen_large_dataset.py

# 训练 BERT
python scripts\train_intent.py --epochs 3 --bs 64

# 重建向量索引（改知识库后）
python scripts\build_index.py
```

## 🔌 FastAPI 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 微信风格聊天页面 |
| `/api/chat` | POST | 非流式问答 `{"query": "..."}` → JSON（含 intent/tags/sources） |
| `/api/chat/stream` | POST | 流式问答，NDJSON 逐段推送 |
| `/api/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 在线文档 |

```powershell
# 示例：调用问答接口
python -c "import requests,json; print(requests.post('http://localhost:8001/api/chat',json={'query':'年假有几天'}).json()['answer'])"
```

## 📊 技术指标

| 指标 | 数值 |
|------|------|
| 训练数据 | 180,000 条 |
| 类别数 | 9（8 业务 + 其他闲聊） |
| 测试集准确率 | 98.2%（**合成测试集**，由模板生成，非真实业务数据） |
| 知识库 | 28 文档 / 218 chunks |
| BERT 推理 | <10ms |
| LLM 延迟 | 2-3s |
| 并发能力 | uvicorn 多 worker（当前 `--workers 2`），多用户并发提问互不阻塞 |

## 📄 开源协议

本项目采用 [MIT License](LICENSE)，版权归 **oom-alchemists** 所有。

## 🤝 参与贡献

动手前请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。要点：

- `main` 分支受保护，**采用 PR 流程**，不接受直接推送
- 功能开发请从最新 `main` 切出 `feat/xxx` 分支，PR 需至少 1 人 review
- Commit message 格式：`<type>: <简短描述>`（`feat` / `fix` / `docs` / `refactor` / `perf` / `chore`）
- 模型权重与向量索引不进仓库，请勿尝试提交
