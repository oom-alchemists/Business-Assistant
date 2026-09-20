# 项目目录结构

> 企业内部智能客服系统 · 完整文件说明

```
assistant/
│
├── 🚀 启动入口
│   ├── start_server.bat          # 一键启动 Streamlit（局域网 8501）
│   ├── start_fastapi.bat         # FastAPI 4 worker 并发接口（8001）
│   ├── start_public.bat          # FastAPI + cpolar 公网穿透
│   └── README.md                  # 项目总览
│
├── 📚 文档
│   ├── TECH_STACK.md              # 技术栈详解（架构/决策/性能）
│   ├── PROJECT_STRUCTURE.md       # 本文件（目录说明）
│   └── docs/
│       └── architecture.html      # 可视化架构图（浏览器打开）
│
├── 🧠 模型（自包含，无需联网下载）
│   ├── models/
│   │   ├── bert-base-chinese/     # BERT 预训练模型（392MB）
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   └── vocab.txt
│   │   └── bge-small-zh-v1.5/     # Embedding 模型（100MB）
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       └── tokenizer/
│   │
│   └── checkpoints/
│       └── intent_bert/
│           └── best.pt            # 训练好的 9 类分类器（~100MB）
│
├── 💾 数据
│   ├── data/
│   │   ├── class.txt               # 9 个类别名（8 业务类 + 1 其他闲聊类）
│   │   ├── train.txt               # 训练集 20.25 万条（文本\t标签ID）
│   │   ├── test.txt                # 测试集 1 万条
│   │   ├── dev.txt                 # 验证集 1 万条
│   │   ├── intent_train.csv        # 早期小数据集（备用）
│   │   ├── eval/
│   │   │   └── test_queries.jsonl  # 评估查询集
│   │   │
│   │   ├── knowledge_base/         # 📂 知识库（28 个 .md）
│   │   │   ├── policy/             # 考勤制度
│   │   │   │   ├── attendance.md   # 工作时间/请假
│   │   │   │   ├── overtime.md     # 加班调休
│   │   │   │   ├── abnormal.md     # 考勤异常处理
│   │   │   │   └── remote_work.md  # 远程/居家办公
│   │   │   ├── it/                 # IT 支持
│   │   │   │   ├── faq.md          # VPN/密码/打印机
│   │   │   │   ├── software.md     # 软件安装
│   │   │   │   ├── network.md      # 网络故障排查
│   │   │   │   └── email.md        # 企业邮箱
│   │   │   ├── finance/            # 财务报销
│   │   │   │   ├── expense.md      # 报销流程
│   │   │   │   ├── invoice.md      # 发票管理
│   │   │   │   ├── travel.md       # 差旅标准
│   │   │   │   └── advance.md      # 借支备用金
│   │   │   ├── hr/                 # 人力资源
│   │   │   │   ├── onboarding.md   # 入职离职
│   │   │   │   ├── social_security.md # 社保公积金
│   │   │   │   ├── training.md     # 培训晋升
│   │   │   │   └── referral.md     # 内推引才
│   │   │   ├── admin/              # 行政后勤
│   │   │   │   ├── logistics.md    # 食堂班车宿舍
│   │   │   │   ├── meeting.md      # 会议室车辆
│   │   │   │   └── activity.md     # 活动社团工会
│   │   │   ├── engineering/        # 工程设备
│   │   │   │   ├── equipment.md    # 设备维修
│   │   │   │   ├── safety.md       # 安全规程
│   │   │   │   └── calibration.md  # 计量校准
│   │   │   ├── legal/              # 法务合同
│   │   │   │   ├── contract.md     # 合同审批
│   │   │   │   ├── confidential.md # 保密知识产权
│   │   │   │   └── compliance.md   # 商业合规
│   │   │   └── sales/             # 销售市场
│   │   │       ├── sales.md        # 报价订单
│   │   │       ├── complaint.md    # 客户投诉
│   │   │       └── forecast.md     # 预测目标
│   │   │
│   │   └── chroma/                 # ChromaDB 向量索引（218 chunks）
│   │       └── chroma.sqlite3
│   │
│   └── logs/
│       └── assistant_2026-09-15.log  # 运行日志
│
├── ⚙️ 应用代码
│   ├── app/
│   │   ├── __init__.py
│   │   ├── webui.py                # 🎯 Streamlit 主界面（核心）
│   │   ├── api.py                  # 🎯 FastAPI 并发接口（uvicorn 多 worker）
│   │   ├── chat_engine.py          # 🎯 问答核心（两界面共用：分类/检索/生成）
│   │   ├── static/
│   │   │   └── index.html          # FastAPI 网页前端（微信风格，流式）
│   │   ├── bert_demo.py            # 早期 BERT 演示（备用）
│   │
│   └── src/                        # 模块化代码（备用/参考）
│       ├── router/                 # 意图路由
│       │   ├── bert_router.py      # BERT 分类路由
│       │   ├── llm_router.py       # LLM 路由
│       │   └── rule_router.py      # 规则路由
│       ├── retrieval/              # 检索
│       │   ├── vector_store.py     # 向量库
│       │   └── reranker.py         # 重排序
│       ├── generator/              # 生成
│       │   ├── llm_client.py       # LLM 客户端
│       │   └── prompt_templates.py # Prompt 模板
│       ├── pipeline/
│       │   └── assistant.py        # 完整 pipeline
│       └── utils/
│           ├── logger.py           # 日志
│           └── metrics.py          # 指标
│
├── 🔧 脚本
│   ├── scripts/
│   │   ├── gen_large_dataset.py     # 🎯 生成 18 万训练数据
│   │   ├── gen_train_data.py       # 生成小数据集（旧）
│   │   ├── train_intent.py        # 🎯 BERT 训练脚本
│   │   ├── build_index.py         # 🎯 建 ChromaDB 向量索引
│   │   ├── build_kb.py            # 知识库构建（备用）
│   │   └── run_eval.py             # 评估脚本
│   │
│   └── config/
│       ├── config.yaml             # 全局配置
│       └── intents.yaml            # 意图定义
│
└── 🧪 测试
    └── tests/
        └── test_pipeline.py        # pipeline 单元测试
```

## 文件作用速查

| 文件 | 作用 | 是否核心 |
|---|---|---|
| `app/webui.py` | Streamlit 对话界面 | ⭐ 核心 |
| `app/api.py` | FastAPI 并发接口（8001） | ⭐ 核心 |
| `app/chat_engine.py` | 问答核心（BERT/Chroma/LLM 封装） | ⭐ 核心 |
| `app/static/index.html` | FastAPI 网页前端 | ⭐ 核心 |
| `scripts/build_index.py` | 知识库改动后重建向量索引 | ⭐ 核心 |
| `scripts/train_intent.py` | BERT 分类器训练 | ⭐ 核心 |
| `scripts/gen_large_dataset.py` | 生成训练数据 | ⭐ 核心 |
| `checkpoints/intent_bert/best.pt` | 训练好的分类器权重 | ⭐ 核心 |
| `data/chroma/` | 向量索引，RAG 检索用 | ⭐ 核心 |
| `models/bert-base-chinese/` | BERT 预训练模型 | 依赖 |
| `models/bge-small-zh-v1.5/` | Embedding 模型 | 依赖 |
| `data/knowledge_base/` | 28 个知识库文档 | 内容 |
| `start_server.bat` | 启动 Streamlit | 启动 |
| `start_fastapi.bat` | 启动 FastAPI 并发服务 | 启动 |
| `start_public.bat` | 启动公网穿透 | 启动 |
| `TECH_STACK.md` | 技术栈详解 | 文档 |
| `PROJECT_STRUCTURE.md` | 本文件 | 文档 |

## 常用操作

```powershell
# 启动 Streamlit（局域网）
start_server.bat

# 启动 FastAPI 并发服务（4 worker）
start_fastapi.bat

# 启动公网穿透（FastAPI + cpolar）
start_public.bat

# 重建向量索引（改了知识库后）
python scripts\build_index.py

# 重新训练 BERT
python scripts\train_intent.py --epochs 3 --bs 64

# 重新生成训练数据
python scripts\gen_large_dataset.py
```
