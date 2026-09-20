# 贡献指南

感谢参与 **Business-Assistant** 的开发。

本项目采用 **PR 流程** —— `main` 分支受保护，**不接受直接推送**。

---

## 一、开始之前

### 1.1 搭建本地环境

仓库**不包含**模型权重、训练产物和向量索引（体积超 GitHub 限制，已在 `.gitignore` 排除）。
首次拉取后，必须按 README「⚙️ 首次准备」补齐：

```powershell
git clone https://github.com/oom-alchemists/Business-Assistant.git
cd Business-Assistant

conda activate <你的环境>          # 或你惯用的虚拟环境
pip install -r requirements.txt

python scripts\download_models.py   # 下载全部模型：BERT + Embedding + 分类器权重（约 870MB）
python scripts\build_index.py       # 构建向量索引
```

另需本地安装 [Ollama](https://ollama.com)，并拉取生成模型：

```powershell
ollama pull qwen3.5:9b
```

> **跳过任何一步，启动时都会报「文件不存在」。**

### 1.2 先跑通，再改代码

```powershell
start_fastapi.bat          # 或 start_server.bat
```

浏览器打开 <http://localhost:8001>，确认能正常问答，然后再动手改。

---

## 二、分支策略

每次开发都从最新的 `main` 切出独立分支：

```powershell
git checkout main
git pull origin main          # 开新分支前务必先同步，否则容易产生冲突
git checkout -b feat/你的功能名
```

| 分支前缀 | 用途 | 示例 |
|---|---|---|
| `feat/` | 新功能 | `feat/chat-cache` |
| `fix/` | 修 Bug | `fix/intent-threshold-typo` |
| `docs/` | 仅文档 | `docs/update-readme-setup` |
| `refactor/` | 重构，不改外部行为 | `refactor/extract-prompt-builder` |
| `chore/` | 构建、依赖、配置 | `chore/bump-chromadb` |

---

## 三、Commit 规范

格式：

```
<type>: <简短描述>
```

| type | 含义 |
|---|---|
| `feat` | 新功能 |
| `fix` | 修 Bug |
| `docs` | 只改文档 |
| `refactor` | 重构，不改外部行为 |
| `perf` | 性能优化 |
| `chore` | 构建 / 依赖 / 配置 |

**正例**

```
feat: /api/chat 返回端到端耗时字段
fix: 最高置信度低于阈值时未走闲聊兜底
docs: 补充 Ollama 安装与拉取模型说明
```

**反例**（信息量为零，请避免）

```
update
fix bug
修改
```

另外：**一次提交只做一件事**。不要把「修 Bug」和「顺手重构」混在同一个提交里，
reviewer 无法分辨哪部分改了行为。

---

## 四、提交 PR

1. 推送分支：`git push -u origin feat/你的功能名`
2. 在 GitHub 发起 Pull Request，**目标分支选 `main`**
3. 描述里写清三件事：**改了什么 / 为什么改 / 怎么验证**
4. 至少 **1 人 review** 通过后才可合并
5. 合并后删除该功能分支

---

## 五、⚠️ 本项目的三个坑（动手前必读）

### 5.1 配置不在配置文件里

路径、模型名、阈值**全部硬编码在 `app/chat_engine.py` 顶部的常量区**，项目没有 `config.yaml`。

要调整行为，改那份常量：

| 常量 | 作用 |
|---|---|
| `MODEL_DIR` / `EMBED_DIR` | 模型目录 |
| `CKPT` | 分类器权重路径 |
| `CHROMA_DIR` | 向量索引目录 |
| `OLLAMA_URL` / `OLLAMA_MODEL` | 生成模型接口 |
| `OTHER_THRESHOLD` | 判为「其他/闲聊」的置信度阈值 |
| `CLASSIFY_DEVICE` | 分类与 Embedding 推理设备（默认 `cpu`） |

### 5.2 模型与索引不进仓库

`models/`、`checkpoints/`、`data/chroma/` 都在 `.gitignore` 里。
即使你本地重新训练或重建了索引，**也不要**尝试提交它们 —— 体积会撑爆仓库，
而且 git 历史一旦写入就清不掉。

### 5.3 启动脚本的写法

`.bat` 一律用 `cd /d "%~dp0"` 定位项目根，**禁止写本机绝对路径**（如 `D:\xxx\yyy`）。

同时 `.bat` 必须是 **CRLF** 换行（`.gitattributes` 已锁定）。用编辑器保存时注意别存成 LF，
否则 `cmd.exe` 解析 `goto` 标签和多行 `if ( )` 块时会失败。

---

## 六、代码风格

- Python 遵循 PEP 8；函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_CASE`
- 注释与文档字符串用**中文**（与现有代码保持一致）
- 新增可调参数请放到 `app/chat_engine.py` 顶部常量区，并加注释说明用途
- 不引入非必要的重型依赖；确有必要的，在 PR 描述里说明理由和影响

---

## 七、提交前自检

- [ ] 本地能正常启动，改动涉及的接口/界面试过一遍
- [ ] 没有提交模型文件、日志、`__pycache__`
- [ ] 没有硬编码本机绝对路径
- [ ] 没有把密钥、Token、密码写进代码
- [ ] Commit message 符合上面的规范

---

## 八、遇到问题

优先在 GitHub 开 Issue，写清 **现象 + 复现步骤 + 报错信息**。

环境类问题（跑不起来、依赖报错）也建议开 Issue —— 这样其他人遇到同样问题时能直接搜到答案，
而不是重复在群里问一遍。
