# 题一：PDF RAG 知识库

统一启动入口是根目录 `run.py`，当前源码位于 `src/`。网页导航放在 `src/ui.py`，问答和知识库管理分别放在 `src/chat_ui.py`、`src/knowledge_ui.py`，终端问答放在 `src/cli.py`。检索流程为 PDF 文本提取、按字符切块、SentenceTransformer 编码、HNSW 与 BM25 检索、加权 RRF 融合、DeepSeek 回答。

## 版本管理约定

当前目录结构：`src/` 保存源码，`tests/` 保存测试与评测资料，`docs/` 保存说明，`scripts/` 保存维护脚本，`legacy/` 原样保留 v1～v5（含 v3 weight_rrf），`data/` 保存运行数据。

2026-10-29 已执行两步目录整理。`data/settings/layout_manifest.json` 是整理前的文件校验清单，`migration_plan.json` 与 `migration_report.json` 记录运行数据的迁移及 SHA256 校验，`path_map.json` 让旧请求继续找到迁移后的资料并保留原读取范围。旧设置另存为 `legacy_knowledge_base.json`。迁移脚本 `scripts/migrate_layout.py` 不覆盖不同内容的目标文件，已完成后重复执行不会改写数据；外部原缓存留在原位置。迁移后留下的空目录和 Python 缓存没有删除。

- 本目录是一个独立的本地 Git 仓库，主分支为 `main`。
- 首次提交保留现有 v1、v2、v3、v3 weight_rrf、v4、v5、v6 的源码和小型评测资料。它们是已有文件的导入快照，不代表重新还原了过去各版的开发历史。
- `baseline-v6` 标签指向本次导入基线。以后继续修改 `src`，用提交记录每个完成的步骤，不再复制出 v7、v8 文件夹。
- 历史目录和现有聊天记录均保留在磁盘上。首次建立仓库时没有移动或删除文件。2026-10-29 的目录整理将历史源码移到 `legacy/`，当前源码移到 `src/`，算法保持不变。
- 未配置远程仓库。本地 Git 用于版本回溯；异地备份需要另行配置远程仓库或备份整个仓库。

## 日常操作

在本目录打开终端：

```powershell
git status
git diff
git add -- src/ui.py
git diff --cached
git commit -m "改进历史对话展示"
git log --oneline --decorate
```

一次修改涉及多个文件时，在 `git add --` 后明确列出文件。提交前检查暂存差异，确认没有聊天记录、密钥和生成数据。

查看初始版本中的文件，不覆盖工作区：

```powershell
git show baseline-v6:v6/app.py
```

较大的独立功能可以先创建分支：

```powershell
git switch -c feature/conversation-titles
```

建立基线时本机尚未配置 Git 提交身份。首次基线使用一次性的 `Codex <codex@localhost>` 身份，不修改全局配置。后续手动提交前，请在本仓库设置自己的身份：

```powershell
git config user.name "你的名字"
git config user.email "你的邮箱"
```

## 哪些文件不进入版本管理

`.gitignore` 排除了 Python 缓存、虚拟环境、密钥文件、实际聊天记录、PDF、向量、BM25 缓存和 HNSW 索引。评测问题、小型评测结果和文档仍纳入管理。

默认 PDF 仍使用原来的仓库外路径，索引已迁入项目数据目录。上传原件保存在 `data/uploads/<内容摘要>/原文件名.pdf`，网页、终端和评测使用 `data/cache/<PDF路径摘要>/` 保存文本 JSON、向量、BM25、HNSW 和双建库记录。聊天位于 `data/conversations/`，当前知识库设置位于 `data/settings/knowledge_base.json`。`data/` 中除目录说明外的运行数据均不进入 Git，测试题集 JSON 继续纳入管理。外部教材原件仍使用原路径。旧业务函数未传 `output_path` 时，继续使用 PDF 同目录的缓存。

## 当前源码文件分工

文件保持平铺。入口只安排调用顺序，各功能写成函数；函数开头用“功能短词＋编号步骤”说明实现。

| 文件 | 负责什么 |
| --- | --- |
| `main.py` | 解析启动参数，分派网页或终端模式 |
| `ui.py` | 顶部导航与页面分派 |
| `chat_ui.py` | 问答页面、会话选择、消息和处理状态展示 |
| `knowledge_ui.py` | 独立知识库管理页、文档表单和知识库状态 |
| `cli.py` | 终端输入问题、输出回答 |
| `config.py` | 默认参数、目录及只读 `Settings` 配置 |
| `get_pdf.py` | 逐页解析 PDF 和来源信息 |
| `chunker.py` | 按字符切块，固定 `split_text` 接口 |
| `embedding.py` | 按需加载嵌入模型、逐块编码 |
| `dense.py` | HNSW 创建和稠密召回 |
| `sparse.py` | 分词、BM25 创建和关键词召回 |
| `fusion.py` | 纯加权 RRF，固定 `fuse_rankings` 接口 |
| `query.py` | 编排两路召回并调用融合函数 |
| `prompt.py` | 问答提示词格式化 |
| `cache_store.py` | 文本、向量、BM25、索引和建库记录的读写 |
| `knowledge_base.py` | 建库、缓存复用和只读状态判断 |
| `rag_service.py` | 单次问答流程和旧业务入口转发 |
| `conversations.py` | 消息与会话信息存取、列表及旧记录兼容 |
| `chat_workflow.py` | 可恢复请求、回答重试和标题流程 |
| `document_settings.py` | 文档校验、上传保存、独立索引路径与共用知识库设置 |
| `conversation_lock.py` | 会话和共享 PDF 的互斥锁 |
| `progress.py` | 阶段状态通知与耗时记录 |
| `LLM.py` | 按需创建 API 客户端，生成回答和标题 |
| `eval.py` | 检索评测及结果报告 |

修改切块算法时编辑 `chunker.split_text`，输入文本和切块参数，返回有序字符串列表。修改融合算法时编辑 `fusion.fuse_rankings`，输入两路有序下标，返回有序下标和分数。JSON 第 i 块、向量第 i 行、BM25 第 i 条和 HNSW 标签 i 必须保持对应。

更换切块方法或嵌入模型后，继续显式设置 `force_rebuild=True` 重建，不增加算法版本指纹或自动迁移。只改融合公式时可直接重新评测。`get_pdf.split_text`、`get_pdf.save_to_json`、`query.RRF_top_k_query_indece` 和原来的建库、检索、提示词调用仍可使用。

## 界面与兼容性

- 完整首个问题交给独立的大模型提示词生成短标题，不直接截取或复述原问题。新会话首次回答成功后额外请求一次标题并持久保存；刷新和后续提问不重复生成。标题失败显示“待生成标题”，可单独重试，不重新生成回答。旧会话浏览时不自动调用 API，可点击“生成对话标题”。
- 会话按聊天文件修改时间倒序排列，同名标题补充短 ID，内部仍用原 UUID。已有 JSONL 不需要迁移或改名；标题与请求状态放在旁边的 `.meta.json` 中。修改时间会受手动编辑、复制文件影响。
- 顶部导航切换“问答”和“知识库管理”，管理功能不再放在侧边栏。问答页的侧边栏保留新建及历史会话。
- 在独立管理页中输入本机路径或上传 PDF，选择前 N 页或全文，点击“应用文档和范围”。上传文件优先于路径输入。设置持久保存，所有会话的新提问使用当前设置；已有对话可继续使用，无需绑定文档或新建对话。首次提问自动建库，每份 PDF 分配独立索引，默认教材继续使用原索引。
- 管理页显示当前 PDF、读取上限、实际页数范围和缓存状态。打开页面和应用设置只读检查文档，不触发建库。遗留 `building` 记录显示“上次建库未完成”。
- 每次提交时将使用的文档和范围保存在请求内，失败重试沿用原请求资料，避免切换知识库后混用。旧会话的绑定字段保留在磁盘上，新提问使用管理页设置；旧未完成请求仍可按原资料重试。旧绑定函数仅供旧业务调用兼容，网页不再调用。
- 模型只在处理问题时加载。文件、模型、密钥、网络、限流等失败会显示提示；“重试这个问题”复用原请求，不重复写用户消息。已落盘的回答也不会因完成标记保存中断而再次生成。未完成问题可重试或新建对话。
- 处理面板显示模型加载、知识库准备、检索、回答和标题阶段及耗时。每次重试记录本次耗时和累计尝试次数；中断且无法确定的阶段明确显示“耗时未完整记录”。独立标题重试另行显示最近一次标题耗时。
- 会话及共享 PDF 使用操作系统文件锁，阻止另一标签页把活动请求当成中断或重复执行。同一 PDF 的不同会话不能并行改写派生缓存，遇忙可稍后重试；不同文档互不影响。锁文件保留，不执行目录清理或批量删除。
- 文本块数复用当前页面会话中最近一次问答加载的数据；未提问或缓存已改变时显示“未统计”，不为统计额外读取文本 JSON。
- 历史消息和新回答共用来源渲染；没有来源时不显示空折叠区。
- 问答在建库检查完成后加载一次文本 JSON，Dense、BM25、提示词和返回来源共享这份数据。切块方式、HNSW 参数、加权 RRF 的权重及排名规则、提示词和回答字典格式保持原样。
- 原来的位置参数和路径调用继续可用。检索及来源映射函数支持末尾可选 `data=None`；`answer_question` 支持末尾可选回调 `on_build_status=None`、`on_progress=None`，状态不写入 JSONL 消息。接口例子见 `docs/建库说明.md`。

## 运行与检查

本机使用现有环境 `D:/miniforge3/envs/aip1`，没有新增安装。文件锁复用已安装的 `filelock 3.32.6`；其他环境需要 `filelock>=3.32.6`，以支持保留锁文件与禁用软锁回退。在仓库根目录运行：

```powershell
& 'D:\miniforge3\envs\aip1\python.exe' run.py
```

终端问答和独立检索评测：

```powershell
& 'D:\miniforge3\envs\aip1\python.exe' run.py --cli
& 'D:\miniforge3\envs\aip1\python.exe' src/eval.py
```

根目录 `run.py` 转发到 `src/main.py`。默认启动器通过当前 Python 解释器调用 Streamlit，再使用内部 `--web` 标记运行同一个入口。普通导入模块不会启动服务、读取终端输入、加载模型或创建 API 客户端。

提问依赖本地缓存的嵌入模型；回答和标题生成使用环境变量 `DEEPSEEK_API_KEY`。API 设置 60 秒超时参数并关闭 SDK 自动重试，由页面提供明确的手动重试。网页、命令行、评测和省略页数的业务调用统一默认前 30 页；显式传入其他正整数或 `None` 继续有效。配置集中在 `config.py`，测试或其他调用方可通过 `load_settings(...)` 注入目录和设置，不改页面源码。

离线回归检查：

```powershell
& 'D:\miniforge3\envs\aip1\python.exe' -B -m unittest discover -s tests -p 'test_*.py'
```

现有测试使用真实的小型 PDF、BM25 和 HNSW，嵌入模型与 LLM 使用替身；不下载模型、不调用 API。`test_app.py` 使用 Streamlit 原生 AppTest 验证标题、排序、来源、状态及消息保存；它需要允许本机事件循环使用本地 socket。测试产物保留在临时目录，不执行批量删除。如果执行环境限制默认临时目录写入，可将进程的 `TEMP` 和 `TMP` 指向允许写入的目录后运行。

当前仓库尚未固定依赖版本，启动命令也不代表新电脑已经能够直接复现。后续建议补充依赖清单、配置示例与完整启动说明。
