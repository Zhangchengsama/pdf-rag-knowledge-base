# 题一：PDF RAG 知识库

当前开发入口是 `v6/app.py`。检索流程为 PDF 文本提取、按字符切块、SentenceTransformer 编码、HNSW 与 BM25 检索、加权 RRF 融合、DeepSeek 回答。

## 版本管理约定

- 本目录是一个独立的本地 Git 仓库，主分支为 `main`。
- 首次提交保留现有 v1、v2、v3、v3 weight_rrf、v4、v5、v6 的源码和小型评测资料。它们是已有文件的导入快照，不代表重新还原了过去各版的开发历史。
- `baseline-v6` 标签指向本次导入基线。以后继续修改 `v6`，用提交记录每个完成的步骤，不再复制出 v7、v8 文件夹。
- 历史目录和现有聊天记录均保留在磁盘上。首次建立仓库时没有移动或删除文件，也没有修改 Python 业务代码。
- 未配置远程仓库。本地 Git 用于版本回溯；异地备份需要另行配置远程仓库或备份整个仓库。

## 日常操作

在本目录打开终端：

```powershell
git status
git diff
git add -- v6/app.py
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

本机尚未配置 Git 提交身份。首次基线使用一次性的 `Codex <codex@localhost>` 身份，不修改全局配置。后续手动提交前，请在本仓库设置自己的身份：

```powershell
git config user.name "你的名字"
git config user.email "你的邮箱"
```

## 哪些文件不进入版本管理

`.gitignore` 排除了 Python 缓存、虚拟环境、密钥文件、实际聊天记录、PDF、向量、BM25 缓存和 HNSW 索引。评测问题、小型评测结果和文档仍纳入管理。

当前 PDF 和 HNSW 索引路径在仓库外，Git 不会保存这些文件；PDF 同名的派生 JSON 也写在原 PDF 目录。如果以后将生成数据迁入仓库，应使用专门的缓存目录并加入忽略规则，不要统一忽略所有 JSON。

## v6 界面与兼容性

- 历史对话以首条非空用户消息作为标题，按文件修改时间倒序排列。同名标题补充短 ID，内部选择和保存始终使用原 UUID。已有 JSONL 不需要迁移或改名；修改时间会受手动编辑、复制文件影响。
- 侧边栏显示当前 PDF、读取上限、实际页数范围和缓存状态。打开页面只读检查，不触发建库。遗留 `building` 记录显示“上次建库未完成”。
- 文本块数复用当前页面会话中最近一次问答加载的数据；未提问或缓存已改变时显示“未统计”，不为统计额外读取文本 JSON。
- 历史消息和新回答共用来源渲染；没有来源时不显示空折叠区。
- 问答在建库检查完成后加载一次文本 JSON，Dense、BM25、提示词和返回来源共享这份数据。切块方式、HNSW 参数、加权 RRF 的权重及排名规则、提示词和回答字典格式保持原样。
- 原来的位置参数和路径调用继续可用。检索及来源映射函数新增末尾可选参数 `data=None`；`answer_question` 新增末尾可选回调 `on_build_status=None`，状态不会写入聊天记录。接口例子见 `v6/建库说明.md`。

## 运行与检查

本机已发现的现有环境是 `D:/miniforge3/envs/aip1`。在仓库根目录运行：

```powershell
& 'D:\miniforge3\envs\aip1\python.exe' -m streamlit run v6/app.py
```

页面依赖本地缓存的嵌入模型；生成回答需要环境变量 `DEEPSEEK_API_KEY`。PDF 路径、索引路径与读取页数仍以入口文件里的现有配置为准。前端目前读取前 30 页，命令行及评测入口目前读取前 11 页。

离线回归检查：

```powershell
& 'D:\miniforge3\envs\aip1\python.exe' -B -m unittest discover -s v6/test -p 'test_*.py'
```

现有测试使用真实的小型 PDF、BM25 和 HNSW，嵌入模型与 LLM 使用替身；不下载模型、不调用 API。`test_app.py` 使用 Streamlit 原生 AppTest 验证标题、排序、来源、状态及消息保存；它需要允许本机事件循环使用本地 socket。测试产物保留在临时目录，不执行批量删除。如果执行环境限制默认临时目录写入，可将进程的 `TEMP` 和 `TMP` 指向允许写入的目录后运行。

当前仓库尚未固定依赖版本，启动命令也不代表新电脑已经能够直接复现。后续建议补充依赖清单、配置示例与完整启动说明。
