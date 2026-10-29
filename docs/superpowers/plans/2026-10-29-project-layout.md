# 项目目录整理 Implementation Plan

> **For agentic workers:** 使用 executing-plans，在当前已存在的 refactor/v6-structure 分支单智能体执行；用户已授权两步整理及验证。

**Goal:** 整理项目目录和数据位置，完整保留现有资料与业务行为。
**Architecture:** src 内沿用平铺模块；run.py 只分派到原入口。集中缓存通过可选 output_path 接入，旧调用仍有效。迁移单独执行并保存校验记录。
**Tech Stack:** Python、Streamlit、unittest、PyMuPDF、现有 BM25/HNSW。
**Spec:** docs/superpowers/specs/2026-10-29-project-layout-design.md

## Global Constraints

- 不删除文件或目录；移动前核对绝对路径位于项目中，目标存在时停止。
- 保留旧版本、上传和聊天，算法及回答格式不变。
- 使用 D:/miniforge3/envs/aip1/python.exe，不新增依赖。
- 实际复制前核对大小；单文件超过 1 GB 或文件群超过 3 GB 先告知。

## Task 1: 源码与入口

- [x] 基线：`python -B -m unittest discover -s v6/test`；记录文件哈希和体积。
- [x] 在原结构测试中增加根目录 run.py 启动和 src 根路径断言，运行并确认新断言失败。
- [x] 逐个移动源码到 src、测试到 tests、文档到 docs、历史目录到 legacy。run.py 将 src 加入导入路径，再调用 main.main；config 默认 base_dir 改为项目根目录，evaluation_path 指向 tests。
- [x] 更新测试源码定位，保持所有业务断言。运行原有离线测试及异地工作目录 `run.py --help`。

## Task 2: 运行数据

- [x] 添加真实小型 PDF 的集中缓存测试：output_path 指向 data/cache 下的 index.json，校验建库/复用/检索与旧路径调用一致；先观察失败。
- [x] cache_store.cache_paths/metadata_paths/save_to_json 增加末尾可选 output_path；knowledge_base 状态/建库和 rag_service 转发该路径；BM25 从文本缓存位置读取。无参数调用保持旧格式。
- [x] document_settings 持久化可选 output_path；新选择资料使用 cache/<路径摘要>/index.bin 与 index.json；上传位置、聊天及设置由 config 定位。
- [x] 新增 scripts/migrate_layout.py，迁移前校验冲突与文件清单，逐文件移动旧 data，生成旧资料到新资料的路径映射；复制可复用外部缓存并翻译路径记录，保留原件。验证第二次执行不改动已有文件。
- [x] 对照保存的 SHA256 清单验证旧版本、上传、聊天内容完整；更新 README 和忽略规则。全量测试，实际启动网页并验证导航，检查 Git 差异。
