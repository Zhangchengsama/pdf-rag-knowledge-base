---
title: "AI 工程能力习题集：从真实需求到可交付系统"
author: "学习版 v0.1"
date: "2026-09-12"
---

# 前言：这不是“先学完再做”的教材

这份习题集的目标不是把人工智能、后端、部署、计算机视觉、LLM、Agent、3D 点云等知识从头到尾讲一遍，而是用五个真实外包需求作为五道“母题”，建立一条 **需求驱动（top-down）、边做边学、以交付为验收** 的工程能力训练路线。

核心方法只有一句话：

> **先把客户要的最终系统画出来，再沿着数据流向下拆；做到哪一步卡住，就只补这一层所需的知识，补完立刻回到工程。**

这与“先系统学 FastAPI、再学 Docker、再学 RAG、再学 PyTorch……”的自下而上路径不同。后者容易形成“知识都听过，但没有一个系统真正交付过”的状态；本习题集追求的是另一种能力：给你一个不完全熟悉的需求，你能拆成架构、里程碑、验收指标、实现模块，并在遇到问题时知道下一步该查什么、做什么实验。

本书中的五个例题来自 2026 年 8-9 月公开的 Upwork 真实需求。为了避免把客户原文与本书扩展要求混在一起，每题均分为：

1. **真实需求摘要**：只对公开招聘信息进行事实性转述；
2. **训练版任务书**：为了学习而补充的工程要求；
3. **验收 Gate**：每一步做到什么才允许继续；
4. **卡点导航**：卡住时具体应该补什么知识、查什么关键词；
5. **“交差”与“优秀”的分界**：知道什么时候已经具备收费交付能力。

> 重要说明：公开岗位中的预算只能说明该客户当时的报价，不能等同于技术难度和合理市场价格。部分岗位的预算显然只是占位或偏低，本书只把它们当作能力标尺。

---

# 五道题共同考的，其实是一套工程骨架

五个项目表面差别很大：一个是 RAG，一个是 LLM 微调，一个是边缘端 CV，一个是 Agent，一个是 LiDAR 点云。但真正的“可收费工程能力”可以统一成下面这条链：

$$
\boxed{
\text{Problem}
\rightarrow \text{Data}
\rightarrow \text{Baseline}
\rightarrow \text{Algorithm}
\rightarrow \text{Evaluation}
\rightarrow \text{API/System}
\rightarrow \text{Deployment}
\rightarrow \text{Maintenance}
}
$$

## 每个项目都必须回答的八个问题

1. **输入是什么？** 文件、图片、传感器数据、Webhook、自然语言？
2. **输出是什么？** 分类标签、答案、结构化 JSON、三维坐标、自动动作？
3. **“正确”如何定义？** Accuracy、F1、Recall@k、引用正确率、延迟、资源占用，还是业务规则？
4. **最简单 baseline 是什么？** 不允许一开始就上最复杂模型。
5. **失败会发生在哪一层？** 数据、预处理、检索、模型、接口、网络、部署、权限？
6. **如何复现？** 固定依赖、配置、随机种子、模型版本、数据版本。
7. **如何被别人调用？** CLI、REST API、Web UI、消息队列、ROS topic？
8. **如何证明已经完成？** 自动测试、评测集、benchmark、Demo、README、部署文档。

## 五题共用的“最低工程底座”

不要先花一个月专门学底座。第一次用到时学会即可，但五个项目最终都应具备以下结构：

```text
project/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example
├── configs/
├── src/
│   ├── core/
│   ├── api/
│   └── utils/
├── scripts/
├── tests/
├── eval/
├── data/          # 只放可公开/可提交的小样例
├── artifacts/     # 指标、图表、模型卡等
├── Dockerfile
└── docker-compose.yml   # 需要多服务时再加
```

**最低通用要求：**

- Python 项目可以从全新环境安装；
- 配置与代码分离；
- secret 不进 Git；
- 日志能看出系统在哪一层失败；
- 核心路径至少有自动测试；
- 有一个一键运行入口；
- 有 README：环境、启动、输入、输出、测试、已知限制；
- 有一个客观评测脚本，而不是“我试了几次感觉挺好”。

## 学习纪律：先纵向打通，再横向优化

每题都遵循同样的开发次序：

**第一步：Thin Vertical Slice（最薄垂直切片）**

让一条最简单的数据真正从输入走到输出。例如 RAG 不要先研究 15 种 chunking，先让 3 个 PDF 能被索引并回答；Agent 不要先接 CRM，先让一个本地 JSON 输入能得到结构化评分。

**第二步：建立评测。**

没有评测就不要优化。优化必须能回答“比 baseline 好了多少”。

**第三步：逐个替换最弱模块。**

一次只改变一个主要变量，否则你无法知道提升来自哪里。

**第四步：工程化。**

异常处理、配置、测试、日志、Docker、部署说明。

**第五步：最后才追求高级方法。**

GraphRAG、DPO、学习式视觉模型、多传感器融合都属于后半程；baseline 没打通时禁止用高级名词掩盖工程问题。

---

# 例题一：200 份 PDF 的企业 RAG 知识库

## 真实需求卡片

**公开需求：** *RAG Chatbot Over 200 PDFs - Python, LangChain, Pinecone, OpenAI*  
**平台：** Upwork  
**发布时间：** 2026-08-19（页面在 2026-09-12 仍可访问）  
**客户背景：** 小型咨询公司，长期积累约 200 份 PDF，包括客户交付物、市场研究、内部 SOP、案例材料。  
**公开报价：** 50 美元 fixed-price（显著不能代表该需求的合理工程价格，只作为原始信息记录）。  
**原始链接：** [Upwork 原始页面：企业 RAG 知识库](https://www.upwork.com/freelance-jobs/apply/RAG-Chatbot-Over-200-PDFs-Python-LangChain-Pinecone-OpenAI_~022090007690068445607/)

### 客户明确要求（摘要）

- 使用 Python，并以 FastAPI 提供后端服务；
- 使用 LangChain 构建检索流程，LlamaIndex 经验也可；
- 使用 Pinecone，或 pgvector / Weaviate / Qdrant 等同类向量数据库；
- 使用 OpenAI API（如 GPT-4o 或同等级模型）生成答案；
- 能做成规模文档 ingestion：PDF 解析、chunking、embedding、metadata；
- 对多文档语料做 semantic/vector search；
- 使用 Supabase 做认证与按用户保存聊天历史；
- 能用 Next.js / React / TypeScript 做整洁的内部聊天界面；
- 回答必须以公司文档为依据，并返回 **源 PDF 文件名 + 页码**；
- 文档里没有答案时必须明确说不知道，不能用模型常识补答案；
- 第一阶段先交付核心检索管线，再依据真实文档表现进行 tuning；
- 未来可能扩展 Slack integration 与 role-based access。

## 训练版任务书

你要从零实现一个可演示、可评测、可 Docker 部署的“小型企业知识库”。开始时不需要真的准备 200 份 PDF，先用 10-30 份结构不同的 PDF 建立训练集，最后再扩展。

### 最终交付物

1. `ingest`：批量导入 PDF，保存文档、页码、段落等 metadata；
2. `index`：生成 embedding 并写入向量数据库；
3. `retrieve`：给定问题返回 top-k 相关 chunk；
4. `answer`：只基于检索证据生成回答；
5. `citation`：回答中能追溯到 PDF 文件名、页码、证据文本；
6. `abstention`：证据不足时拒答；
7. FastAPI 接口；
8. 一个最小聊天 UI；
9. 用户认证/聊天历史（第二阶段加入）；
10. 自动评测脚本；
11. Dockerfile / docker-compose；
12. README、架构图、已知限制、benchmark 报告。

## 推荐技术架构

```text
PDF files
   │
   ▼
Parser ──-> page/section metadata
   │
   ▼
Chunker
   │
   ▼
Embedding Model
   │
   ▼
Qdrant / pgvector
   │
Question
   │
   ▼
Query rewrite (optional)
   │
   ▼
Vector / Hybrid Retrieval
   │
   ▼
Reranker (advanced)
   │
   ▼
Evidence Pack
   │
   ▼
LLM grounded generation
   │
   ├──-> answer
   └──-> citations
```

建议第一版使用：**Python + FastAPI + Qdrant + 一个 embedding API/本地 embedding + 任一 LLM API + 简单 HTML/React UI**。不要一开始同时引入 LangChain、LlamaIndex、复杂 Agent 框架；先理解每层数据结构。

## Gate 设计


9.12 Q1
卡在完全不知道怎么开始这个任务完全没有相关基础知识，连你给的方案拆解都看不懂，先给我指个路，建立一下基本概念吧。首先pdf处理该怎么逐页解析？调用什么包？

我肯定是没有两百多个pdf文件的，先试试用高中化学+物理+生物的一共16本教科书吧，怎么样。这个我还是有的

### Gate 1A：先做到“能搜”，不要急着聊天

**任务：** 选 5-10 份 PDF，逐页解析为：

```json
{
  "doc_id": "...",
  "file_name": "xxx.pdf",
  "page": 12,
  "text": "..."
}
```

把文本分块，生成 embedding，写入向量库。输入一个自然语言问题，返回 top-5 chunk。

**通过标准：** 你手工准备 20 个问题，其中至少 16 个问题的正确证据出现在 top-5 中。

**如果卡住，学习：**

- PDF 文本层与页面结构；
- token、chunk size、overlap；
- cosine similarity / dot product；
- embedding 的输入输出；
- 向量数据库 collection / payload / filter。

**关键词：** `PDF parsing page metadata`, `text chunking RAG`, `embedding cosine similarity`, `Qdrant payload filtering`, `Recall@k retrieval`。

### Gate 1B：给检索做真正的评测

建立 `eval/questions.jsonl`：

```json
{"question":"...", "gold_doc":"A.pdf", "gold_page":4}
```

实现至少：

$$
Recall@k=\frac{\#\{\text{gold evidence appears in top-k}\}}{\#\{\text{queries}\}}
$$

再记录 MRR 或平均正确证据排名。

**通过标准：** 你能用数据解释：改变 chunk size、top-k 后，检索效果为什么变化。

### Gate 1C：加入 grounded generation 与 citation

Prompt 不允许模型凭常识作答，生成时只提供检索证据。输出最好采用结构化格式：

```json
{
  "answer": "...",
  "citations": [
    {"file":"A.pdf", "page":4, "quote":"..."}
  ],
  "answerable": true
}
```

**通过标准：**

- 引用确实来自提供给模型的 evidence；
- 文件名/页码不是模型自己生成，而是程序从 metadata 填入；
- 无证据问题能稳定拒答。

### Gate 1D：Web API + UI + 持久化

实现：

- `POST /ingest`
- `POST /chat`
- `GET /health`
- 会话 ID 与历史；
- 基本错误码和输入验证；
- 简单聊天页面。

**通过标准：** 新电脑执行 README 中的步骤可以启动，并完成“上传/索引/提问/查看引用”。

### Gate 1E：优秀档

加入并对比：

- BM25 + vector 的 hybrid retrieval；
- reranker；
- metadata filter；
- semantic/section-aware chunking；
- 证据不足阈值；
- prompt injection / malicious document 基础防护；
- latency 与 cost 统计。

## 交差与优秀

**能交差：** 200 份左右普通文本 PDF 可稳定 ingestion；答案有可验证引用；未知问题拒答；API/UI/Docker 可运行；有一个最小 eval set。

**优秀：** 你不仅能做系统，还能给客户一张表解释：不同 chunking、embedding、retrieval、reranker 对 Recall@k、citation accuracy、answer accuracy、latency 和 cost 的影响，并能根据失败案例定位到检索还是生成层。

## 本题你真正学到的能力

RAG 只是表面。本题真正训练的是：**数据管线 + 搜索系统 + LLM 约束 + API + 数据库 + 全栈交付 + 可验证性**。

---

# 例题二：LoRA/QLoRA + RAG/GraphRAG 的领域分类 POC

## 真实需求卡片

**公开需求：** *AI/NLP Freelancer needed - Fine-tune LLM + RAG/GraphRAG hands-on POC (legal charge classification)*  
**平台：** Upwork  
**发布时间：** 2026-09-09  
**公开报价：** 50 美元 fixed-price（明显更像学习型/占位预算，不应据此评价技术工作价值）。  
**原始链接：** [Upwork 原始页面：LLM 微调 + RAG/GraphRAG](https://www.upwork.com/freelance-jobs/apply/NLP-Freelancer-needed-Fine-tune-LLM-RAG-GraphRAG-hands-POC-legal-charge-classification_~022097800284743185555/)

### 客户明确要求（摘要）

客户希望通过一个小型法律罪名分类 POC 理解完整 AI pipeline，强调不要求生产级准确率，但要求开发者真正理解当前方法并能讲清楚。

- 收集/自行构造 30-50 条案例；
- 每条标注适用罪名与相关法律条文；
- 给出一份简短 annotation guideline；
- 构建基础 RAG（FAISS/Chroma）；
- 时间允许则做一个简单 GraphRAG：实体/关系抽取、小型知识图，并与纯向量 RAG 比较；
- 对小型开源 LLM 做 LoRA/QLoRA，示例模型包括 Qwen2.5-7B、Llama 3.2 3B；
- 用 held-out test set 做 Accuracy、F1、confusion matrix；
- 至少进行一次改进迭代：prompt、retrieval、hyperparameter，或有 preference data 时尝试 DPO；
- 交付 2-3 页过程与结果报告；
- 讲解 LoRA vs QLoRA、DPO/RLHF、GraphRAG vs vector RAG、hybrid retrieval、reranking、contextual retrieval、Self-RAG 等；
- 自己搭建 Colab / RunPod / local GPU 环境；
- 交付全部代码、标注数据和报告。

> 本习题集复现此题时，重点是学习技术流程，不应把模型输出当成现实法律意见。

## 训练版任务书

这道题的重点不是“把一个 7B 模型训起来”，而是建立完整实验逻辑：

$$
\text{Task definition}
\rightarrow \text{Dataset}
\rightarrow \text{Baselines}
\rightarrow \text{Fine-tuning}
\rightarrow \text{Evaluation}
\rightarrow \text{Error Analysis}
\rightarrow \text{Improvement}
$$

如果显存/数据不足，可先把领域替换成“技术支持工单分类”，但流程必须保持一致。

### 最终交付物

1. 数据集与标签说明；
2. train/validation/test split；
3. zero-shot baseline；
4. few-shot/prompt baseline；
5. 一个普通 RAG baseline；
6. LoRA 或 QLoRA SFT；
7. 训练配置、日志、模型 adapter；
8. Accuracy / macro-F1 / per-class precision-recall / confusion matrix；
9. 至少一次 controlled improvement experiment；
10. error analysis 文档；
11. 可选 GraphRAG / DPO；
12. README 与 2-3 页技术总结。

## 先建立“方法选择”认知

不要把“客户说微调”理解成“必须微调”。先区分：

- **RAG**：更适合给模型提供动态事实、企业知识、需要引用的内容；
- **SFT/LoRA**：更适合改变任务行为、输出模式、风格、分类边界；
- **DPO**：需要 preference pair，优化“偏好哪一种回答”；
- **Prompt/Few-shot**：如果任务本身简单，应先做便宜 baseline。

优秀工程师必须能够得出“这个问题不需要微调”的结论。

## Gate 设计

### Gate 2A：先把标签问题定义清楚

写 annotation guideline：每个类别的定义、边界、正例、反例、冲突如何处理。

**通过标准：** 随机抽 10 条样本，过一天后自己重新标，标签基本一致；最好找第二个人独立标一小部分，检查定义是否模糊。

**卡点学习：** supervised learning、label noise、class imbalance、data leakage、train/val/test split。

### Gate 2B：做不训练模型的 baseline

先跑：

1. zero-shot；
2. few-shot；
3. 如果存在法规/知识文本，再加 RAG。

保持同一 test set，记录每个 baseline。

**通过标准：** 有统一 evaluation script，任何模型都输出同一 JSON schema，评测代码不依赖具体模型。

### Gate 2C：LoRA/QLoRA

LoRA 的核心可写为：

$$
W' = W + \Delta W, \qquad \Delta W = BA, \qquad r \ll d.
$$

训练时冻结基础权重，只训练低秩 adapter。QLoRA 再把基础模型量化以降低显存占用。

你需要真正理解这些配置：

- `r` / rank；
- `lora_alpha`；
- target modules；
- learning rate；
- effective batch size；
- gradient accumulation；
- epoch；
- warmup；
- max sequence length；
- quantization config。

**通过标准：** 不是“Trainer 跑完没报错”，而是能回答训练前后 test F1 的变化、过拟合迹象、最差类别及原因。

### Gate 2D：错误分析与一次改进

把错误至少分类为：

- 标签本身模糊；
- 类别不平衡；
- prompt/format failure；
- 模型知识不足；
- retrieval failure；
- 相似类别边界混淆；
- 长文本截断/上下文丢失。

然后只选一个最主要失败来源做改进实验。

**通过标准：** 改动前后有同一 test set 的对照表，且你能解释为什么理论上它应该改善对应失败。

### Gate 2E：优秀档

可以尝试：

- GraphRAG 与 vector RAG 对比；
- reranker / hybrid retrieval；
- DPO（只有你真的有 chosen/rejected preference pairs 时）；
- quantization / inference benchmark；
- calibration / confidence；
- 数据量学习曲线。

## 交差与优秀

**能交差：** 能独立准备数据、跑 baseline、完成 LoRA/QLoRA、严格隔离 test set、给出指标/混淆矩阵/错误分析，并能复现训练。

**优秀：** 能判断 fine-tuning 是否真的必要；能说明提升是来自数据、检索、prompt 还是权重更新；能设计一轮有因果意义的对照实验，而不是“随机换超参数直到分数更高”。

## 本题真正训练的能力

**机器学习实验设计、训练工程、评测意识、方法选择与 scientific debugging。**

---

# 例题三：CPU/Edge 上运行的医学图像质量检测

## 真实需求卡片

**公开需求：** *Computer Vision / Edge ML Engineer - Lightweight Medical Image Quality Assessment*  
**平台：** Upwork  
**发布时间：** 2026-09-07  
**公开预算：** 页面标题 1,800 美元 fixed-price；需求正文写 indicative budget 1,800-2,500 美元，预计 4-6 周 part-time。  
**原始链接：** [Upwork 原始页面：Edge CV 图像质量检测](https://www.upwork.com/freelance-jobs/apply/Computer-Vision-Edge-Engineer-Lightweight-Medical-Image-Quality-Assessment_~022096930437957381037/)

### 客户明确要求（摘要）

开发轻量 Image Quality Assessment 模块，检测包括：

- blur / out-of-focus；
- under-exposure / over-exposure；
- excessive glare / reflections；
- 其他图像质量失败。

系统输出要可解释：PASS/FAIL、拒绝原因，必要时输出每项 quality score。

**关键约束：低计算成本。** 平均 CPU 可高效运行，最好未来能移动端/边缘端部署；不接受依赖大模型、GPU 或云端推理的方案。客户明确希望工程师判断何时经典 OpenCV 足够，何时才需要小型 learned model。

客户要求的工作包括：

1. 分析有标签的医学图像并归纳失败模式；
2. 针对每个质量指标 benchmark 候选方法；
3. 构建 IQA pipeline；
4. 确定/验证阈值或分类标准；
5. 优化 CPU/edge；
6. 在 held-out dataset 上评估；
7. benchmark inference latency、memory、model size；
8. 交付可维护 Python 代码、测试、技术文档；
9. 打包为 REST API/microservice，最好 FastAPI + Docker；
10. API 接收图像并返回结构化结果、错误处理、输入验证；
11. 如使用 learned component，要交付模型权重。

## 训练版任务书

这题非常适合训练“不要滥用深度学习”。你要实现一个多指标图像质量判定器，并证明所选方法在准确率与资源之间是合理折中。

### 推荐输出

```json
{
  "status": "FAIL",
  "issues": ["blur", "over_exposure"],
  "scores": {
    "sharpness": 0.21,
    "exposure": 0.93,
    "glare": 0.12
  }
}
```

## 候选 baseline

### Blur

最简单可从 Laplacian variance 开始：

$$
s_{blur} = Var(\nabla^2 I)
$$

也可以比较频域高频能量、Tenengrad/Sobel gradient 等。

### Exposure

可从直方图、均值、percentile、饱和像素比例出发。例如：

$$
r_{high}=\frac{\#\{I>t_h\}}{N},\qquad
r_{low}=\frac{\#\{I<t_l\}}{N}.
$$

### Glare / reflection

可尝试 HSV/亮度高值区域、局部饱和、高亮 connected component、面积比例与形态特征；如果规则对场景变化太敏感，再引入小型分类器。

## Gate 设计

### Gate 3A：数据体检

先不要训练。

建立数据表：文件名、标签、分辨率、设备/来源（若有）、质量问题是否多标签、类别数量。随机查看每类至少几十张，写一页 failure taxonomy。

**通过标准：** 你能解释每个标签在视觉上“长什么样”，知道有没有明显 label noise / class imbalance。

### Gate 3B：经典 CV baseline

对每种质量问题至少实现一个无学习方法，并用训练/验证集调阈值。

**通过标准：** test set 在开发前就封存；阈值不能看 test set 调。

指标至少：Precision、Recall、F1。对于医疗采集质量控制，通常还要明确 false pass 与 false reject 哪种更昂贵。

### Gate 3C：只有 baseline 不够时才加小模型

候选：MobileNetV3、EfficientNet-Lite、tiny CNN，或只对某个最难的指标使用 learned classifier。

做一张方法比较表：

| 方法 | F1 | CPU p50/p95 latency | Peak/RSS memory | Model size |
|---|---:|---:|---:|---:|
| Rule baseline | | | | 0 |
| Small CNN | | | | |
| Larger model | | | | |

**通过标准：** 选择不是“谁分最高就用谁”，而是对 accuracy/latency/memory/model size 做工程权衡。

### Gate 3D：API 与容器化

实现 `POST /assess`，输入 JPEG/PNG，输出结构化 JSON；限制文件大小、检查 MIME/解码失败、记录耗时；Docker 中不依赖 GPU。

### Gate 3E：优秀档

- ONNX Runtime / TensorFlow Lite benchmark；
- 不同 CPU 线程数；
- INT8 quantization；
- threshold calibration；
- 不同设备/曝光条件下的 robustness；
- “为什么失败”的可视化证据（例如高亮 glare 区域）；
- dataset shift 检测思路。

## 交差与优秀

**能交差：** held-out test 指标明确；每类质量有独立指标；CPU 可运行；API/Docker 可集成；源码、配置、模型权重、测试和运行说明齐全。

**优秀：** 能证明为什么某些指标用经典 CV，某些指标用 learned model；能定量展示精度-延迟-内存-模型大小 Pareto trade-off，并对误判做系统分析。

## 本题真正训练的能力

**算法选择、传统视觉、轻量深度学习、benchmark、边缘部署、性能工程。**

---

# 例题四：连接 CRM 的 Lead Qualification AI Agent

## 真实需求卡片

**公开需求：** *Experienced AI Agent Developer Needed - RAG Pipeline, n8n Workflow Automation, LLM Chatbot*  
**平台：** Upwork  
**页面中的实际任务：** 一个单 Agent 的销售线索 qualification workflow，而不是复杂多智能体系统。  
**发布时间：** 2026 年 9 月上旬。  
**原始链接：** [Upwork 原始页面：CRM Lead Qualification Agent](https://www.upwork.com/freelance-jobs/apply/Experienced-Agent-Developer-Needed-RAG-Pipeline-n8n-Workflow-Automation-LLM-Chatbot_~022095572193205961798/)

### 客户明确要求（摘要）

目标是做一个 production-ready lead qualification agent：

- incoming lead 通过 webhook 进入系统；
- Python + FastAPI 后端；
- LangChain 组织 agent logic；
- GPT-4o 或 Claude 3.5 Sonnet 作为 LLM；
- 根据预定义 qualification criteria 自动打分；
- 通过 GoHighLevel 或 HubSpot REST API 读写 CRM；
- 把 score 与 qualification result 写回 CRM；
- 根据分数触发个性化 follow-up email；
- 优雅处理 edge cases；
- 记录每一次 decision，便于审核；
- low-confidence 结果交人工 review，而不是强行猜；
- 不需要 UI，也不要复杂多层 architecture；
- 至少用 3 种 lead scenario 完整测试；
- 交付干净、有文档的代码和短 walkthrough。

## 先纠正一个重要认知：这题不需要“很 Agent”

真实业务往往更适合一个可控 workflow：

```text
Webhook
  ↓
Schema Validation
  ↓
Normalize / Enrich
  ↓
LLM Structured Scoring
  ↓
Confidence Gate
  ├── low → Human Review
  └── high
       ↓
CRM Update
       ↓
Follow-up Trigger
       ↓
Audit Log
```

如果普通状态机可以完成，就不要为了“Agent”两个字引入复杂自主规划。

## 训练版任务书

### 最终交付物

1. FastAPI webhook endpoint；
2. Pydantic 输入 schema；
3. lead scoring criteria 配置；
4. LLM structured output；
5. confidence / human review 机制；
6. mock CRM connector；
7. 真实 HubSpot/其他 CRM connector（有测试账号再接）；
8. follow-up action abstraction；
9. retry / timeout / error handling；
10. idempotency 防重复处理；
11. audit log；
12. 单元测试 + 端到端测试；
13. Docker 与部署说明。

## Gate 设计

### Gate 4A：完全不用 LLM，先把系统骨架打通

POST 一个 JSON：

```json
{
  "lead_id":"L001",
  "company_size":120,
  "budget":50000,
  "need":"..."
}
```

用固定规则打分并返回结果。

**通过标准：** schema validation、HTTP status、日志、测试都存在。

**卡点学习：** HTTP、REST、JSON、FastAPI、Pydantic、同步/异步基础。

### Gate 4B：引入 LLM，但必须 Structured Output

让模型输出：

```json
{
  "score": 82,
  "qualified": true,
  "reason_codes": ["budget_fit", "company_size_fit"],
  "confidence": 0.88
}
```

不要解析自由文本。

**通过标准：** 模型返回异常时系统不会崩；格式错误可重试或降级。

### Gate 4C：接 CRM

先写一个 interface：

```text
get_lead(id)
update_lead(id, result)
create_note(id, note)
```

本地用 FakeCRM 实现，测试通过后再换成 HubSpot connector。

**通过标准：** 外部 API 超时、429、5xx 时有明确行为；secret 通过环境变量提供。

### Gate 4D：可靠性

必须理解并实现：

- timeout；
- retry with backoff；
- idempotency key；
- duplicate webhook；
- rate limit；
- logging / correlation ID；
- partial failure；
- human-in-the-loop。

**通过标准：** 人为制造 CRM 500、LLM timeout、重复 webhook、坏 JSON，系统行为符合设计。

### Gate 4E：优秀档

- job queue / background worker；
- prompt/version tracking；
- cost + token monitoring；
- tracing；
- replay failed events；
- deterministic rules + LLM hybrid；
- per-decision audit trail；
- security review（webhook signature、PII logging）。

## 交差与优秀

**能交差：** webhook → score → CRM writeback → follow-up 的完整链稳定运行，至少三个业务场景测试；失败有日志；低置信度走人工。

**优秀：** 能把系统当分布式业务流程看，而不是“调一次 LLM API”；能设计幂等、重试、审计、异常恢复、成本控制，并明确哪些判断应该由规则完成、哪些才交给 LLM。

## 本题真正训练的能力

**系统集成、Web API、第三方服务、可靠性工程、可观测性，以及“不要过度 Agent 化”的判断力。**

---

# 例题五：LiDAR / 3D 点云感知管线

## 真实需求卡片

**公开需求：** *LiDAR / 3D Point Cloud Perception Engineer*  
**平台：** Upwork  
**发布时间：** 2026 年 8 月中旬左右（页面在 2026-09-12 显示约 4 周前发布）  
**项目形态：** 6 个月以上、每周 30 小时以上的长期复杂项目；公开时薪区间 10-20 美元（同样只记录原始信息）。  
**原始链接：** [Upwork 原始页面：LiDAR / 3D 点云感知](https://www.upwork.com/freelance-jobs/apply/LiDAR-Point-Cloud-Perception-Engineer_~022088079541550497426/)

### 客户背景与要求（摘要）

客户在开发车辆式地理空间监测平台 SmartView，融合：

- LiDAR；
- RGB camera；
- GNSS；
- IMU。

团队已有 CV、定位/GNSS、GIS、后端人员，这个角色只负责 **LiDAR / 3D point-cloud perception layer**，并非主要负责 SLAM 或 GIS。

工作包括：

- 处理移动平台原始 LiDAR 点云；
- filtering / preprocessing；
- ground / non-ground separation；
- clustering 与 feature extraction；
- 3D object/feature detection 或 segmentation；
- 计算 object-level 属性：XYZ centroid、range、height、width/dimensions、3D geometry/bounding regions；
- 必要时开发 geometric detector；
- 在 sensor/vehicle frame 产生稳定 object position；
- 支持 camera ↔ LiDAR association；
- 与定位工程师协作，把观测变换到 world coordinates；
- 建立可重复 Python/C++ pipeline；
- 典型库：PCL、Open3D、PDAL；
- 典型传感器：Hesai、Ouster、Velodyne；
- Camera-LiDAR fusion、ROS2、GNSS/IMU 为加分项。

客户第一阶段想看到的链条非常清楚：

```text
LiDAR point cloud
→ selected object / feature
→ 3D geometry
→ XYZ position in vehicle frame
→ structured output
```

并要求应聘者证明自己**亲手实现过**点云处理，而不是只看过/标注过点云。

## 训练版任务书

你要做一个独立的 point-cloud object extraction pipeline。第一版不碰实时传感器，先用公开 PCD/PLY/KITTI 类数据。

### 最终输出样例

```json
{
  "frame_id":"000123",
  "objects":[
    {
      "id":3,
      "centroid_vehicle":[4.18,-1.22,0.83],
      "dimensions":[0.52,0.41,1.77],
      "yaw":0.31,
      "point_count":428,
      "range":4.43
    }
  ]
}
```

## 推荐 pipeline

$$
P=\{p_i=(x_i,y_i,z_i,r_i)\}_{i=1}^N
$$

```text
Raw Point Cloud
   ↓
Coordinate sanity check
   ↓
ROI crop
   ↓
Voxel downsample
   ↓
Outlier removal
   ↓
Ground segmentation
   ↓
Clustering
   ↓
Cluster filtering
   ↓
Centroid + AABB/OBB + dimensions
   ↓
Structured output + visualization
```

地面可以先用 RANSAC plane：

$$
ax+by+cz+d=0.
$$

聚类可先做 DBSCAN / Euclidean clustering。对 cluster \(C_j\)：

$$
c_j=\frac{1}{|C_j|}\sum_{p_i\in C_j}p_i.
$$

再计算 AABB/OBB、长宽高、range。

## Gate 设计

### Gate 5A：你必须先真正理解坐标系

对每个点明确它在哪个 frame。建立统一记号：

$$
\mathbf p_{vehicle}=
T_{vehicle\leftarrow lidar}\mathbf p_{lidar}.
$$

**通过标准：** 能手工构造几个点，验证平移/旋转后位置符合预期；能解释 extrinsic 的方向，不靠“试到点云看起来对”。

**卡点学习：** homogeneous coordinates、SE(3)、rotation matrix、quaternion、extrinsic calibration。

### Gate 5B：单帧几何 baseline

完成 crop、voxel、outlier、plane segmentation、clustering、bounding box。

**通过标准：** 对一组固定 point cloud，每个模块能保存可视化中间结果；参数都在 config，不散落在代码里。

### Gate 5C：对象测量

输出 centroid、range、dimensions、AABB/OBB；建立至少 20 个手工/半自动标注对象验证误差。

可定义：

$$
e_c=\|\hat c-c^*\|_2,
$$

并统计尺寸误差。

### Gate 5D：从 sensor frame 到 vehicle/world frame

先只做已知外参变换，再引入 pose trajectory：

$$
T_{world\leftarrow lidar}(t)
=
T_{world\leftarrow vehicle}(t)
T_{vehicle\leftarrow lidar}.
$$

**通过标准：** 同一静态物体在连续帧变换到 world frame 后应聚集到近似同一位置。

### Gate 5E：Camera-LiDAR association

如果有相机内外参：

$$
\mathbf p_{cam}=T_{cam\leftarrow lidar}\mathbf p_{lidar},
$$

再做 pinhole projection 到图像平面。先实现几何投影，再讨论 detection fusion。

### Gate 5F：优秀档

- 多帧融合与 tracking；
- 运动补偿 / timestamp synchronization；
- CenterPoint / PointPillars 等 learned detector；
- ROS2 node；
- 点云 pipeline profiling；
- 不同传感器/密度下 robustness；
- GNSS/IMU trajectory 与 world-frame map 对齐。

## 交差与优秀

**能交差：** raw cloud → filtering → ground removal → clustering → object geometry → vehicle-frame XYZ 全链路可复现；代码/可视化/结构化输出齐全。

**优秀：** 不仅能跑算法，还能系统处理 frame convention、外参、移动平台、时间同步、传感器噪声、遮挡、稀疏点、误检，并知道什么时候该从几何方法升级到 learned 3D detector。

## 本题真正训练的能力

**3D 几何、点云算法、坐标变换、传感器工程、性能与数据可视化。**

---

# 五道题的推荐实现顺序

建议顺序不是按“最酷”排列，而是按 **复用能力最大、每一步都有可见成果、失败成本较低** 排列：

$$
\boxed{
\text{RAG} \rightarrow
\text{Edge CV} \rightarrow
\text{Agent} \rightarrow
\text{Fine-tuning} \rightarrow
\text{LiDAR/3D}
}
$$

## 第一阶段：RAG - 学会“把 AI 做成系统”

你第一次集中补：

- Python 项目结构；
- FastAPI；
- 向量数据库；
- Docker；
- 基本前端/HTTP；
- evaluation harness；
- logging/config/secrets。

这一题做完之后，后面四题不再重复学习“怎么把一个算法包装成能运行的服务”。

**阶段结束标准：** 你能把项目发给别人，对方只看 README 就启动成功；你有一套自动评测而不是 Demo-only。

## 第二阶段：Edge CV - 学会“算法不等于模型”

复用 FastAPI/Docker/testing，再补：

- OpenCV；
- image statistics；
- threshold tuning；
- train/val/test；
- precision/recall/F1；
- PyTorch 小模型；
- CPU benchmark/ONNX。

**阶段结束标准：** 你第一次真正能用数据做工程取舍，而不是只看 accuracy。

## 第三阶段：Agent - 学会“商业系统会怎么坏”

复用 API/Docker，再补：

- webhook；
- 外部 REST API；
- structured output；
- timeout/retry；
- idempotency；
- rate limit；
- audit log；
- human-in-the-loop。

**阶段结束标准：** 你开始能把故障当作设计的一部分，而不是 exception 出现以后再补。

## 第四阶段：Fine-tuning - 学会“真正做一次 ML 实验”

此时再进入训练工程，因为你已经有评测习惯。

补：

- Transformers；
- tokenizer/chat template；
- SFT；
- PEFT / LoRA；
- quantization / QLoRA；
- GPU memory；
- training curves；
- experiment tracking；
- error analysis。

**阶段结束标准：** 你能证明“为什么要微调、微调是否真的提升、提升来自哪里”。

## 第五阶段：LiDAR/3D - 作为领域型 capstone

这一题不一定是最难写代码，但它最考察“概念错误会让所有结果看起来似乎对、实际上全错”的能力。

补：

- Open3D/PCL；
- 3D geometry；
- SE(3)；
- sensor extrinsics；
- 3D clustering/segmentation；
- multi-frame / time sync；
- ROS2（扩展）。

**阶段结束标准：** 你能独立从 raw sensor-like data 做出结构化物体结果，并用定量/可视化两种方式验证坐标与测量正确性。

---

# 不要按“周”学习，按 Gate 学习

时间计划容易制造错觉：学了两周不等于会。建议以 Gate 作为进度单位。

每个 Gate 的工作循环固定为：

1. **写验收标准**：先写“什么结果算过”；
2. **画数据流**：输入、输出、数据结构；
3. **做最小实验**：尽量不引入框架；
4. **记录 baseline**；
5. **遇到卡点再学**；
6. **回到工程复现**；
7. **自动测试/评测**；
8. **写一页复盘**：学到了什么、失败原因、下一 Gate。

## 卡点分类法

以后遇到问题，不要笼统地说“RAG 效果不好”“模型不行”。先把卡点归类。

### A. 概念卡点

症状：不知道为什么这样设计，例如“不理解 embedding 为什么能检索”。

处理：暂停写工程代码，做一个 20-50 行 toy example，把数学对象/输入输出搞清楚。

### B. API/框架卡点

症状：知道原理，但不会 FastAPI dependency、Qdrant filter、PEFT config。

处理：查 **官方文档 + 最小 runnable example**，不要先搜大项目源码。

### C. Debug 卡点

症状：系统行为不对，但不知道哪层出问题。

处理：把 pipeline 切开，保存每一层中间产物；给每层写独立测试。

### D. 性能卡点

症状：结果正确但慢/爆显存/占内存。

处理：先 profile，再优化；禁止凭感觉重写。

### E. 方案卡点

症状：有三四种技术路线，不知道选哪个。

处理：回到业务指标，设计最小 benchmark；选择不是“最先进”，而是“约束下最合适”。

---

# 当你卡住时，具体该去学什么

下面是一张“需求 -> 知识补丁”索引。你不需要提前全部学完。

| 你遇到的问题 | 先补的概念 | 再看工具/文档 | 学到什么程度就回项目 |
|---|---|---|---|
| 不懂 REST/API | HTTP method、status、JSON、client-server | FastAPI Tutorial | 能写 3 个 endpoint + Pydantic validation |
| 环境总坏 | venv/uv/依赖锁、环境变量 | Python packaging | 新环境可重复安装 |
| 不懂 Docker | image、container、volume、port | Docker Get Started | 能容器化单个 FastAPI 服务 |
| RAG 搜不到 | embedding、similarity、Recall@k、chunk | Qdrant/LangChain retrieval docs | 能做独立 retrieval eval |
| 回答幻觉 | grounded generation、citation、abstention | LLM structured output/prompt docs | 无证据时能稳定拒答 |
| LoRA 不懂 | Transformer linear layer、low rank、SFT | Hugging Face PEFT | 能解释 rank/alpha/target module |
| QLoRA 爆显存 | quantization、optimizer memory、batch | PEFT/bitsandbytes | 能估算主要显存组成并调 batch |
| CV 阈值不稳 | histogram、gradient、ROC/PR | OpenCV + sklearn metrics | 能在 val set 选阈值 |
| 模型快但不准/准但慢 | profiling、latency、memory、Pareto | ONNX Runtime | 能输出 benchmark 表 |
| webhook 重复 | at-least-once delivery、idempotency | API provider docs | 重放同事件不会重复副作用 |
| CRM 429/5xx | retry/backoff/rate limit | HTTP/API docs | 故障注入测试通过 |
| 点云很乱 | voxel、outlier、RANSAC、DBSCAN | Open3D point cloud docs | 单帧可视化每一步 |
| 坐标对不上 | SE(3)、frame convention、extrinsic | 机器人学/ROS tf2 | 手算+单测验证变换方向 |

---

# 推荐资料入口（只作为“卡住时的工具箱”）

原则：**优先官方文档，其次高质量教材/论文，再其次博客。不要为了读资料而读资料。**

## 通用工程

- [FastAPI 官方 Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker Get Started](https://docs.docker.com/get-started/)
- Python `logging` / `pytest` / `pydantic` 官方文档（遇到再查）。

## RAG

- [Qdrant Quickstart](https://qdrant.tech/documentation/quick-start/)
- [LangChain Retrieval / LangChain Docs](https://docs.langchain.com/)
- 学习重点不是框架 API，而是：chunk、embedding、retrieval、rerank、metadata、evaluation。

## Fine-tuning

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face PEFT / LoRA](https://huggingface.co/docs/peft/)
- [Hugging Face TRL（SFT/DPO 等）](https://huggingface.co/docs/trl/)
- 先学 SFT + LoRA；没有 preference data 不要为了炫技硬做 DPO。

## CV / Edge

- [OpenCV Tutorials](https://docs.opencv.org/)
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)

## 3D / Robotics

- [Open3D Point Cloud Tutorials](https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html)
- [ROS 2 Docs / tf2](https://docs.ros.org/)
- [PCL Tutorials](https://pcl.readthedocs.io/)
- 如果进入 learned 3D detection，再看 PointPillars / SECOND / CenterPoint 及 OpenPCDet / MMDetection3D；不要提前。

---

# 我作为“助手/老师”应该怎样参与

以后我们可以把这份习题集当成长期主线。为了让你真正自己做，而不是我替你做完，建议我们的交互遵循下面的模式。

## 你每次来问问题时，尽量告诉我五件事

```text
题号 / Gate：例如 1B Retrieval Evaluation
目标：我现在希望做到什么
实际结果：现在发生了什么
最小材料：关键代码 / 日志 / 输出 / 截图
我已经尝试过：A、B、C
```

如果你暂时不知道该提供什么，也可以直接把现场发来，我帮你判断需要什么信息。

## 我应该按四层回答你

1. **定位：** 这个问题属于数据、算法、评测、接口、部署还是基础概念；
2. **原理：** 把必要理论讲到足以 debug，而不是堆专有名词；
3. **最小实验：** 设计一个最小可复现实验确认判断；
4. **工程实现：** 再告诉你在当前项目中怎么改。

如果你的问题暴露了前置知识缺口，我应该明确告诉你：

> “先不要继续改主项目，花一小段时间做这个 toy exercise；做到 X 就回来。”

而不是让你去“系统学完一门课”。

## 哪些东西应该坚持自己手搓

为了真正建立能力，建议以下内容至少第一版自己写：

- 每题的 architecture diagram；
- 数据结构/schema；
- baseline；
- evaluation script；
- 关键算法模块；
- failure analysis；
- README 的核心使用说明。

AI/Codex 可以帮助：

- 脚手架；
- boilerplate；
- 测试模板；
- API SDK 用法；
- 重构；
- 查错误；
- 生成重复性较高的工程代码。

但有一条硬规则：**任何进入核心 pipeline 的代码，你至少应该能解释输入、输出、主要假设、失败方式。**

---

# 能力等级与最终“毕业标准”

## Level 0：教程使用者

- 会把别人仓库跑起来；
- 换数据就经常不知道哪里改；
- 没有独立评测。

## Level 1：独立实现者

- 能从需求拆出模块；
- 能自己实现 baseline；
- 能把项目跑通。

## Level 2：可收费交付

- 有明确验收标准；
- 有 test/eval；
- 能部署；
- 能处理常见异常；
- 能写 README；
- 能解释限制；
- 客户不需要理解你的代码就能使用交付物。

## Level 3：优秀 Freelancer / AI Engineer

- 能从业务指标倒推技术；
- 能拒绝不必要的复杂模型；
- 能做方案对比；
- 能定量解释 trade-off；
- 能定位 failure layer；
- 能设计可靠性与维护方案；
- 会问正确的问题，而不是等客户写技术方案。

## 五题毕业检查表

当你完成五题后，应至少能独立做到：

- [ ] 把模糊需求改写成验收标准；
- [ ] 画端到端 architecture；
- [ ] 设计最小 baseline；
- [ ] 建立 test/eval set；
- [ ] 用指标做方法对比；
- [ ] 写 Python 项目而不是 notebook-only；
- [ ] 提供 FastAPI/结构化接口；
- [ ] Docker 化并复现；
- [ ] 对外部 API 做 timeout/retry/idempotency；
- [ ] 对模型输出做 schema validation；
- [ ] 训练至少一个 LoRA/QLoRA 并完成 error analysis；
- [ ] 做 CPU latency/memory benchmark；
- [ ] 处理一次真实 3D point-cloud pipeline；
- [ ] 明确 frame/extrinsic 并用单测验证；
- [ ] 写出客户可读的交付文档。

---

# 每个项目的 Definition of Done 模板

每道题最终都复制下面模板填写。

## 功能

- [ ] 输入格式明确并验证；
- [ ] 主功能端到端工作；
- [ ] 输出 schema 稳定；
- [ ] 关键 edge cases 已定义。

## 正确性

- [ ] 有冻结的 test/eval set；
- [ ] 有 baseline；
- [ ] 有最终指标；
- [ ] 有 failure/error analysis；
- [ ] 优化有对照实验。

## 工程

- [ ] clean environment 可安装；
- [ ] config/secrets 分离；
- [ ] 日志充分；
- [ ] 核心模块有 test；
- [ ] Docker 可启动；
- [ ] API/CLI 文档完整。

## 交付

- [ ] README；
- [ ] architecture 图；
- [ ] 示例输入输出；
- [ ] benchmark；
- [ ] 已知限制；
- [ ] Demo/演示步骤；
- [ ] 源码、配置、模型/adapter（如适用）。

---

# 五个真实需求来源

1. Upwork, *RAG Chatbot Over 200 PDFs - Python, LangChain, Pinecone, OpenAI*, posted 2026-08-19.  
   [Upwork 原始页面：企业 RAG 知识库](https://www.upwork.com/freelance-jobs/apply/RAG-Chatbot-Over-200-PDFs-Python-LangChain-Pinecone-OpenAI_~022090007690068445607/)

2. Upwork, *AI/NLP Freelancer needed - Fine-tune LLM + RAG/GraphRAG hands-on POC (legal charge classification)*, posted 2026-09-09.  
   [Upwork 原始页面：LLM 微调 + RAG/GraphRAG](https://www.upwork.com/freelance-jobs/apply/NLP-Freelancer-needed-Fine-tune-LLM-RAG-GraphRAG-hands-POC-legal-charge-classification_~022097800284743185555/)

3. Upwork, *Computer Vision / Edge ML Engineer - Lightweight Medical Image Quality Assessment*, posted 2026-09-07.  
   [Upwork 原始页面：Edge CV 图像质量检测](https://www.upwork.com/freelance-jobs/apply/Computer-Vision-Edge-Engineer-Lightweight-Medical-Image-Quality-Assessment_~022096930437957381037/)

4. Upwork, *Experienced AI Agent Developer Needed - RAG Pipeline, n8n Workflow Automation, LLM Chatbot*, posted early September 2026; page task describes a focused CRM lead-qualification agent.  
   [Upwork 原始页面：CRM Lead Qualification Agent](https://www.upwork.com/freelance-jobs/apply/Experienced-Agent-Developer-Needed-RAG-Pipeline-n8n-Workflow-Automation-LLM-Chatbot_~022095572193205961798/)

5. Upwork, *LiDAR / 3D Point Cloud Perception Engineer*, posted around August 2026.  
   [Upwork 原始页面：LiDAR / 3D 点云感知](https://www.upwork.com/freelance-jobs/apply/LiDAR-Point-Cloud-Perception-Engineer_~022088079541550497426/)

---

# 版本约定

**v0.1（2026-09-12）**：建立五道真实需求母题、Gate、实现路线、卡点导航和统一 Definition of Done。

后续建议每完成一个 Gate 就更新一次：

- 实际采用的技术栈；
- 真实 benchmark；
- 遇到的坑；
- 你自己的结论；
- 是否需要调整下一 Gate。

这份文档最终应该逐渐从“习题集”变成你的 **AI Engineering Playbook + Portfolio 证据链**。
