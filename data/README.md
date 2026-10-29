# 运行数据目录

- uploads：上传原件，沿用内容摘要子目录及原文件名。
- cache：每份文档独立保存文本 JSON、向量 NPY、BM25 PKL、HNSW BIN 和建库记录。legacy 子目录保留迁移前的派生文件。
- conversations：聊天 JSONL、原有元数据及锁文件。
- settings：当前知识库设置、旧路径映射及迁移前后校验记录。legacy_knowledge_base.json 保留迁移前设置。

以上运行数据不进入 Git。上传原件、聊天和设置需要另行备份；缓存通常可重建。空目录会在使用或迁移时创建。
