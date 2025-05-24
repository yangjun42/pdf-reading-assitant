# RAG增强版PDF阅读助手 - 部署指南

## 🚀 概述

这是PDF阅读助手的RAG增强版本，集成了Azure AI Search进行语义检索，解决了章节摘要和问答的上下文割裂问题。

## 📋 前置要求

### 1. Azure服务配置

#### Azure AI Search
1. 在Azure门户创建Azure AI Search服务
2. 获取服务端点和管理员密钥
3. 确保选择支持语义搜索的定价层（Basic或以上）

#### GitHub Models (推荐)
1. 获取GitHub Token并启用Models访问权限
2. 或者配置Azure OpenAI作为备用方案

### 2. 本地环境

- Python 3.8+
- pip或conda包管理器

## 🛠️ 安装步骤

### 1. 克隆并进入项目目录
```bash
cd /Users/yangju/Library/CloudStorage/OneDrive-UniversityofHelsinki/Projects/pdf-reading-assitant
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv_rag
source venv_rag/bin/activate  # macOS/Linux
# 或者
venv_rag\Scripts\activate     # Windows
```

### 3. 安装RAG版本依赖
```bash
pip install -r requirements_rag.txt
```

### 4. 配置环境变量
```bash
cp env_rag_example.txt .env
```

编辑 `.env` 文件，配置以下关键变量：
```env
# 必需：GitHub Models API Key
GITHUB_TOKEN=ghp_your_actual_github_token

# 必需：Azure AI Search配置
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your_search_admin_key

# 可选：Azure OpenAI备用配置
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
```

## 🏃‍♂️ 运行应用

### 启动RAG增强版服务器
```bash
python app_rag_enhanced.py
```

服务器将在 `http://localhost:8001` 启动（注意端口8001避免与原版本冲突）

### 同时运行两个版本（可选）
```bash
# 终端1：运行原版本
python app_semantic_kernel.py  # 端口8000

# 终端2：运行RAG增强版
python app_rag_enhanced.py     # 端口8001
```

## 🔍 功能特性

### RAG增强功能
1. **文档索引化**：上传PDF时自动创建Azure Search索引
2. **语义检索**：在摘要和问答时检索相关内容片段
3. **上下文增强**：结合当前章节和相关参考内容
4. **智能分块**：文档自动分割为重叠的语义块

### 新增API响应字段
- `rag_enabled`: 指示RAG功能是否启用
- `operation_type`: 操作类型（如"rag_summarize", "rag_question"）
- `extraction_method`: 包含"_with_rag"后缀表示RAG增强

## 🧪 测试RAG功能

### 1. 上传PDF文档
- 选择包含多个章节的学术论文
- 观察响应中的`rag_enabled: true`

### 2. 测试跨章节摘要
- 为某个章节生成摘要
- 检查是否包含其他章节的相关信息

### 3. 测试智能问答
- 询问需要跨章节信息的问题
- 观察回答是否引用了多个来源

## 🐛 故障排除

### Azure Search连接问题
```bash
# 检查网络连接
curl -H "api-key: YOUR_KEY" "https://your-service.search.windows.net/indexes?api-version=2023-11-01"
```

### RAG功能被禁用
检查日志中的警告信息：
```
Azure Search credentials not found. RAG features will be disabled.
```

解决方案：确保`.env`文件中的Azure Search配置正确

### 内存使用过高
- 调整环境变量中的RAG配置：
```env
RAG_CHUNK_SIZE=500      # 减小块大小
RAG_TOP_K=3            # 减少检索数量
```

## 📊 性能监控

访问 `http://localhost:8001/cache/stats` 查看：
- 内容缓存统计
- RAG索引缓存状态
- 启用状态确认

## 🔄 版本对比

| 功能 | 原版本 | RAG增强版 |
|------|--------|-----------|
| 基础PDF处理 | ✅ | ✅ |
| 章节摘要 | ✅ | ✅ + 跨章节参考 |
| 问答功能 | ✅ | ✅ + 语义检索 |
| 上下文范围 | 单章节 | 全文档语义检索 |
| 部署复杂度 | 简单 | 需要Azure Search |
| 响应质量 | 基础 | 显著提升 |

## 📈 下一步计划

完成RAG集成后，可以继续以下改进：
1. 多Agent协作架构
2. MCP（Model Context Protocol）集成
3. 向量数据库优化
4. 实时协作功能

## 🔗 相关链接

- [Azure AI Search文档](https://docs.microsoft.com/azure/search/)
- [Semantic Kernel文档](https://learn.microsoft.com/semantic-kernel/)
- [GitHub Models](https://github.com/marketplace/models) 