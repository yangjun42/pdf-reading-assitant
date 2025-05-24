# PDF文档阅读助手

一个智能的PDF文档分析系统，支持文档分析、章节摘要和智能问答功能。

## 功能特点

- 📄 **智能文档分析**: 自动提取PDF目录结构、标题、关键词和摘要
- 🔍 **多种提取方式**: 优先使用PyMuPDF提取PDF书签，回退到LLM分析
- 💬 **章节问答**: 基于章节内容的智能问答系统
- 📝 **章节摘要**: 自动生成章节详细摘要
- 💰 **智能节省成本**: 优先使用免费的GitHub Models，超限时自动切换到Azure OpenAI
- 🚀 **高性能缓存**: 内存缓存机制，避免重复计算
- 📊 **Token统计**: 实时显示token消耗和成本分析

## 环境配置

### 必需的环境变量

在项目根目录创建 `.env` 文件，添加以下配置：

```bash
# GitHub Models (免费，优先使用)
GITHUB_TOKEN=ghp_your_github_token_here

# Azure OpenAI Service (备用)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
```

### 获取GitHub Token

1. 访问 [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. 点击 "Generate new token (classic)"
3. 选择适当的权限范围（通常只需要基础权限）
4. 复制生成的token到 `.env` 文件中

### 获取Azure OpenAI配置

1. 在Azure Portal中创建OpenAI资源
2. 获取API密钥和终结点URL
3. 部署gpt-4o-mini模型
4. 将配置信息添加到 `.env` 文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
python app.py
```

应用将在 `http://localhost:8000` 启动。

## 使用说明

1. **上传PDF**: 选择PDF文件并点击"上传并分析"
2. **查看信息**: 系统会自动提取文档信息、关键词和摘要
3. **浏览目录**: 点击目录项目查看章节内容
4. **获取摘要**: 自动生成章节摘要
5. **智能问答**: 在章节页面输入问题获取AI回答

## 技术特点

### 智能LLM切换
- 优先使用GitHub Models (免费，8000 token限制)
- 自动检测token数量，超限时切换到Azure OpenAI
- 失败时自动降级到备用服务

### 高效PDF处理
- 优先使用PyMuPDF提取PDF书签 (推荐，几乎无token消耗)
- 支持自动页面分割
- 失败时回退到LLM分析

### 智能缓存
- 内存缓存避免重复分析
- MD5键值确保缓存准确性
- 支持章节内容和分析结果缓存

## 成本优化

通过多层优化策略，大幅降低token消耗：

1. **PDF书签提取**: 相比LLM分析节省90%+ token
2. **GitHub Models优先**: 免费服务优先使用
3. **智能缓存**: 避免重复计算，节省80%+ token
4. **章节级处理**: 按需加载，避免全文分析

## 依赖库

- FastAPI: Web框架
- PyMuPDF: PDF处理
- OpenAI: LLM服务客户端
- tiktoken: Token计算
- Bootstrap: 前端UI框架

## 许可证

MIT License 