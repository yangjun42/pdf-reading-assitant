"""
PDF Reading Assistant - RAG Enhanced Version

基于 Semantic Kernel + Azure AI Search 的智能PDF文档阅读助手
主要改进：
1. 集成 Azure AI Search 进行语义检索
2. 解决章节摘要和问答的上下文割裂问题
3. 实现文档全文向量化和索引
4. 提供更准确和全面的回答
"""

import os
import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Annotated
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Semantic Kernel imports
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import AuthorRole, ChatMessageContent

# Azure AI Search imports
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType, 
    SearchableField, VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile, SemanticConfiguration, SemanticField,
    SemanticPrioritizedFields, SemanticSearch
)
from azure.core.credentials import AzureKeyCredential

# Azure AI imports
from openai import AsyncOpenAI

# PDF processing
import fitz  # PyMuPDF
import pymupdf4llm  # For better text extraction
import tiktoken  # For token counting

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

app = FastAPI(title="PDF Reading Assistant - RAG Enhanced")

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 内存缓存
chapter_content_cache = {}
pdf_info_cache = {}
search_index_cache = {}  # 缓存已创建的搜索索引

class TokenUsage(BaseModel):
    """Token使用统计模型"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    service: Optional[str] = None

class ChapterInfo(BaseModel):
    """章节信息模型"""
    title: str
    page_start: int
    page_end: Optional[int] = None
    level: int = 1
    summary: Optional[str] = None

class PaperAnalysis(BaseModel):
    """论文分析结果模型"""
    title: str
    chapters: List[ChapterInfo]
    total_pages: int
    keywords: List[str] = []
    subject: str = ""
    extraction_method: str = "unknown"
    token_usage: Optional[TokenUsage] = None

# Azure AI Search 服务配置
class AzureSearchService:
    """Azure AI Search 服务管理类"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        
        if not self.endpoint or not self.api_key:
            logger.warning("Azure Search credentials not found. RAG features will be disabled.")
            self.enabled = False
            return
        
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.enabled = True
        logger.info("Azure Search service initialized successfully")
    
    def get_index_name(self, pdf_filename: str) -> str:
        """为PDF文件生成唯一的索引名称"""
        # 清理文件名，只保留字母数字字符
        clean_name = ''.join(c for c in pdf_filename if c.isalnum() or c in ['-', '_'])
        return f"pdf-index-{clean_name.lower()}"
    
    def sanitize_document_id(self, filename: str, chunk_index: int) -> str:
        """生成Azure Search兼容的文档ID
        只包含字母、数字、下划线、短横线和等号
        """
        # 移除文件扩展名和特殊字符
        base_name = os.path.splitext(filename)[0]
        # 只保留ASCII字母数字字符、下划线和短横线
        clean_name = ''.join(c for c in base_name if c.isascii() and (c.isalnum() or c in ['-', '_']))
        return f"{clean_name}-chunk-{chunk_index}"
    
    def create_document_index(self, index_name: str) -> SearchIndex:
        """创建文档搜索索引"""
        try:
            # 定义索引字段
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
                SimpleField(name="pdf_filename", type=SearchFieldDataType.String, filterable=True),
            ]
            
            # 创建索引
            index = SearchIndex(
                name=index_name,
                fields=fields,
                semantic_search=SemanticSearch(
                    configurations=[
                        SemanticConfiguration(
                            name="semantic-config",
                            prioritized_fields=SemanticPrioritizedFields(
                                content_fields=[SemanticField(field_name="content")],
                                keywords_fields=[SemanticField(field_name="title")]
                            )
                        )
                    ]
                )
            )
            
            # 创建或更新索引
            result = self.index_client.create_or_update_index(index)
            logger.info(f"Created/updated search index: {index_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create search index: {str(e)}")
            raise
    
    def get_search_client(self, index_name: str) -> SearchClient:
        """获取搜索客户端"""
        return SearchClient(
            endpoint=self.endpoint,
            index_name=index_name,
            credential=self.credential
        )
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """将文本分割成重叠的块"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # 尝试在句号处截断，避免截断句子
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.7:  # 如果句号位置合理
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    async def index_document(self, pdf_path: str, pdf_content: str, pdf_filename: str) -> str:
        """将PDF文档索引到Azure Search"""
        try:
            index_name = self.get_index_name(pdf_filename)
            
            # 检查缓存
            if index_name in search_index_cache:
                logger.info(f"Using cached search index: {index_name}")
                return index_name
            
            # 创建索引
            self.create_document_index(index_name)
            
            # 获取搜索客户端
            search_client = self.get_search_client(index_name)
            
            # 分割文档为块
            chunks = self.chunk_text(pdf_content)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # 准备文档数据
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {
                    "id": self.sanitize_document_id(pdf_filename, i),
                    "content": chunk,
                    "title": f"Chunk {i+1}",
                    "page_number": i // 2 + 1,  # 粗略估计页码
                    "chunk_index": i,
                    "pdf_filename": pdf_filename
                }
                documents.append(doc)
            
            # 批量上传文档
            if documents:
                result = search_client.upload_documents(documents)
                logger.info(f"Indexed {len(documents)} chunks successfully")
                
                # 缓存索引名
                search_index_cache[index_name] = True
            
            return index_name
            
        except Exception as e:
            logger.error(f"Failed to index document: {str(e)}")
            raise
    
    async def search_relevant_content(
        self, 
        index_name: str, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """搜索相关内容"""
        try:
            search_client = self.get_search_client(index_name)
            
            # 执行语义搜索
            results = search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name="semantic-config",
                top=top_k,
                include_total_count=True
            )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result["content"],
                    "title": result["title"],
                    "page_number": result.get("page_number", 0),
                    "score": result.get("@search.score", 0),
                    "chunk_index": result.get("chunk_index", 0)
                })
            
            logger.info(f"Found {len(formatted_results)} relevant chunks for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            return []

# RAG增强的PDF处理插件
class RAGEnhancedPDFPlugin:
    """
    RAG增强的PDF处理插件
    集成Azure AI Search进行语义检索
    """
    
    def __init__(self, search_service: AzureSearchService):
        self.cache = chapter_content_cache
        self.pdf_cache = pdf_info_cache
        self.search_service = search_service
    
    def get_cache_key(self, pdf_path: str, operation: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [pdf_path, operation]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_token_count(self, text: str, model: str = "gpt-4o-mini") -> int:
        """计算文本的token数量"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)
    
    @kernel_function(
        description="从PDF文件中提取基本信息，包括标题、页数、关键词等",
        name="extract_pdf_metadata"
    )
    def extract_pdf_metadata(
        self, 
        pdf_path: Annotated[str, "PDF文件路径"]
    ) -> Annotated[str, "PDF元数据信息的JSON字符串"]:
        """提取PDF基本元数据"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            total_pages = len(doc)
            
            result = {
                "title": metadata.get("title", ""),
                "total_pages": total_pages,
                "keywords": metadata.get("keywords", ""),
                "subject": metadata.get("subject", ""),
                "author": metadata.get("author", ""),
                "creator": metadata.get("creator", "")
            }
            
            doc.close()
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"提取PDF元数据时出错: {str(e)}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    @kernel_function(
        description="从PDF文件中提取目录结构和章节信息",
        name="extract_pdf_outline"
    )
    def extract_pdf_outline(
        self, 
        pdf_path: Annotated[str, "PDF文件路径"]
    ) -> Annotated[str, "PDF目录结构的JSON字符串"]:
        """提取PDF目录大纲"""
        try:
            doc = fitz.open(pdf_path)
            outline = doc.get_toc()
            total_pages = len(doc)
            
            chapters = []
            if outline:
                # 如果PDF有书签目录
                for item in outline:
                    level, title, page = item
                    chapters.append({
                        "title": title.strip(),
                        "page_start": page,
                        "page_end": None,
                        "level": level
                    })
                
                # 计算每个章节的结束页码
                for i in range(len(chapters)):
                    if i < len(chapters) - 1:
                        chapters[i]["page_end"] = chapters[i + 1]["page_start"] - 1
                    else:
                        chapters[i]["page_end"] = total_pages
            else:
                # 如果没有书签目录，使用简单的页面分割
                pages_per_chapter = max(1, total_pages // 10)
                for i in range(0, total_pages, pages_per_chapter):
                    start_page = i + 1
                    end_page = min(i + pages_per_chapter, total_pages)
                    chapters.append({
                        "title": f"第 {len(chapters) + 1} 部分 (第{start_page}-{end_page}页)",
                        "page_start": start_page,
                        "page_end": end_page,
                        "level": 1
                    })
            
            result = {
                "chapters": chapters,
                "total_pages": total_pages,
                "extraction_method": "outline" if outline else "automatic_split"
            }
            
            doc.close()
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"提取PDF大纲时出错: {str(e)}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
    
    @kernel_function(
        description="从PDF文件的指定页面范围提取文本内容",
        name="extract_pdf_text"
    )
    def extract_pdf_text(
        self, 
        pdf_path: Annotated[str, "PDF文件路径"],
        start_page: Annotated[int, "起始页码"] = 1,
        end_page: Annotated[int, "结束页码"] = -1
    ) -> Annotated[str, "提取的文本内容"]:
        """从PDF文件中提取文本内容"""
        try:
            # 检查缓存
            cache_key = self.get_cache_key(pdf_path, "extract_text", start_page=start_page, end_page=end_page)
            if cache_key in self.cache:
                logger.info(f"从缓存中获取文本内容: {cache_key}")
                return self.cache[cache_key]
            
            # 如果没有指定页码范围，使用 PyMuPDF4LLM 提取全文
            if start_page == 1 and end_page == -1:
                md_text = pymupdf4llm.to_markdown(pdf_path)
                self.cache[cache_key] = md_text
                return md_text
            
            # 提取指定页面的文本
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # 确定页码范围
            start = start_page - 1 if start_page > 0 else 0
            end = end_page if end_page > 0 else total_pages
            
            # 提取指定页面的文本
            text = ""
            for page_num in range(start, min(end, total_pages)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            
            # 缓存结果
            self.cache[cache_key] = text
            return text
            
        except Exception as e:
            logger.error(f"提取PDF文本时出错: {str(e)}")
            return f"错误: {str(e)}"
    
    async def search_and_enhance_context(
        self, 
        pdf_filename: str, 
        chapter_content: str, 
        query: str
    ) -> str:
        """搜索相关内容并增强上下文"""
        if not self.search_service.enabled:
            logger.info("Azure Search not enabled, returning original content")
            return chapter_content
        
        try:
            index_name = self.search_service.get_index_name(pdf_filename)
            
            # 搜索相关内容
            relevant_chunks = await self.search_service.search_relevant_content(
                index_name, query, top_k=3
            )
            
            if not relevant_chunks:
                return chapter_content
            
            # 构建增强的上下文
            enhanced_context = f"当前章节内容：\n{chapter_content}\n\n"
            enhanced_context += "相关参考内容：\n"
            
            for i, chunk in enumerate(relevant_chunks, 1):
                enhanced_context += f"\n参考片段 {i} (页面 {chunk['page_number']})：\n"
                enhanced_context += f"{chunk['content']}\n"
            
            logger.info(f"Enhanced context with {len(relevant_chunks)} relevant chunks")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Failed to enhance context: {str(e)}")
            return chapter_content

# LLM服务类（增强版）
class RAGEnhancedLLMService:
    def __init__(self):
        self.kernel = Kernel()
        
        # GitHub Models 客户端
        self.github_client = AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("GITHUB_TOKEN"),
        ) if os.getenv("GITHUB_TOKEN") else None
        
        # 创建聊天完成服务
        if self.github_client:
            github_service = OpenAIChatCompletion(
                ai_model_id="gpt-4o-mini",
                async_client=self.github_client,
                service_id="github"
            )
            self.kernel.add_service(github_service)
        
        # 初始化搜索服务
        self.search_service = AzureSearchService()
        
        # 添加RAG增强的PDF处理插件
        self.pdf_plugin = RAGEnhancedPDFPlugin(self.search_service)
        self.kernel.add_plugin(self.pdf_plugin, plugin_name="PDFProcessor")
        
        logger.info("RAG增强的LLM服务和插件初始化完成")
    
    async def create_agent(self, name: str, instructions: str) -> ChatCompletionAgent:
        """创建聊天完成代理"""
        agent = ChatCompletionAgent(
            service=self.kernel.get_service(type=OpenAIChatCompletion),
            name=name,
            instructions=instructions,
            kernel=self.kernel
        )
        return agent

# 初始化服务
llm_service = RAGEnhancedLLMService()

# 创建专门的分析代理
async def get_rag_content_summarizer():
    """获取RAG增强的内容摘要代理"""
    return await llm_service.create_agent(
        name="RAGContentSummarizer",
        instructions="""你是一个专业的内容摘要专家，具备RAG增强的分析能力。
        请为提供的章节内容生成高质量的摘要，包括：
        1. 章节的主要观点
        2. 关键发现或结论
        3. 重要方法或技术
        4. 如果提供了相关参考内容，请结合参考内容进行分析
        5. 指出章节内容与文档其他部分的关联
        请用中文输出。
        """
    )

async def get_rag_qa_specialist():
    """获取RAG增强的问答专家代理"""
    return await llm_service.create_agent(
        name="RAGQASpecialist",
        instructions="""你是一个专业的学术论文问答专家，具备RAG增强的分析能力。
        请基于提供的章节内容和相关参考内容回答用户的问题。
        要求：
        1. 优先使用当前章节的信息回答问题
        2. 如果章节内容不足，参考相关内容进行补充
        3. 明确指出信息来源（当前章节 vs 参考内容）
        4. 如果问题超出提供的内容范围，请说明
        5. 回答要准确、专业、清晰
        请用中文回答。
        """
    )

# API 路由
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """处理PDF上传 - RAG增强版"""
    try:
        logger.info(f"开始处理文件上传: {file.filename}")
        
        # 保存上传的文件
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 提取PDF全文内容
        pdf_plugin = llm_service.pdf_plugin
        full_text = pdf_plugin.extract_pdf_text(str(file_path))
        
        # 创建搜索索引（如果启用了Azure Search）
        index_name = None
        if llm_service.search_service.enabled:
            try:
                index_name = await llm_service.search_service.index_document(
                    str(file_path), full_text, file.filename
                )
                logger.info(f"Document indexed successfully: {index_name}")
            except Exception as e:
                logger.warning(f"Failed to index document, continuing without RAG: {str(e)}")
        
        # 提取元数据
        metadata_str = pdf_plugin.extract_pdf_metadata(str(file_path))
        metadata = json.loads(metadata_str)
        
        # 提取目录结构
        outline_str = pdf_plugin.extract_pdf_outline(str(file_path))
        outline_data = json.loads(outline_str)
        
        # 处理关键词
        keywords = []
        if metadata.get("keywords"):
            import re
            keywords_list = re.split(r'[,;，；\n\r]+', metadata["keywords"].strip())
            keywords = [kw.strip() for kw in keywords_list if kw.strip()]
        
        # 确定提取方法
        extraction_method = outline_data["extraction_method"]
        if index_name:
            extraction_method += "_with_rag"
        
        # 组装分析结果
        analysis = {
            "title": metadata.get("title") or f"PDF文档 - {file.filename}",
            "total_pages": outline_data["total_pages"],
            "chapters": outline_data["chapters"],
            "keywords": keywords,
            "subject": metadata.get("subject", ""),
            "extraction_method": extraction_method,
            "rag_enabled": llm_service.search_service.enabled and index_name is not None,
            "token_usage": {
                "input_tokens": 0,  # PDF处理不消耗LLM token
                "output_tokens": 0,
                "total_tokens": 0,
                "service": "rag_enhanced_plugin"
            }
        }
        
        return JSONResponse({
            "status": "success",
            "file_path": str(file_path),
            "analysis": analysis,
            "extraction_method": analysis["extraction_method"],
            "token_usage": analysis["token_usage"]
        })
        
    except Exception as e:
        logger.error(f"处理上传时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/summarize")
async def summarize_chapter_endpoint(
    file_path: str = Form(...),
    chapter_title: str = Form(...),
    page_start: int = Form(...),
    page_end: int = Form(...)
):
    """生成章节摘要 - RAG增强版"""
    try:
        logger.info(f"开始处理RAG增强的章节摘要请求: {chapter_title}")
        
        # 获取RAG增强的摘要代理
        summarizer = await get_rag_content_summarizer()
        thread = None
        
        # 提取章节文本
        pdf_plugin = llm_service.pdf_plugin
        chapter_text = pdf_plugin.extract_pdf_text(file_path, page_start, page_end)
        
        # RAG增强：搜索相关内容
        pdf_filename = Path(file_path).name
        enhanced_context = await pdf_plugin.search_and_enhance_context(
            pdf_filename, chapter_text, f"{chapter_title} 摘要 主要内容"
        )
        
        # 构建增强的摘要请求
        summary_prompt = f"""请为论文的"{chapter_title}"章节生成详细摘要。

{enhanced_context}"""
        
        # 使用正确的API调用方式
        summary_response = ""
        async for response in summarizer.invoke_stream(
            messages=summary_prompt,
            thread=thread
        ):
            if hasattr(response, 'content') and response.content:
                summary_response += str(response.content)
            thread = response.thread
        
        # 计算token使用
        input_tokens = pdf_plugin.get_token_count(summary_prompt)
        output_tokens = pdf_plugin.get_token_count(summary_response)
        
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "service": "rag_enhanced_github"
        }
        
        return JSONResponse({
            "status": "success",
            "summary": summary_response,
            "token_usage": token_usage,
            "cached": False,
            "operation_type": "rag_summarize",
            "rag_enhanced": llm_service.search_service.enabled
        })
        
    except Exception as e:
        logger.error(f"生成RAG增强摘要时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/ask")
async def ask_chapter_question(
    file_path: str = Form(...),
    chapter_title: str = Form(...),
    page_start: int = Form(...),
    page_end: int = Form(...),
    question: str = Form(...)
):
    """回答关于章节的问题 - RAG增强版"""
    try:
        logger.info(f"开始处理RAG增强的章节问题: {chapter_title}")
        
        # 获取RAG增强的问答代理
        qa_agent = await get_rag_qa_specialist()
        thread = None
        
        # 提取章节文本
        pdf_plugin = llm_service.pdf_plugin
        chapter_text = pdf_plugin.extract_pdf_text(file_path, page_start, page_end)
        
        # RAG增强：基于问题搜索相关内容
        pdf_filename = Path(file_path).name
        enhanced_context = await pdf_plugin.search_and_enhance_context(
            pdf_filename, chapter_text, question
        )
        
        # 构建增强的问答请求
        qa_prompt = f"""基于论文的"{chapter_title}"章节内容和相关参考内容回答问题。

{enhanced_context}

问题：{question}"""
        
        # 使用正确的API调用方式
        answer_response = ""
        async for response in qa_agent.invoke_stream(
            messages=qa_prompt,
            thread=thread
        ):
            if hasattr(response, 'content') and response.content:
                answer_response += str(response.content)
            thread = response.thread
        
        # 计算token使用
        input_tokens = pdf_plugin.get_token_count(qa_prompt)
        output_tokens = pdf_plugin.get_token_count(answer_response)
        
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "service": "rag_enhanced_github"
        }
        
        return JSONResponse({
            "status": "success",
            "answer": answer_response,
            "token_usage": token_usage,
            "cached": False,
            "operation_type": "rag_question",
            "rag_enhanced": llm_service.search_service.enabled
        })
        
    except Exception as e:
        logger.error(f"回答RAG增强问题时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/cache/stats")
async def get_cache_stats():
    """获取缓存统计信息"""
    return JSONResponse({
        "chapter_content_cache_size": len(chapter_content_cache),
        "pdf_info_cache_size": len(pdf_info_cache),
        "search_index_cache_size": len(search_index_cache),
        "rag_enabled": llm_service.search_service.enabled,
        "cache_keys": {
            "chapter_content": list(chapter_content_cache.keys())[:10],
            "pdf_info": list(pdf_info_cache.keys())[:10],
            "search_indexes": list(search_index_cache.keys())
        }
    })

@app.delete("/cache/clear")
async def clear_cache():
    """清空缓存"""
    global chapter_content_cache, pdf_info_cache, search_index_cache
    chapter_content_cache.clear()
    pdf_info_cache.clear()
    search_index_cache.clear()
    return JSONResponse({
        "status": "success",
        "message": "所有缓存已清空（包括RAG索引缓存）"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # 使用不同端口避免冲突 