"""
PDF Reading Assistant - Semantic Kernel Version

基于 Semantic Kernel 框架的智能PDF文档阅读助手
主要改进：
1. 使用 Semantic Kernel 的 Agent 和 Plugin 架构
2. 模块化的 PDF 处理功能
3. 更好的可扩展性和维护性
4. 符合课程技术栈标准
"""

import os
import json
import logging
import hashlib
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
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import AuthorRole, ChatMessageContent

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

app = FastAPI(title="PDF Reading Assistant - Semantic Kernel")

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 内存缓存
chapter_content_cache = {}
pdf_info_cache = {}

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

# PDF处理插件类
class PDFProcessingPlugin:
    """
    PDF处理插件 - 使用 Semantic Kernel 的 Plugin 架构
    包含所有PDF相关的处理功能
    """
    
    def __init__(self):
        self.cache = chapter_content_cache
        self.pdf_cache = pdf_info_cache
    
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

# LLM服务类（保持与原版本兼容的接口）
class LLMService:
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
        
        # Azure OpenAI 备用服务
        # TODO: 添加 Azure OpenAI 服务配置
        
        # 添加PDF处理插件
        self.pdf_plugin = PDFProcessingPlugin()
        self.kernel.add_plugin(self.pdf_plugin, plugin_name="PDFProcessor")
        
        logger.info("LLM服务和插件初始化完成")
    
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
llm_service = LLMService()

# 创建专门的分析代理
async def get_document_analyzer():
    """获取文档分析代理"""
    return await llm_service.create_agent(
        name="DocumentAnalyzer",
        instructions="""你是一个专业的学术论文分析助手。
        你可以使用以下工具：
        - extract_pdf_metadata: 提取PDF元数据
        - extract_pdf_outline: 提取PDF目录结构
        - extract_pdf_text: 提取PDF文本内容
        
        请分析PDF文档并提供详细的结构化信息。
        """
    )

async def get_content_summarizer():
    """获取内容摘要代理"""
    return await llm_service.create_agent(
        name="ContentSummarizer",
        instructions="""你是一个专业的内容摘要专家。
        请为提供的章节内容生成高质量的摘要，包括：
        1. 章节的主要观点
        2. 关键发现或结论
        3. 重要方法或技术
        请用中文输出。
        """
    )

async def get_qa_specialist():
    """获取问答专家代理"""
    return await llm_service.create_agent(
        name="QASpecialist",
        instructions="""你是一个专业的学术论文问答专家。
        请基于提供的章节内容回答用户的问题。
        要求：
        1. 只使用章节中的信息来回答问题
        2. 如果问题超出章节范围，请说明
        3. 回答要准确、专业、清晰
        4. 使用中文回答
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
    """处理PDF上传 - 使用Semantic Kernel架构"""
    try:
        logger.info(f"开始处理文件上传: {file.filename}")
        
        # 保存上传的文件
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 直接使用插件功能进行分析（不使用Agent，避免复杂性）
        pdf_plugin = llm_service.pdf_plugin
        
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
        
        # 组装分析结果
        analysis = {
            "title": metadata.get("title") or f"PDF文档 - {file.filename}",
            "total_pages": outline_data["total_pages"],
            "chapters": outline_data["chapters"],
            "keywords": keywords,
            "subject": metadata.get("subject", ""),
            "extraction_method": outline_data["extraction_method"],
            "token_usage": {
                "input_tokens": 0,  # PDF处理不消耗LLM token
                "output_tokens": 0,
                "total_tokens": 0,
                "service": "semantic_kernel_plugin"
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
    """生成章节摘要 - 使用Semantic Kernel架构"""
    try:
        logger.info(f"开始处理章节摘要请求: {chapter_title}")
        
        # 获取内容摘要代理
        summarizer = await get_content_summarizer()
        thread = None  # 初始化为None，会在invoke_stream中创建
        
        # 提取章节文本
        pdf_plugin = llm_service.pdf_plugin
        chapter_text = pdf_plugin.extract_pdf_text(file_path, page_start, page_end)
        
        # 构建摘要请求
        summary_prompt = f"""请为论文的"{chapter_title}"章节生成详细摘要。

章节内容：
{chapter_text}"""
        
        # 使用正确的API调用方式
        summary_response = ""
        async for response in summarizer.invoke_stream(
            messages=summary_prompt,
            thread=thread
        ):
            if hasattr(response, 'content') and response.content:
                summary_response += str(response.content)
            thread = response.thread  # 更新thread
        
        # 计算token使用
        input_tokens = pdf_plugin.get_token_count(summary_prompt)
        output_tokens = pdf_plugin.get_token_count(summary_response)
        
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "service": "semantic_kernel_github"
        }
        
        return JSONResponse({
            "status": "success",
            "summary": summary_response,
            "token_usage": token_usage,
            "cached": False,
            "operation_type": "summarize"
        })
        
    except Exception as e:
        logger.error(f"生成摘要时出错: {str(e)}")
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
    """回答关于章节的问题 - 使用Semantic Kernel架构"""
    try:
        logger.info(f"开始处理章节问题: {chapter_title}")
        
        # 获取问答专家代理
        qa_agent = await get_qa_specialist()
        thread = None  # 初始化为None，会在invoke_stream中创建
        
        # 提取章节文本
        pdf_plugin = llm_service.pdf_plugin
        chapter_text = pdf_plugin.extract_pdf_text(file_path, page_start, page_end)
        
        # 构建问答请求
        qa_prompt = f"""基于论文的"{chapter_title}"章节内容回答问题。

章节内容：
{chapter_text}

问题：{question}"""
        
        # 使用正确的API调用方式
        answer_response = ""
        async for response in qa_agent.invoke_stream(
            messages=qa_prompt,
            thread=thread
        ):
            if hasattr(response, 'content') and response.content:
                answer_response += str(response.content)
            thread = response.thread  # 更新thread
        
        # 计算token使用
        input_tokens = pdf_plugin.get_token_count(qa_prompt)
        output_tokens = pdf_plugin.get_token_count(answer_response)
        
        token_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "service": "semantic_kernel_github"
        }
        
        return JSONResponse({
            "status": "success",
            "answer": answer_response,
            "token_usage": token_usage,
            "cached": False,
            "operation_type": "question"
        })
        
    except Exception as e:
        logger.error(f"回答问题时出错: {str(e)}")
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
        "cache_keys": {
            "chapter_content": list(chapter_content_cache.keys())[:10],
            "pdf_info": list(pdf_info_cache.keys())[:10]
        }
    })

@app.delete("/cache/clear")
async def clear_cache():
    """清空缓存"""
    global chapter_content_cache, pdf_info_cache
    chapter_content_cache.clear()
    pdf_info_cache.clear()
    return JSONResponse({
        "status": "success",
        "message": "缓存已清空"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 