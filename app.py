"""
PDF Paper Analysis Web Application

A FastAPI-based web application that allows users to:
1. Upload PDF papers
2. View the paper's table of contents
3. Generate summaries for specific chapters
4. Ask questions about specific chapters

Optimized version with:
- Direct outline extraction using PyMuPDF
- Caching mechanism for chapter content
- Token consumption tracking
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import AsyncAzureOpenAI, AsyncOpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF
import pymupdf4llm  # For better text extraction
import tiktoken  # For token counting

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载上级目录的环境变量
load_dotenv()

app = FastAPI(title="PDF Paper Analysis")

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 设置静态文件和模板
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# LLM服务配置和客户端
class LLMService:
    def __init__(self):
        # GitHub Models 客户端 (免费，优先使用)
        self.github_client = AsyncOpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.getenv("GITHUB_TOKEN"),
        ) if os.getenv("GITHUB_TOKEN") else None
        
        # Azure OpenAI 客户端 (备用)
        self.azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        ) if os.getenv("AZURE_OPENAI_API_KEY") else None
        
        # GitHub Models 配置
        self.github_model = "gpt-4o-mini"  # GitHub Models 提供的免费模型
        self.github_token_limit = 8000  # GitHub Models 的token限制
        
        # Azure OpenAI 配置  
        self.azure_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        
        logger.info(f"LLM服务初始化 - GitHub Models: {'可用' if self.github_client else '不可用'}, Azure OpenAI: {'可用' if self.azure_client else '不可用'}")
    
    async def chat_completion(self, messages: List[Dict], max_tokens: int = None, response_format: Dict = None):
        """智能选择LLM服务进行对话完成"""
        
        # 计算输入token数
        input_text = "\n".join([msg.get("content", "") for msg in messages])
        input_tokens = get_token_count(input_text)
        
        # 尝试使用GitHub Models (如果token在限制内且客户端可用)
        if self.github_client and input_tokens < self.github_token_limit:
            try:
                logger.info(f"使用GitHub Models，输入token数: {input_tokens}")
                kwargs = {
                    "model": self.github_model,
                    "messages": messages
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if response_format:
                    kwargs["response_format"] = response_format
                    
                response = await self.github_client.chat.completions.create(**kwargs)
                logger.info("GitHub Models 调用成功")
                return response, "github"
                
            except Exception as e:
                logger.warning(f"GitHub Models 调用失败: {str(e)}, 切换到Azure OpenAI")
        
        # 回退到Azure OpenAI
        if self.azure_client:
            try:
                logger.info(f"使用Azure OpenAI，输入token数: {input_tokens}")
                kwargs = {
                    "model": self.azure_model,
                    "messages": messages
                }
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if response_format:
                    kwargs["response_format"] = response_format
                    
                response = await self.azure_client.chat.completions.create(**kwargs)
                logger.info("Azure OpenAI 调用成功")
                return response, "azure"
                
            except Exception as e:
                logger.error(f"Azure OpenAI 调用也失败: {str(e)}")
                raise e
        
        raise HTTPException(status_code=500, detail="所有LLM服务都不可用")

# 初始化LLM服务
llm_service = LLMService()

# 内存缓存
chapter_content_cache = {}
pdf_info_cache = {}

# Token计算器
def get_token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """计算文本的token数量"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # 如果模型不支持，使用近似计算
        return len(text.split()) * 1.3  # 粗略估计

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

class TokenUsage(BaseModel):
    """Token使用统计模型"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    service: Optional[str] = None  # 添加服务提供商信息

class AnalysisResponse(BaseModel):
    """分析响应模型"""
    status: str
    file_path: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    token_usage: Optional[TokenUsage] = None
    operation_type: str = "analysis"
    cached: bool = False

def extract_outline_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """从PDF中提取目录大纲"""
    try:
        doc = fitz.open(pdf_path)
        outline = doc.get_toc()  # 获取目录
        total_pages = len(doc)
        
        chapters = []
        if outline:
            # 如果PDF有书签目录
            for item in outline:
                level, title, page = item
                chapters.append({
                    "title": title.strip(),
                    "page_start": page,
                    "page_end": None,  # 将在后面计算
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
            pages_per_chapter = max(1, total_pages // 10)  # 假设分成10章
            for i in range(0, total_pages, pages_per_chapter):
                start_page = i + 1
                end_page = min(i + pages_per_chapter, total_pages)
                chapters.append({
                    "title": f"第 {len(chapters) + 1} 部分 (第{start_page}-{end_page}页)",
                    "page_start": start_page,
                    "page_end": end_page,
                    "level": 1
                })
        
        doc.close()
        return chapters, total_pages
        
    except Exception as e:
        logger.error(f"提取PDF大纲时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提取PDF大纲时出错: {str(e)}")

def get_cache_key(pdf_path: str, operation: str, **kwargs) -> str:
    """生成缓存键"""
    key_parts = [pdf_path, operation]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def extract_text_from_pdf(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None, use_cache: bool = True) -> tuple[str, bool]:
    """从PDF文件中提取文本内容，可以指定页码范围，返回(文本, 是否来自缓存)"""
    try:
        # 检查缓存
        cache_key = get_cache_key(pdf_path, "extract_text", start_page=start_page, end_page=end_page)
        if use_cache and cache_key in chapter_content_cache:
            logger.info(f"从缓存中获取文本内容: {cache_key}")
            return chapter_content_cache[cache_key], True
        
        # 如果没有指定页码范围，使用 PyMuPDF4LLM 提取全文
        if start_page is None and end_page is None:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            if use_cache:
                chapter_content_cache[cache_key] = md_text
            return md_text, False
        
        # 提取指定页面的文本
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # 确定页码范围
        start = start_page - 1 if start_page else 0
        end = end_page if end_page else total_pages
        
        # 提取指定页面的文本
        text = ""
        for page_num in range(start, end):
            if page_num < total_pages:
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
        
        doc.close()
        
        # 缓存结果
        if use_cache:
            chapter_content_cache[cache_key] = text
            
        return text, False
        
    except Exception as e:
        logger.error(f"提取PDF文本时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提取PDF文本时出错: {str(e)}")

async def analyze_pdf_with_llm(pdf_path: str) -> Dict[str, Any]:
    """优化的PDF分析函数，优先使用PDF书签目录"""
    try:
        logger.info(f"开始分析PDF文件: {pdf_path}")
        
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 检查缓存
        cache_key = get_cache_key(pdf_path, "analyze_pdf")
        if cache_key in pdf_info_cache:
            logger.info("从缓存中获取PDF分析结果")
            return pdf_info_cache[cache_key]
        
        # 首先尝试直接从PDF提取目录结构
        try:
            chapters, total_pages = extract_outline_from_pdf(pdf_path)
            
            # 从PDF获取标题和关键词
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            title = metadata.get("title", "")
            keywords = metadata.get("keywords", "")
            subject = metadata.get("subject", "")  # 摘要/主题描述
            
            logger.info(f"PDF metadata - title: {title}, keywords: {keywords}, subject: {subject}")
            
            if not title and len(doc) > 0:
                # 如果没有元数据标题，尝试从第一页提取
                first_page_text = doc[0].get_text()[:500]  # 只取前500字符
                # 使用LLM提取标题
                title_response = await llm_service.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "请从文档的开头部分提取论文标题，只返回标题文本，不要其他内容。"
                        },
                        {
                            "role": "user",
                            "content": f"文档开头内容：\n{first_page_text}"
                        }
                    ],
                    max_tokens=100
                )
                title = title_response[0].choices[0].message.content.strip()
                used_service = title_response[1]
            
            doc.close()
            
            if not title:
                title = f"PDF文档 - {os.path.basename(pdf_path)}"
            
            # 处理关键词 - 如果为空，不显示
            processed_keywords = []
            if keywords and keywords.strip():
                # 分割关键词，支持多种分隔符
                import re
                keywords_list = re.split(r'[,;，；\n\r]+', keywords.strip())
                processed_keywords = [kw.strip() for kw in keywords_list if kw.strip()]
            
            result = {
                "title": title,
                "total_pages": total_pages,
                "chapters": chapters,
                "keywords": processed_keywords,  # 添加关键词字段
                "subject": subject.strip() if subject else "",  # 添加摘要字段
                "extraction_method": "outline" if len(chapters) > 1 else "automatic_split",
                "token_usage": {
                    "input_tokens": get_token_count(first_page_text) if 'first_page_text' in locals() else 0,
                    "output_tokens": get_token_count(title) if title else 0,
                    "total_tokens": get_token_count(first_page_text) + get_token_count(title) if 'first_page_text' in locals() and title else 0,
                    "service": used_service if 'used_service' in locals() else "metadata"
                }
            }
            
            # 缓存结果
            pdf_info_cache[cache_key] = result
            logger.info(f"PDF分析完成，使用方法: {result['extraction_method']}, 关键词数量: {len(processed_keywords)}")
            return result
            
        except Exception as outline_error:
            logger.warning(f"无法提取PDF大纲，回退到LLM分析: {str(outline_error)}")
            
            # 回退到原始方法：使用LLM分析全文
            pdf_text = extract_text_from_pdf(pdf_path)[0]
            logger.info("PDF文本提取完成，准备LLM分析")
            
            input_tokens = get_token_count(pdf_text)
            logger.info(f"输入token数: {input_tokens}")
            
            # 使用OpenAI API分析PDF
            response = await llm_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": """你是一个专业的学术论文分析助手。
                        请分析提供的PDF论文，提取以下信息并以JSON格式返回：
                        {
                            "title": "论文标题",
                            "total_pages": 总页数,
                            "keywords": ["关键词1", "关键词2", "关键词3"],
                            "subject": "文档摘要或主题描述",
                            "chapters": [
                                {
                                    "title": "章节标题",
                                    "page_start": 起始页码,
                                    "page_end": 结束页码,
                                    "level": 章节层级
                                }
                            ]
                        }
                        请确保返回的是有效的JSON格式。如果无法提取关键词，请返回空数组。如果无法提取摘要，请返回空字符串。"""
                    },
                    {
                        "role": "user",
                        "content": f"请分析这个PDF论文的目录结构和基本信息。\n\n论文内容：\n{pdf_text}"
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response[0].choices[0].message.content)
            output_tokens = get_token_count(response[0].choices[0].message.content)
            used_service = response[1]
            
            # 添加token使用统计
            result["extraction_method"] = "llm_analysis"
            result["token_usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "service": used_service
            }
            
            logger.info(f"LLM分析完成，消耗token: {result['token_usage']['total_tokens']}")
            
            # 验证返回的数据格式
            if not isinstance(result, dict):
                raise ValueError("返回的数据不是有效的JSON对象")
            
            if "title" not in result or "total_pages" not in result or "chapters" not in result:
                raise ValueError("返回的数据缺少必要的字段")
            
            if not isinstance(result["chapters"], list):
                raise ValueError("chapters字段不是有效的数组")
            
            # 确保关键词字段存在
            if "keywords" not in result:
                result["keywords"] = []
            
            # 确保摘要字段存在
            if "subject" not in result:
                result["subject"] = ""
            
            # 确保每个章节都有必要的字段
            for chapter in result["chapters"]:
                if not all(key in chapter for key in ["title", "page_start"]):
                    raise ValueError("章节数据缺少必要的字段")
            
            # 缓存结果
            pdf_info_cache[cache_key] = result
            return result
        
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"返回的数据格式错误: {str(e)}")
    except ValueError as e:
        logger.error(f"数据验证错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"分析PDF时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析PDF时出错: {str(e)}")

async def summarize_chapter(pdf_path: str, chapter: ChapterInfo) -> tuple[str, TokenUsage, bool]:
    """生成特定章节的摘要，返回(摘要, token统计, 是否来自缓存)"""
    try:
        logger.info(f"开始生成章节摘要: {chapter.title}")
        
        # 检查缓存
        cache_key = get_cache_key(pdf_path, "summarize", title=chapter.title, start=chapter.page_start, end=chapter.page_end)
        if cache_key in chapter_content_cache:
            cached_result = chapter_content_cache[cache_key]
            if isinstance(cached_result, dict) and "summary" in cached_result:
                logger.info("从缓存中获取章节摘要")
                # 修复：从缓存获取时，token_usage 已经是字典格式
                cached_token_usage = cached_result["token_usage"]
                if isinstance(cached_token_usage, dict):
                    token_usage = TokenUsage(**cached_token_usage)
                else:
                    token_usage = cached_token_usage
                return cached_result["summary"], token_usage, True
        
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 提取PDF文本内容
        pdf_text, text_cached = extract_text_from_pdf(pdf_path, chapter.page_start, chapter.page_end)
        logger.info(f"PDF文本提取完成，来自缓存: {text_cached}")
        
        input_tokens = get_token_count(pdf_text)
        logger.info(f"章节文本token数: {input_tokens}")
        
        # 使用OpenAI API生成章节摘要
        response = await llm_service.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""你是一个专业的学术论文分析助手。
                    请为论文的"{chapter.title}"章节生成一个详细的摘要。
                    摘要应该包含：
                    1. 章节的主要观点
                    2. 关键发现或结论
                    3. 重要方法或技术
                    请用中文输出。"""
                },
                {
                    "role": "user",
                    "content": f"请分析这个PDF论文的第{chapter.page_start}到{chapter.page_end}页的内容。\n\n论文内容：\n{pdf_text}"
                }
            ]
        )
        
        summary = response[0].choices[0].message.content
        output_tokens = get_token_count(summary)
        used_service = response[1]
        
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            service=used_service
        )
        
        logger.info(f"章节摘要生成完成，消耗token: {token_usage.total_tokens}")
        
        # 缓存结果
        cache_result = {
            "summary": summary,
            "token_usage": token_usage.dict()
        }
        chapter_content_cache[cache_key] = cache_result
        
        return summary, token_usage, False
        
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"生成摘要时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成摘要时出错: {str(e)}")

async def answer_chapter_question(pdf_path: str, chapter: ChapterInfo, question: str) -> tuple[str, TokenUsage, bool]:
    """回答关于特定章节的问题，返回(回答, token统计, 是否来自缓存)"""
    try:
        logger.info(f"开始回答章节问题: {chapter.title}")
        
        # 检查缓存
        cache_key = get_cache_key(pdf_path, "question", title=chapter.title, start=chapter.page_start, end=chapter.page_end, question=question)
        if cache_key in chapter_content_cache:
            cached_result = chapter_content_cache[cache_key]
            if isinstance(cached_result, dict) and "answer" in cached_result:
                logger.info("从缓存中获取问题回答")
                # 修复：从缓存获取时，token_usage 已经是字典格式
                cached_token_usage = cached_result["token_usage"]
                if isinstance(cached_token_usage, dict):
                    token_usage = TokenUsage(**cached_token_usage)
                else:
                    token_usage = cached_token_usage
                return cached_result["answer"], token_usage, True
        
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 提取PDF文本内容
        pdf_text, text_cached = extract_text_from_pdf(pdf_path, chapter.page_start, chapter.page_end)
        logger.info(f"PDF文本提取完成，来自缓存: {text_cached}")
        
        prompt = f"章节内容：\n{pdf_text}\n\n问题：{question}"
        input_tokens = get_token_count(prompt)
        logger.info(f"问答输入token数: {input_tokens}")
        
        # 使用OpenAI API回答问题
        response = await llm_service.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"""你是一个专业的学术论文分析助手。
                    请基于论文的"{chapter.title}"章节内容回答用户的问题。
                    要求：
                    1. 只使用章节中的信息来回答问题
                    2. 如果问题超出章节范围，请说明
                    3. 回答要准确、专业、清晰
                    4. 使用中文回答"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        answer = response[0].choices[0].message.content
        output_tokens = get_token_count(answer)
        used_service = response[1]
        
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            service=used_service
        )
        
        logger.info(f"问题回答完成，消耗token: {token_usage.total_tokens}")
        
        # 缓存结果
        cache_result = {
            "answer": answer,
            "token_usage": token_usage.dict()
        }
        chapter_content_cache[cache_key] = cache_result
        
        return answer, token_usage, False
        
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"回答问题时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"回答问题时出错: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """渲染主页"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """处理PDF上传"""
    try:
        logger.info(f"开始处理文件上传: {file.filename}")
        
        # 保存上传的文件
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 分析PDF
        analysis = await analyze_pdf_with_llm(str(file_path))
        
        return JSONResponse({
            "status": "success",
            "file_path": str(file_path),
            "analysis": analysis,
            "extraction_method": analysis.get("extraction_method", "unknown"),
            "token_usage": analysis.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        })
        
    except HTTPException as e:
        logger.error(f"处理上传时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e.detail)
        }, status_code=e.status_code)
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
    """生成章节摘要"""
    try:
        logger.info(f"开始处理章节摘要请求: {chapter_title}")
        
        chapter = ChapterInfo(
            title=chapter_title,
            page_start=page_start,
            page_end=page_end
        )
        
        summary, token_usage, cached = await summarize_chapter(file_path, chapter)
        
        return JSONResponse({
            "status": "success",
            "summary": summary,
            "token_usage": token_usage.dict(),
            "cached": cached,
            "operation_type": "summarize"
        })
        
    except HTTPException as e:
        logger.error(f"生成摘要时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e.detail)
        }, status_code=e.status_code)
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
    """回答关于章节的问题"""
    try:
        logger.info(f"开始处理章节问题: {chapter_title}")
        
        chapter = ChapterInfo(
            title=chapter_title,
            page_start=page_start,
            page_end=page_end
        )
        
        answer, token_usage, cached = await answer_chapter_question(file_path, chapter, question)
        
        return JSONResponse({
            "status": "success",
            "answer": answer,
            "token_usage": token_usage.dict(),
            "cached": cached,
            "operation_type": "question"
        })
        
    except HTTPException as e:
        logger.error(f"回答问题时出错: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e.detail)
        }, status_code=e.status_code)
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
            "chapter_content": list(chapter_content_cache.keys())[:10],  # 只显示前10个
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