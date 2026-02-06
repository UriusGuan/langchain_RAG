# main.py
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from rag_core import rag_system
import json

app = FastAPI(
    title="智能 RAG 问答系统",
    description="基于 LangChain 的检索增强生成问答系统",
    version="1.0.0"
)

# 设置模板目录
templates = Jinja2Templates(directory="templates")

# 挂载静态文件（如果需要）
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首页，显示 RAG 问答界面"""
    return templates.TemplateResponse(
        "rag.html",
        {"request": request}
    )


@app.post("/api/query")
async def query_rag(
        question: str = Form(..., description="用户问题"),
        include_context: bool = Form(False, description="是否包含检索上下文")
):
    """
    RAG 查询接口

    Args:
        question: 用户提出的问题
        include_context: 是否返回检索到的上下文信息
    """
    try:
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="问题不能为空")

        # 调用 RAG 系统
        result = rag_system.query(question, include_context)

        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": result["error"]}
            )

        if include_context:
            return JSONResponse({
                "status": "success",
                "data": {
                    "question": question,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "context_count": len(result["contexts"])
                }
            })
        else:
            return JSONResponse({
                "status": "success",
                "data": {
                    "question": question,
                    "answer": result
                }
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/query")
async def query_rag_get(
        question: str,
        include_context: bool = False
):
    """
    RAG 查询接口（GET 方式）
    """
    try:
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="问题不能为空")

        result = rag_system.query(question, include_context)

        if "error" in result:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": result["error"]}
            )

        if include_context:
            return JSONResponse({
                "status": "success",
                "data": {
                    "question": question,
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "context_count": len(result["contexts"])
                }
            })
        else:
            return JSONResponse({
                "status": "success",
                "data": {
                    "question": question,
                    "answer": result
                }
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/history")
async def get_conversation_history():
    """获取对话历史"""
    try:
        history = rag_system.get_conversation_history()
        return JSONResponse({
            "status": "success",
            "data": {
                "history": history,
                "count": len(history)
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.delete("/api/history")
async def clear_conversation_history():
    """清空对话历史"""
    try:
        result = rag_system.clear_conversation_history()
        return JSONResponse({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return JSONResponse({
        "status": "healthy",
        "service": "rag-qa-system",
        "vector_store": "loaded" if rag_system.vector_store else "not_loaded",
        "chain": "built" if rag_system.chain else "not_built"
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
