from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llm import planner

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# in memory db
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    prompt: str


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        # initialize chat history for each session
        sessions[req.session_id] = []
    history = sessions[req.session_id]
    history.append({"role": "user", "content": req.prompt})

    try:
        response = planner.chat.completions.create(
            model="deepseek-ai/deepseek-v4-flash",
            messages=history,
            max_tokens=16384,
            stream=False,
            extra_body={
                    "temperature": 0.7,
                    "top_p": 0.9, 
                    "chat_template_kwargs": {"thinking": True, "reasoning_effort": "medium"}
                }
            )
        assistant_message = response.choices[0].message
        reasoning = getattr(assistant_message, "reasoning", None) or getattr(assistant_message, "reasoning_content", None)
        content = assistant_message.content

        history.append({
            "role": "assistant",
            "content": content
        })

        return {
            "reasoning": reasoning,
            "content": content
        }
    except Exception as e:
        # If the API fails, remove the last user prompt so the history doesn't get out of sync
        history.pop()
        raise HTTPException(status_code=500, detail=str(e))