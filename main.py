```main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai.types.chat import ChatCompletionMessageParam
from llm import planner, coder, planner_system_prompt, coder_system_prompt
from github_client import fetch_file
from typing import AsyncGenerator
import asyncio
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# in memory db
sessions: dict[str, list[ChatCompletionMessageParam]] = {}



class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    repo: str | None = None       # "owner/repo"
    file_path: str | None = None  # "src/main.py"


@app.get("/")
def home():
    return FileResponse("static/index.html")

async def call_planner(history: list[ChatCompletionMessageParam]) -> tuple[str | None, str]:
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = planner.chat.completions.create(
                model="deepseek-ai/deepseek-v4-flash",
                messages=history,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "medium"}}
            )
            msg = response.choices[0].message
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            content = msg.content

            if content is None:
                raise ValueError("Planner returned empty content")
            
            return reasoning, content
        except Exception as e:
            print(f"Planner attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                history.pop()
                raise HTTPException(status_code=500, detail=f"Planner LLM API failed after {max_retries} attempts: {str(e)}")
    raise RuntimeError("call_planner exited retry loop without returning or raising")

async def stream_coder(plan: str, original_file: str) -> AsyncGenerator[str, None]:
    max_retries = 2
    user_content = plan
    if original_file:
        user_content = f"Original file contents:\n```\n{original_file}\n```\n\nBlueprint:\n{plan}"

    for attempt in range(max_retries):
        try:
            stream = await coder.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=[{"role": "system", "content": coder_system_prompt}, {"role": "user", "content": user_content}],  # type: ignore[reportCallIssue]
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            return
        except Exception as e:
            print(f"Coder attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                yield f"[Error: Streaming failed: {str(e)}]"
                raise HTTPException(status_code=500, detail=f"Coder LLM API failed after {max_retries} attempts: {str(e)}")



@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        # initialize chat history for each session
        sessions[req.session_id] = [{"role": "system", "content": planner_system_prompt}]
    history = sessions[req.session_id]

    file_content = None
    if req.repo and req.file_path:
        file_content, file_sha = fetch_file(req.repo, req.file_path)
        augment_prompt = f"""
        Existing file contents ({req.file_path}):\n```\n{file_content}\n```\n\nUser request: {req.prompt}
        """
    else:
        augment_prompt = req.prompt
    history.append({"role": "user", "content": augment_prompt})

    reasoning, plan = await call_planner(history)

    history.append({"role": "assistant", "content": plan})

    async def event_stream() -> AsyncGenerator[str, None]:
        full_output = ""
        try:
            async for chunk in stream_coder(plan, file_content):  # type: ignore[reportCallIssue]
                full_output += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except HTTPException as e:
            yield f"data: {json.dumps({'error': e.detail})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if full_output:
                history.append({"role": "assistant", "content": full_output})
            elif not full_output:
                history.append({"role": "assistant", "content": "No output"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

```static/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Coder Chat</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; background-color: #f9f9f9; }
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
        .message { padding: 10px 15px; border-radius: 15px; max-width: 80%; word-wrap: break-word; white-space: pre-wrap; }
        .user-message { background-color: #007bff; color: white; align-self: flex-end; }
        .assistant-message { background-color: #e9ecef; color: black; align-self: flex-start; }
        #input-container { display: flex; padding: 20px; background-color: #fff; border-top: 1px solid #ddd; }
        #prompt-input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 5px; resize: none; height: 60px; }
        #send-button { margin-left: 10px; padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        #send-button:disabled { background-color: #ccc; cursor: not-allowed; }
        .input-field { margin-bottom: 10px; padding: 5px; border: 1px solid #ccc; border-radius: 5px; width: 100%; box-sizing: border-box; }
        #meta-inputs { padding: 20px 20px 0 20px; background: #fff; border-top: 1px solid #ddd; }
    </style>
</head>
<body>

    <div id="chat-container"></div>
    
    <div id="meta-inputs">
        <input type="text" id="repo-input" class="input-field" placeholder="owner/repo (optional)">
        <input type="text" id="file-path-input" class="input-field" placeholder="src/main.py (optional)">
    </div>

    <div id="input-container">
        <textarea id="prompt-input" placeholder="Enter your prompt here..."></textarea>
        <button id="send-button" onclick="sendMessage()">Send</button>
    </div>

    <script>
        const sessionId = Math.random().toString(36).substring(7);
        let abortController = null;

        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendButton = document.getElementById('send-button');

        promptInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function appendMessage(content, role) {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', role === 'user' ? 'user-message' : 'assistant-message');
            messageDiv.textContent = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return messageDiv;
        }

        async function sendMessage() {
            const prompt = promptInput.value.trim();
            if (!prompt) return;

            if (abortController) {
                abortController.abort();
            }
            abortController = new AbortController();

            appendMessage(prompt, 'user');
            promptInput.value = '';
            sendButton.disabled = true;

            const assistantMessageDiv = appendMessage('', 'assistant');
            const repo = document.getElementById('repo-input').value.trim() || null;
            const filePath = document.getElementById('file-path-input').value.trim() || null;

            const data = {
                session_id: sessionId,
                prompt: prompt,
                repo: repo,
                file_path: filePath
            };

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data),
                    signal: abort