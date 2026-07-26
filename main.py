from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam
from llm import planner, coder, planner_system_prompt, coder_system_prompt
from github_client import fetch_file, create_pr_from_output
import asyncio
import json
from typing import AsyncGenerator

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# in memory db
sessions: dict[str, list[ChatCompletionMessageParam]] = {}


class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    repo: str | None = None       # "owner/repo"
    file_paths: str | None = None  # "src/main.py"


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


async def call_coder(plan: str, original_files: str) -> AsyncGenerator[str, None]:
    max_retries = 2
    user_content = plan
    if original_files:
        user_content = (
            "=== ORIGINAL FILE CONTENTS (for reference, do not repeat verbatim) ===\n"
            f"{original_files}\n"
            "=== END ORIGINAL FILE CONTENTS ===\n\n"
            f"Blueprint:\n{plan}"
        )

    for attempt in range(max_retries):
        try:
            response = coder.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=[{"role": "system", "content": coder_system_prompt}, {"role": "user", "content": user_content}],  # type: ignore[reportCallIssue]
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
            # Stream started successfully; iterate over chunks
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Handle both reasoning and content fields if present
                token = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if token:
                    yield token
                if delta.content:
                    yield delta.content
            return
        except Exception as e:
            print(f"Coder attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=f"Coder LLM API failed after {max_retries} attempts: {str(e)}")


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = [{"role": "system", "content": planner_system_prompt}]
    history = sessions[req.session_id]

    file_contents: dict[str, str] = {}
    file_shas: dict[str, str] = {}

    if req.repo and req.file_paths:
        file_paths = []
        for path in req.file_paths.split(","):
            file_paths.append(path.strip())

        for path in file_paths:
            content, sha = fetch_file(req.repo, path)
            file_contents[path] = content
            file_shas[path] = sha

        augment_prompt = ""
        for path, content in file_contents.items():
            augment_prompt += f"Existing file contents ({path}):\n===\n{content}\n===\n\n"
        augment_prompt += f"User request: {req.prompt}"

    else:
        augment_prompt = req.prompt
    history.append({"role": "user", "content": augment_prompt})

    reasoning, plan = await call_planner(history)
    history.append({"role": "assistant", "content": plan})

    combined_files = "\n\n".join(file_contents.values()) if file_contents else ""

    async def event_stream() -> AsyncGenerator[str, None]:
        full_output: list[str] = []
        try:
            async for token in call_coder(plan, combined_files):
                full_output.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            output_code = "".join(full_output)
            history.append({"role": "assistant", "content": output_code})

            pr_url = None
            pr_error = None
            if req.repo and file_shas and output_code:
                try:
                    pr_url = create_pr_from_output(req.repo, output_code, file_shas, plan, req.prompt)
                except Exception as e:
                    pr_error = str(e)

            final_data = {"type": "final", "content": output_code, "pr_url": pr_url}
            if pr_error:
                final_data["error"] = pr_error
            yield f"data: {json.dumps(final_data)}\n\n"

        except HTTPException as e:
            yield f"data: {json.dumps({'type': 'error', 'message': e.detail})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")