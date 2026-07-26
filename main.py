from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai.types.chat import ChatCompletionMessageParam
from llm import planner, coder, planner_system_prompt, coder_system_prompt
from github_client import fetch_file, create_pr_from_output
import asyncio

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

async def call_coder(plan: str, original_files: str) -> None | str:
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
            )
            msg = response.choices[0].message
            content = msg.content
            return content
        except Exception as e:
            print(f"Planner attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=f"Coder LLM API failed after {max_retries} attempts: {str(e)}")



@app.post("/chat")
async def chat(req: ChatRequest):
    if req.session_id not in sessions:
        # initialize chat history for each session
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
    output_code = await call_coder(plan, combined_files)
    history.append({"role": "assistant", "content": output_code})

    if req.repo and file_shas and output_code:
        pr_url = create_pr_from_output(req.repo, output_code, file_shas, plan, req.prompt)
        return {"content": output_code, "pr_url": pr_url}

    return {"content": output_code}




"""
NEED TO IMPLEMENT MULTIPLE FILE SHARING AND MULTIPLE FILE changes for PR

"""