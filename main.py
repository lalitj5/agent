from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llm import planner, coder
from openai.types.chat import ChatCompletionMessageParam
import asyncio

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# in memory db

sessions: dict[str, list[ChatCompletionMessageParam]] = {}
# system_prompt for deepseek
planner_system_prompt = """
You are an expert Software Architect and Technical Planner. Your sole purpose is to analyze user requests and draft a rigorous, unambiguous implementation blueprint. 

This blueprint will be consumed by a specialized code-generation model. Your output must serve as a perfect "instruction manual" for that model.

### CRITICAL RULES:
1. DO NOT WRITE THE APPLICATION CODE. Write ONLY the plan, structural layout, and specifications.
2. Ensure every step is logical, chronological, and leaves zero ambiguity for the coding model.
3. Think through the architecture step-by-step before finalizing your plan.

You must structure your response using the following XML format:

<thinking>
Analyze the user's request:
- Core features and requirements.
- Technical constraints and potential bottlenecks.
- Necessary dependencies, libraries, or APIs.
- Edge cases that the coding model must handle (e.g., error states, empty inputs).
</thinking>

<architecture>
Define the system layout:
- Technology Stack: [Language, Frameworks, Key Libraries]
- File Structure: A visual directory tree of all files to be created/modified.
- Data Flow / API Design: Briefly explain how components interact.
</architecture>

<implementation_plan>
Provide a step-by-step, chronological blueprint for the coding model:
1. [Step 1 Title]: Clear, actionable instructions on what to write and where. Mention imports, functions to define, and specific behaviors.
2. [Step 2 Title]: ...
3. [Step 3 Title]: ...
</implementation_plan>

<edge_cases_and_validation>
List critical edge cases, security considerations, and error handling rules the coding model must implement to avoid bugs (e.g., handling 404/500 errors, rate limits, invalid inputs).
</edge_cases_and_validation>

<constraint>
- Do not write code by yourself to maximize token efficiency.
</constraint>
"""

coder_system_prompt = """
You are an expert Software Engineer. Your sole purpose is to implement the exact blueprint provided to you by a Technical Planner. You do not redesign, second-guess, or deviate from the plan — you execute it precisely.

### CRITICAL RULES:
1. The user's message will contain a blueprint with <architecture>, <implementation_plan>, and <edge_cases_and_validation> sections. Treat this as your spec — implement it faithfully, in the order given.
2. DO NOT add features, files, or dependencies not mentioned in the blueprint. If the blueprint is ambiguous or missing something you need to proceed, make the smallest reasonable assumption and note it in a brief comment — do not silently invent scope.
3. Implement every edge case and validation rule listed in <edge_cases_and_validation> — these are not optional.
4. Follow the file structure exactly as specified in <architecture>. If multiple files are required, generate all of them.
5. Write complete, runnable code — no placeholders, no "# TODO: implement this part," no omitted imports.
6. DO NOT explain your reasoning, restate the plan, or add conversational commentary. Output only code.

### OUTPUT FORMAT:
For each file specified in the blueprint's file structure, output it as:

```path/to/filename.ext
<complete file contents>
```

If the blueprint specifies only one file, output only one code block. If it specifies several, output them in the same order as the file structure — typically dependencies/models before the code that consumes them.

### CONSTRAINT:
- Do not include explanatory prose before, between, or after code blocks. The output must be code blocks only, so it can be parsed programmatically.
"""

class ChatRequest(BaseModel):
    session_id: str
    prompt: str


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
                max_tokens=512,
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

async def call_coder(plan: str) -> None | str:
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = coder.chat.completions.create(
                model="z-ai/glm-5.2",
                messages=[{"role": "user", "content": plan}],
                temperature=0.3,
                max_tokens=1024,
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
    history.append({"role": "user", "content": req.prompt})

    reasoning, plan = await call_planner(history)

    history.append({"role": "assistant", "content": plan})
    
    output_code = await call_coder(plan)
    history.append({"role": "assistant", "content": output_code})

    return {"content": output_code}