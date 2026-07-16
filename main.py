from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from llm import planner
import asyncio

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# in memory db
sessions = {}
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
        sessions[req.session_id] = [{"role": "system", "content": planner_system_prompt}]
    history = sessions[req.session_id]
    history.append({"role": "user", "content": req.prompt})

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = planner.chat.completions.create(
                model="deepseek-ai/deepseek-v4-flash",
                messages=history,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9, 
                extra_body={
                        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "medium"}
                    }
                )
            
            assistant_message = response.choices[0].message
            reasoning = getattr(assistant_message, "reasoning", None) or getattr(assistant_message, "reasoning_content", None)
            content = assistant_message.content

            history.append({"role": "assistant", "content": content})

            return {"reasoning": reasoning, "content": content}
        except Exception as e:
            # if API fails, remove the last user prompt so the history doesn't get out of sync
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1) # Wait 1 second before retrying
            else:
                # Out of retries, clean up history and raise error
                history.pop()
                raise HTTPException(status_code=500, detail=f"LLM API failed after {max_retries} attempts: {str(e)}")