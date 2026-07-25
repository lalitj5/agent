# llm.py
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import os
import sys
from dotenv import load_dotenv

load_dotenv()

#========================
# deepseek v4 flash
#========================
planner = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv('DEEPSEEK_API_KEY')
)

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


#========================
# glm 5.2
#========================
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

coder = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv('GLM_API_KEY')
)

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


#+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

"""
code_completion = coder.chat.completions.create(
  model="z-ai/glm-5.2",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=1,
  max_tokens=16384,
  seed=42,
  
  stream=True
)

for chunk in code_completion:
  if not getattr(chunk, "choices", None):
    continue
  if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
    continue
  delta = chunk.choices[0].delta
  if getattr(delta, "content", None) is not None:
    print(delta.content, end="")

#========================
# nemotron
#========================
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv('NEMO_API_KEY')
)


completion = client.chat.completions.create(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  messages=[{"role":"user","content":""}],
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
  stream=True
)

for chunk in completion:
  if not chunk.choices:
    continue
  reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
  if reasoning:
    print(reasoning, end="")
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
"""