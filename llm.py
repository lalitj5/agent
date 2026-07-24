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

messages: list[ChatCompletionMessageParam] =[{"role":"user","content":""}]

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
"""
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