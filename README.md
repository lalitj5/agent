# A Broke Man's Claude Code

As a college student, claude code bills can ring up pretty fast! Using NVIDIA's developer pack, I am attempting to simulate the Mixture of Experts (MoE) architecture for coding agents and architected a multi-agent coding harness orchestrating DeepSeek V4 Flash as a planning agent and GLM 5.2 as a code-generation agent over NVIDIA’s free-tier inference endpoints **(Saving me 20 dollars per month!)**.

## Features:
- Fetches files from your github repo automatically and augments with user prompt
- Uses faster DeepSeek V4 Flash to create a plan for the changes
- Wires the plan to GLM 5.2, which is the coder agent, and GLM will stream back the output code in the chat window
- **Automatic PR generation with human-in-the-loop approval** (LATEST)


## Coming soon
- End-to-End auto verification for code (Start with having model write tests for code)
- Once models are rate-limited, switch to next available model, automatically
- Interact with multimodal inputs and tool calling

## Working on right now
- Agent brain for persistent context windows
hooking up database to save chats, also to move to vector database like Pinecone to perform semantic search on prompted codebase, rather than injecting hundreds of lines
