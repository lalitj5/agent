import os
import re
import time
from github import Auth
from github.Auth import Token
from github.MainClass import Github
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN=os.getenv('GITHUB_TOKEN')
if GITHUB_TOKEN is None:
    raise RuntimeError("GITHUB_TOKEN environment variable is not set")

auth = Auth.Token(GITHUB_TOKEN)
gh = Github(auth=auth)

def fetch_file(repo_full_name: str, file_path: str, branch: str = "main") -> tuple[str, str]:
    repo = gh.get_repo(repo_full_name)
    contents = repo.get_contents(file_path, ref=branch)
    file_content = contents.decoded_content.decode("utf-8") # type: ignore[reportCallIssue]
    if file_content is None:
        raise ValueError()
    print(file_content[:20])
    return file_content, contents.sha # type: ignore[reportCallIssue]

def create_branch(repo_full_name: str, base_branch: str = "main") -> str:
    repo = gh.get_repo(repo_full_name)
    base_ref = repo.get_git_ref(f"heads/{base_branch}")
    new_branch_name = f"agent/{int(time.time())}"
    repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=base_ref.object.sha)
    return new_branch_name

def parse_coder_output(output_code: str) -> dict[str, str]:
    """Splits fenced code blocks like ```path/to/file.py\n<content>\n``` into {path: content}."""
    pattern = r"```([^\n`]+)\n(.*?)```"
    matches = re.findall(pattern, output_code, re.DOTALL)
    return {path.strip(): content.strip() for path, content in matches}

def update_files(repo_full_name: str, file_updates: dict[str, str], file_shas: dict[str, str], branch: str, commit_message: str):
    repo = gh.get_repo(repo_full_name)
    for path, content in file_updates.items():
        if path not in file_shas:
            raise ValueError(f"No known sha for {path}.")
        repo.update_file(
            path=path,
            message=commit_message,
            content=content,
            sha=file_shas[path],
            branch=branch,
        )

def open_pr(repo_full_name: str, branch: str, base_branch: str, title: str, body: str):
    repo = gh.get_repo(repo_full_name)
    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch,
        base=base_branch
    )
    return pr.html_url


def create_pr_from_output(repo_full_name: str, output_code: str, file_shas: dict[str, str], plan: str, prompt: str) -> str:
    file_updates = parse_coder_output(output_code)
    if not file_updates:
        raise ValueError("Coder output contained no parseable file blocks")

    branch = create_branch(repo_full_name)
    update_files(repo_full_name, file_updates, file_shas, branch, commit_message=f"Agent: {prompt[:40]}")
    return open_pr(repo_full_name, branch, "main", title=f"Agent: {prompt[:40]}", body=plan)