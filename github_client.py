import os
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

def update_file(repo_full_name: str, file_path: str, new_content: str, file_sha: str, branch: str, commit_message: str):
    if (file_sha == ""):
        raise ValueError("File sha cannot be empty string")
    
    repo = gh.get_repo(repo_full_name)
    repo.update_file(
        path=file_path,
        message=commit_message,
        content=new_content,
        sha=file_sha, # the SHA from earlier fetch_file call
        branch=branch
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

def create_pr_from_output(repo, file_path, file_sha, output_code, plan, prompt) -> str:
    branch = create_branch(repo)
    update_file(repo, file_path, output_code, file_sha, branch, commit_message=f"Agent: {prompt[:50]}")
    return open_pr(repo, branch, "main", title=f"Agent: {prompt[:50]}", body=plan)