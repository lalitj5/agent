import os
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

