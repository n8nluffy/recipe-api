import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from github import Github, Auth
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

load_dotenv()

github_token = os.getenv("GITHUB_TOKEN")
git = Github(auth=Auth.Token(github_token)) if github_token else None
repo_path = os.getenv("REPOSITORY")
print(f"Repo path: {repo_path}")
pr_number = int(os.getenv("PR_NUMBER")) if os.getenv("PR_NUMBER") else 0
print(f"PR number: {pr_number}")


def get_pr_details() -> Dict[str, Any]:
    """
    This function returns the details of the PR including the author, title, body, diff URL, state, and commit SHAs.
    :return: Dictionary containing the PR details
    """
    repo = git.get_repo(repo_path)
    pull_request = repo.get_pull(number=pr_number)
    commit_SHAs = []
    commits = pull_request.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)
    return {
        'author': pull_request.user.login,
        'title': pull_request.title,
        'body': pull_request.body,
        'diff_url': pull_request.diff_url,
        'state': pull_request.state,
        'commit_SHAs': commit_SHAs
    }


def get_file_content(file_path: str) -> str:
    """
    This function takes a file path as input and returns the content of the file.
    :param file_path:
    :return: Content of the file
    """
    repo = git.get_repo(repo_path)
    file_content = repo.get_contents(file_path).decoded_content.decode('utf-8')
    return file_content


def get_pr_commit_details(head_sha: str) -> List[Dict[str, Any]]:
    """
    This function takes a commit SHA as input and returns the details of the commit including the changed files, additions, deletions, and the patch.
    :param head_sha:
    :return: Dictionary containing the commit details
    """
    repo = git.get_repo(repo_path)
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def add_comment_to_state(ctx: Context, draft_comment: str):
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)


async def add_final_review_comment(ctx: Context, final_review_comment: str):
    current_state = await ctx.store.get("state")
    current_state["final_review_comment"] = final_review_comment
    await ctx.store.set("state", current_state)


def post_review_to_github(review_comment: str):
    repo = git.get_repo(repo_path)
    pull_request = repo.get_pull(number=pr_number)
    pull_request.create_review(body=review_comment, event="COMMENT")


print(f"OpenAI model being used: {os.getenv('OPENAI_MODEL', 'gpt-4')}")
print(f"OpenAI API base URL: {os.getenv('OPENAI_BASE_URL')}")

llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

sys_prompt = """You are the context gathering agent. When gathering context, you MUST gather:
  - The details: author, title, body, diff_url, state, and head_sha;
  - Changed files;
  - Any requested for files; 
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context for the commentor agent to write a thorough review comment.",
    tools=[get_pr_commit_details, get_file_content, get_pr_details, add_comment_to_state],
    system_prompt=sys_prompt,
    can_handoff_to=["CommentorAgent"]
)

commentor_prompt = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the Commentor Agent.
 - You should directly address the author. So your comments should sound like:
    - Thanks for fixing this.
    - I think all places where we call quote should be fixed.
    - Can you roll this fix out everywhere?
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
"""
commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    llm=llm,
    system_prompt=commentor_prompt,
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_agent_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment.
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must:
     - Be a ~200-300 word review in markdown format.
     - Specify what is good about the PR:
     - Did the author follow ALL contribution rules? What is missing?
     - Are there notes on test availability for new functionality? If there are new models, are there migrations for them?
     - Are there notes on whether new endpoints were documented?
     - Are there suggestions on which lines could be improved upon? Are these lines quoted?
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns.
 When you are satisfied, post the review to GitHub.
"""
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the drafted comment from the CommentorAgent, ensures it meets the criteria for a good PR review, and posts it to GitHub.",
    system_prompt=review_agent_prompt,
    tools=[add_final_review_comment, post_review_to_github]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": ""
    },
)


async def main():
    query = f"Write a review for PR: {pr_number}"
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    if git:
        git.close()
