import asyncio
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from github import Github, Auth
from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded successfully")

repo_url = os.getenv("REPOSITORY")
github_token = os.getenv("GITHUB_TOKEN")
logger.info(f"Repository URL configured: {repo_url if repo_url else 'Not set'}")
logger.info(f"GitHub token configured: {'Yes' if github_token else 'No'}")

git = Github(auth=Auth.Token(github_token)) if github_token else None
if git:
    logger.info("GitHub client initialized successfully")
else:
    logger.warning("GitHub client not initialized - token not provided")

repo_path = repo_url.replace("https://github.com/", "").replace(".git", "") if repo_url else ""
pr_number = int(os.getenv("PR_NUMBER")) if os.getenv("PR_NUMBER") else 0
logger.info(f"Repository path: {repo_path if repo_path else 'Not set'}")
logger.info(f"PR number configured: {pr_number if pr_number else 'Not set'}")


def get_pr_details() -> Dict[str, Any]:
    """
    This function returns the details of the PR including the author, title, body, diff URL, state, and commit SHAs.
    :return: Dictionary containing the PR details
    """
    logger.info(f"Fetching PR details for PR #{pr_number} from repository: {repo_path}")
    try:
        repo = git.get_repo(repo_path)
        logger.debug(f"Repository object retrieved: {repo.full_name}")

        pull_request = repo.get_pull(number=pr_number)
        logger.info(f"Pull request #{pr_number} retrieved successfully")
        logger.debug(f"PR Title: {pull_request.title}")
        logger.debug(f"PR Author: {pull_request.user.login}")
        logger.debug(f"PR State: {pull_request.state}")

        commit_SHAs = []
        commits = pull_request.get_commits()
        logger.info("Fetching commit SHAs from pull request")
        for c in commits:
            commit_SHAs.append(c.sha)
            logger.debug(f"Added commit SHA: {c.sha[:7]}...")

        logger.info(f"Successfully retrieved {len(commit_SHAs)} commit(s) for PR #{pr_number}")

        pr_details = {
            'author': pull_request.user.login,
            'title': pull_request.title,
            'body': pull_request.body,
            'diff_url': pull_request.diff_url,
            'state': pull_request.state,
            'commit_SHAs': commit_SHAs
        }
        logger.info("PR details compiled successfully")
        return pr_details
    except Exception as e:
        logger.error(f"Error fetching PR details: {str(e)}", exc_info=True)
        raise


def get_file_content(file_path: str) -> str:
    """
    This function takes a file path as input and returns the content of the file.
    :param file_path:
    :return: Content of the file
    """
    logger.info(f"Fetching content for file: {file_path}")
    try:
        repo = git.get_repo(repo_path)
        logger.debug(f"Repository object retrieved for file fetch")

        file_content = repo.get_contents(file_path).decoded_content.decode('utf-8')
        content_length = len(file_content)
        logger.info(f"Successfully retrieved file content: {file_path} ({content_length} characters)")
        logger.debug(f"First 100 characters: {file_content[:100]}...")

        return file_content
    except Exception as e:
        logger.error(f"Error fetching file content for {file_path}: {str(e)}", exc_info=True)
        raise


def get_pr_commit_details(head_sha: str) -> List[Dict[str, Any]]:
    """
    This function takes a commit SHA as input and returns the details of the commit including the changed files, additions, deletions, and the patch.
    :param head_sha:
    :return: Dictionary containing the commit details
    """
    logger.info(f"Fetching commit details for SHA: {head_sha[:7]}...")
    try:
        repo = git.get_repo(repo_path)
        logger.debug(f"Repository object retrieved for commit fetch")

        commit = repo.get_commit(head_sha)
        logger.info(f"Commit {head_sha[:7]} retrieved successfully")

        changed_files: list[dict[str, Any]] = []
        files = commit.files
        logger.info(f"Processing {len(files)} changed file(s) in commit")

        for f in files:
            file_info = {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch,
            }
            changed_files.append(file_info)
            logger.debug(f"Processed file: {f.filename} (status: {f.status}, +{f.additions}/-{f.deletions})")

        logger.info(f"Successfully processed {len(changed_files)} changed file(s)")
        return changed_files
    except Exception as e:
        logger.error(f"Error fetching commit details for SHA {head_sha[:7]}: {str(e)}", exc_info=True)
        raise


async def add_comment_to_state(ctx: Context, draft_comment: str):
    logger.info("Adding draft comment to workflow state")
    logger.debug(f"Draft comment length: {len(draft_comment)} characters")

    current_state = await ctx.store.get("state")
    logger.debug(f"Current state before update: {list(current_state.keys())}")
    print(f"Current state before update: {current_state}")

    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)

    logger.info("Draft comment successfully added to state")
    logger.debug(f"Updated state keys: {list(current_state.keys())}")
    print(f"Context updated with draft, state: {current_state}")


async def add_final_review_comment(ctx: Context, final_review_comment: str):
    logger.info("Adding final review comment to workflow state")
    logger.debug(f"Final review comment length: {len(final_review_comment)} characters")

    current_state = await ctx.store.get("state")
    logger.debug(f"Current state before adding final review: {list(current_state.keys())}")

    current_state["final_review_comment"] = final_review_comment
    await ctx.store.set("state", current_state)

    logger.info("Final review comment successfully added to state")
    logger.debug(f"Updated state keys: {list(current_state.keys())}")


def post_review_to_github(review_comment: str):
    logger.info(f"Posting review to GitHub for PR #{pr_number}")
    logger.debug(f"Review comment length: {len(review_comment)} characters")

    try:
        repo = git.get_repo(repo_path)
        logger.debug(f"Repository object retrieved: {repo.full_name}")

        pull_request = repo.get_pull(number=pr_number)
        logger.info(f"Pull request #{pr_number} retrieved for posting review")

        pull_request.create_review(body=review_comment, event="COMMENT")
        logger.info(f"Review successfully posted to GitHub PR #{pr_number}")
        logger.info("Review posting completed")
    except Exception as e:
        logger.error(f"Error posting review to GitHub: {str(e)}", exc_info=True)
        raise


llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
logger.info(f"OpenAI LLM initialized with model: {os.getenv('OPENAI_MODEL', 'gpt-4')}")
logger.info(f"OpenAI API key configured: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
logger.info(f"OpenAI base URL configured: {os.getenv('OPENAI_BASE_URL') if os.getenv('OPENAI_BASE_URL') else 'Default'}")

sys_prompt = """You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent."""

logger.info("Creating ContextAgent with tools: get_pr_commit_details, get_file_content, get_pr_details, add_comment_to_state")
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context for the commentor agent to write a thorough review comment.",
    tools=[get_pr_commit_details, get_file_content, get_pr_details, add_comment_to_state],
    system_prompt=sys_prompt,
    can_handoff_to=["CommentorAgent"]
)
logger.info("ContextAgent created successfully")

commentor_prompt = """
You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You should directly address the author. So your comments should sound like: \n
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?"
- You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
"""

logger.info("Creating CommentorAgent with tool: add_comment_to_state")
commentor_agent = FunctionAgent(
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    llm=llm,
    system_prompt=commentor_prompt,
    tools=[add_comment_to_state],
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)
logger.info("CommentorAgent created successfully")

review_agent_prompt = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
"""

logger.info("Creating ReviewAndPostingAgent with tools: add_final_review_comment, post_review_to_github")
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the drafted comment from the CommentorAgent, ensures it meets the criteria for a good PR review, and posts it to GitHub.",
    system_prompt=review_agent_prompt,
    tools=[add_final_review_comment, post_review_to_github]
)
logger.info("ReviewAndPostingAgent created successfully")

logger.info("Creating AgentWorkflow with three agents: ContextAgent, CommentorAgent, ReviewAndPostingAgent")
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review_comment": ""
    },
)
logger.info(f"AgentWorkflow created successfully with root agent: {review_and_posting_agent.name}")


async def main():
    logger.info("=== Starting PR Review Workflow ===")
    query = f"Write a review for PR: {pr_number}"
    logger.info(f"Query prepared: {query}")

    prompt = RichPromptTemplate(query)
    logger.info("RichPromptTemplate created")

    logger.info("Initiating workflow execution")
    handler = workflow_agent.run(prompt.format())
    logger.info("Workflow handler created, starting event stream processing")

    current_agent = None
    event_count = 0
    async for event in handler.stream_events():
        event_count += 1
        logger.debug(f"Processing event #{event_count}: {type(event).__name__}")

        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            logger.info(f"Agent switch detected - Now running: {current_agent}")
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            logger.info("Received AgentOutput event")
            if event.response.content:
                logger.info("Final response received from agent")
                logger.debug(f"Response length: {len(event.response.content)} characters")
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                tool_names = [call.tool_name for call in event.tool_calls]
                logger.info(f"Agent selected {len(tool_names)} tool(s): {', '.join(tool_names)}")
                print("Selected tools: ", tool_names)
        elif isinstance(event, ToolCallResult):
            logger.info(f"Tool execution completed")
            logger.debug(f"Tool output: {str(event.tool_output)[:200]}...")
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            logger.info(f"Calling tool: {event.tool_name}")
            # Don't log the actual arguments to avoid exposing sensitive data
            logger.debug(f"Tool arguments count: {len(event.tool_kwargs)}")
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    logger.info(f"Workflow completed - Total events processed: {event_count}")
    logger.info("=== PR Review Workflow Finished ===")


if __name__ == "__main__":
    logger.info("Script started - Initializing PR Review Agent")
    try:
        asyncio.run(main())
        logger.info("Script execution completed successfully")
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}", exc_info=True)
        raise
    finally:
        if git:
            logger.info("Closing GitHub client connection")
            git.close()
            logger.info("GitHub client closed successfully")
