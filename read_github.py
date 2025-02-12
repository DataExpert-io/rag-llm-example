"""
Script to list all repositories in a given GitHub organization.
Requires:
  - PyGithub (pip install PyGithub)
  - A GitHub Personal Access Token
Usage:
  python get_org_repos.py <organization_name>
"""

import sys
import os
from github import Github
from chunk_text import chunk_gpt_tokens
from upsert_to_pinecone import upsert_to_pinecone


def fetch_contents(repo, path="", contents_list=None):
    """
    Recursively fetch all contents (files and directories) in the given path of the repository.
    """
    if contents_list is None:
        contents_list = []

    # Get the contents of the current directory
    items = repo.get_contents(path)

    for item in items:
        if item.type == "dir":
            # If it's a directory, recursively explore it
            contents_list = fetch_contents(repo, item.path, contents_list)
        else:
            # If it's a Python, Markdown, or SQL file, add to our list
            if item.path.endswith(".py") or item.path.endswith(".md") or item.path.endswith(".sql"):
                contents_list.append((item.path, str(item.decoded_content)))

    return contents_list

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_org_repos.py <organization_name>")
        sys.exit(1)

    org_name = sys.argv[1]

    # Option A: Use a Personal Access Token from an environment variable
    # Make sure to set GITHUB_TOKEN in your environment
    token = os.getenv('GITHUB_PAT')

    if not token:
        print("Please set the GITHUB_PAT environment variable with your GitHub Personal Access Token.")
        sys.exit(1)

    # Create a GitHub instance using the provided token
    g = Github(token)

    try:
        # Get the organization
        org = g.get_organization(org_name)

        # Fetch and list all repositories under this organization
        print(f"Repositories in organization '{org_name}':\n")
        for repo in org.get_repos():
            print(repo.full_name)
            repo = g.get_repo(repo.full_name)
            contents = fetch_contents(repo, path='', )
            print('we found', len(contents), 'files')
            for content_tuple in contents:
                file_name = content_tuple[0]
                content = content_tuple[1]
                chunks = chunk_gpt_tokens(content, chunk_size=200, overlap=50)
                print('upserting', len(chunks), 'tokens into Pinecone')
                for chunk in chunks:
                    upsert_to_pinecone(chunk['chunk_text'], metadata={
                        'repo': repo.full_name,
                        'file': file_name
                    })

    except Exception as e:
        print(f"Error accessing organization '{org_name}': {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()