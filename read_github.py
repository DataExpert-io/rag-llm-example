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


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_org_repos.py <organization_name>")
        sys.exit(1)

    org_name = sys.argv[1]

    # Option A: Use a Personal Access Token from an environment variable
    # Make sure to set GITHUB_TOKEN in your environment
    token = os.getenv('GITHUB_PAT')

    if not token:
        print("Please set the GITHUB_TOKEN environment variable with your GitHub Personal Access Token.")
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
            contents = repo.get_contents('/')
            print(contents[0].decoded_content)
            print(contents[0])

    except Exception as e:
        print(f"Error accessing organization '{org_name}': {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()