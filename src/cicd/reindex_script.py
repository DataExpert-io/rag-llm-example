import os
import subprocess
from pinecone import Pinecone
from git import Repo
from openai import OpenAI
from src.upsert_to_pinecone import upsert_to_pinecone
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Initialize Pinecone and OpenAI

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# --- Initialize Pinecone ---
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)
INDEX_NAME = 'rag-example-index'  # Replace with your Pinecone index name
# Assuming the index has already been created in Pinecone
index = pinecone_client.Index(INDEX_NAME)

def get_changed_files():
    """
    Returns a dict of changed files by status:
      {
        "added": [file1, file2, ...],
        "modified": [file3, ...],
        "deleted": [file4, ...],
        "renamed": [(old_name, new_name), ...]
      }

    We use git diff with name-status to capture file statuses.
    """
    diff_command = ["git", "diff", "--name-status", "HEAD~1", "HEAD"]
    result = subprocess.run(diff_command, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")

    changes = {
        "added": [],
        "modified": [],
        "deleted": [],
        "renamed": []
    }

    # Each line might look like:
    #   A   path/to/file
    #   M   path/to/file
    #   D   path/to/file
    #   R100   old/path  new/path
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] == "A":
            # Added
            changes["added"].append(parts[1])
        elif parts[0] == "M":
            # Modified
            changes["modified"].append(parts[1])
        elif parts[0] == "D":
            # Deleted
            changes["deleted"].append(parts[1])
        elif parts[0].startswith("R"):
            # Renamed (e.g. R100 means 100% similarity => rename, no change)
            # parts = ["R100", "old/path", "new/path"]
            if len(parts) >= 3:
                old_name = parts[1]
                new_name = parts[2]
                changes["renamed"].append((old_name, new_name))

    return changes


def get_embeddings_for_text(text):
    """
    Uses OpenAI's text-embedding-ada-002 model to embed text.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def process_file(file_path):
    """
    Given a file path, read its contents, generate an embedding.
    For large files, you might want to chunk them up.
    Returns a single embedding vector for the entire file.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    embedding = get_embeddings_for_text(text)
    return (text, embedding)




def delete_embeddings(file_paths):
    """
    Delete a list of file IDs from the Pinecone index.
    In this scenario, the 'id' is the file path we used when upserting.
    """
    index.delete(ids=file_paths)


def main():
    repo = Repo(".")

    organization = 'DataExpert-io'
    # For remote repositories
    repo_name = repo.remotes.origin.url.split('.git')[0].split('/')[-1]

    changes = get_changed_files()
    added_files = changes["added"]
    modified_files = changes["modified"]
    deleted_files = changes["deleted"]
    renamed_files = changes["renamed"]

    # Summarize changes
    print("Detected changes from HEAD~1 to HEAD:")
    print(f"  Added files: {added_files}")
    print(f"  Modified files: {modified_files}")
    print(f"  Deleted files: {deleted_files}")
    print(f"  Renamed files: {renamed_files}")

    # For newly added or modified files, generate embeddings and upsert.
    files_to_process = added_files + modified_files

    # If a file is renamed, you might handle it as 'delete old' + 'add new'
    # or preserve the embeddings with a new ID. Below, we do 'delete old' + 'add new':
    for old_name, new_name in renamed_files:
        deleted_files.append(organization + '/' + repo_name + '/' + old_name)
        files_to_process.append(organization + '/' + repo_name + '/' + new_name)

    # Process and upsert embeddings for new/modified files
    vectors_to_upsert = []
    for file_path in files_to_process:
        if not os.path.exists(file_path):
            # If the file somehow doesn't exist, skip it
            print(f"Skipping {file_path} - file does not exist locally.")
            continue

        try:
            data = process_file(file_path)

            id = organization + '/' + repo_name + '/' + file_path
            vectors_to_upsert.append((id, data[1], {
                'repo_name': repo_name,
                'file_path': file_path,
                'content': data[0],
                'category': 'Github'
            }))
            for vector_to_upsert in vectors_to_upsert:
                upsert_to_pinecone(vector_to_upsert[1], vector_to_upsert[2], vector_to_upsert[0])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Delete embeddings for removed files
    if deleted_files:
        try:
            delete_embeddings(deleted_files)
            print(f"Deleted embeddings for: {deleted_files}")
        except Exception as e:
            print(f"Error deleting embeddings for {deleted_files}: {e}")


if __name__ == "__main__":
    main()
