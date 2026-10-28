"""Push a trained adapter folder to the Hugging Face Hub. The token comes from the environment (HF_TOKEN, e.g. a
Kaggle secret) or the cached `hf auth login`; it is never printed."""

from __future__ import annotations

from pathlib import Path


def push_folder(folder: Path, repo_id: str, private: bool = True, message: str = "upload adapter") -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message=message)
    return f"https://huggingface.co/{repo_id}"
