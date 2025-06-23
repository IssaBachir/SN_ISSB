
from huggingface_hub import Repository
import os
import shutil
import subprocess

def deploy():
    token = os.getenv("HF_API_KEY")
    repo_id = "issabachir6/SN_ISSB"
    repo_dir = "hf_repo"

    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)

    repo = Repository(local_dir=repo_dir, clone_from=repo_id, use_auth_token=token)

    # Copier le modèle
    shutil.copy("models/model_final.pth", f"{repo_dir}/model_final.pth")

    # Configurer l'identité git pour éviter l'erreur
    subprocess.run(['git', 'config', 'user.email', 'action@github.com'], cwd=repo_dir)
    subprocess.run(['git', 'config', 'user.name', 'GitHub Action'], cwd=repo_dir)

    # Commit & push
    repo.push_to_hub(commit_message="Déploiement automatique du modèle")

if __name__ == "__main__":
    deploy()
