from huggingface_hub import HfApi, upload_folder

repos = {
    "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper": "whisper-medium.en-fine-tuned-for-ATC-faster-whisper",
    "jacktol/whisper-medium.en-fine-tuned-for-ATC": "whisper-medium.en-fine-tuned-for-ATC"
}

api = HfApi()

def upload_model(repo_name, model_folder):
    try:
        api.create_repo(repo_id=repo_name)
        print(f"Repository {repo_name} created successfully.")
    except Exception as e:
        print(f"Repository {repo_name} already exists or another error occurred: {e}")

    try:
        upload_folder(
            folder_path=model_folder,
            repo_id=repo_name,
            repo_type="model",
        )
        print(f"Successfully uploaded {model_folder} to {repo_name}.")
    except Exception as e:
        print(f"Error while uploading {model_folder} to {repo_name}: {e}")

for repo_name, model_folder in repos.items():
    print(f"Uploading model from folder {model_folder} to repository {repo_name}...")
    upload_model(repo_name, model_folder)
    print(f"Finished uploading {model_folder}.\n")

print("Both models have been uploaded successfully.")