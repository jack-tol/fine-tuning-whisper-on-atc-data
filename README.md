# Fine-tuning Whisper on ATC Data

Welcome to the repository for the **Fine-tuning Whisper on ATC Data** project. This project involves fine-tuning OpenAI’s Whisper model on domain-specific ATC datasets to significantly improve transcription accuracy for aviation communication, reducing the Word Error Rate (WER) by 84% compared to the pretrained model.

The repository includes scripts for dataset preparation, training, evaluation, and deployment of the fine-tuned model, which is available for use via Hugging Face.

## Project Overview

Fine-tuning Whisper on ATC-specific data significantly improves the transcription of critical communication between pilots and air traffic controllers. The fine-tuned model reduces WER to 15.08% (compared to 94.59% for the pretrained model), addressing many common communication issues in aviation, such as accent variations and ambiguous phrasing.

You can explore the **open-source models** on Hugging Face:

- [Whisper Medium EN Fine-Tuned for ATC](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC)
- [Whisper Medium EN Fine-Tuned for ATC (Faster Whisper)](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper)

Additionally, the **ATC Transcription Assistant** is an application hosted on Hugging Face Spaces that allows you to upload audio (MP3 or WAV) to transcribe ATC communications. Try it out [here](https://huggingface.co/spaces/jacktol/ATC-Transcription-Assistant).

The **dataset** used for fine-tuning can also be found on Hugging Face: [ATC Dataset](https://huggingface.co/datasets/jacktol/atc-dataset).

For further context, read the full blog post covering the fine-tuning process on my website: [Fine-Tuning Whisper for ATC: 84% Improvement in Transcription Accuracy](https://jacktol.net/posts/fine-tuning_whisper_for_atc/).

### Repository Structure

- **custom_dataset_processing/**
  - `create_and_upload_dataset.py`: Script to combine and preprocess ATC datasets for training and testing.
- **evaluation_scripts/**
  - `evaluate_fine-tuned.py`: Evaluate the fine-tuned Whisper model on test data.
  - `evaluate_pretrained.py`: Evaluate the pretrained Whisper model for comparison.
  - `interactive_model_comparison_fine-tuned_vs_pretrained_whisper_on_atc_data.ipynb`: Interactive Jupyter Notebook to compare the fine-tuned model with the pretrained one, allowing you to listen to audio samples and see WER differences.
- **removed_samples/**: Contains data for samples that were removed from the dataset during evaluation due to incorrect ground truth.
- **training_script/**
  - `train.py`: The main training script that fine-tunes the Whisper model on the ATC dataset, with dynamic data augmentation.
- **utils/**
  - `export_model_for_inference.py`: Script to export the fine-tuned model for inference.
  - `upload_local_models_to_huggingface.py`: Script to upload models to the Hugging Face model hub.
  - `whisper_to_optimized_bin.bash`: Bash script to convert the Whisper model to an optimized format using transformers.
- **requirements.txt**: Contains the dependencies required to run the project.

### Key Files

- `create_and_upload_dataset.py`: Combines and preprocesses the ATCO2 and UWB-ATCC datasets, applies filtering, and uploads the processed dataset to Hugging Face.
- `train.py`: Defines the fine-tuning process, including dynamic data augmentation and training configuration.
- `evaluate_fine-tuned.py` & `evaluate_pretrained.py`: Evaluate the fine-tuned and pretrained Whisper models on the ATC dataset to compare WER.

### How to Run

1. **Environment Setup**:
   Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   Dataset Preparation: Use the `create_and_upload_dataset.py` script to process the ATC datasets and upload them to Hugging Face.

Model Training: Fine-tune the Whisper model using the `train.py` script. This script dynamically augments data and trains the model over 10 epochs with early stopping.

Evaluation: Use the evaluation scripts to measure the WER of the fine-tuned model compared to the pretrained model. The Jupyter Notebook in `evaluation_scripts/` offers an interactive way to explore the results.

Model Export & Upload: Use the `export_model_for_inference.py` script to prepare the model for inference, and the `upload_local_models_to_huggingface.py` script to upload your model to Hugging Face.

## Model Usage

The fine-tuned model is readily available on Hugging Face. You have two options to access and use the models:

1. **Download Locally**: You can download the models directly to your local system via these links:

   - [Whisper Medium EN Fine-Tuned for ATC](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC)
   - [Whisper Medium EN Fine-Tuned for ATC (Faster Whisper)](https://huggingface.co/jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper)

2. **Use Online**: Alternatively, you can use the models online via the [ATC Transcription Assistant](https://huggingface.co/spaces/jacktol/ATC-transcription-assistant) on Hugging Face. Simply upload your ATC communication audio file (MP3 or WAV), and the model will generate transcriptions for you directly on the screen.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For further details or inquiries, contact me at contact@jacktol.net.
