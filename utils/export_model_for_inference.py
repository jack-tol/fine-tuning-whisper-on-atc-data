from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

checkpoint_path = 'PATH_TO_YOUR_CHECKPOINT'
base_model_id = 'INPUT_MODEL_ID_FOR_BASE_MODEL (e.g., openai/whisper-medium.en)'

model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
processor = WhisperProcessor.from_pretrained(base_model_id)

exported_model_path = checkpoint_path + '-exported-model'

os.makedirs(exported_model_path, exist_ok=True)
model.save_pretrained(exported_model_path)
processor.save_pretrained(exported_model_path)