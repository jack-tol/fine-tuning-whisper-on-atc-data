# Import necessary libraries
from datasets import load_dataset, Audio
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, ClippingDistortion, Trim

# Load the ATC dataset
dataset = load_dataset("jacktol/atc_dataset")
train_dataset = dataset['train']
test_dataset = dataset['test']

# Define model parameters
model_id = 'openai/whisper-medium.en'
out_dir = 'whisper-medium.en-atc-dataset'
epochs = 10

# Load the feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id, language='English', task='transcribe')
processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')

# Ensure dataset audio sampling rate is 16KHz
train_dataset = train_dataset.cast_column('audio', Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column('audio', Audio(sampling_rate=16000))

# Define augmentation function with clipping
def augment_audio_with_clipping(audio, sampling_rate, augmentation_level):
    noise_severity = augmentation_level * 0.05
    pitch_severity = augmentation_level * 2
    time_stretch_severity = 1.0 + (augmentation_level * 0.2)
    shift_severity = augmentation_level * 0.1
    gain_severity = augmentation_level * 10

    min_noise_amplitude = max(0.001, noise_severity / 10)
    
    augment = Compose([
        AddGaussianNoise(min_amplitude=min_noise_amplitude, max_amplitude=noise_severity, p=0.5),
        TimeStretch(min_rate=1.0, max_rate=time_stretch_severity, p=0.5),
        PitchShift(min_semitones=-pitch_severity, max_semitones=pitch_severity, p=0.5),
        Shift(min_shift=-shift_severity, max_shift=shift_severity, p=0.5),
        Gain(min_gain_in_db=-gain_severity, max_gain_in_db=gain_severity, p=0.5),
        ClippingDistortion(p=0.3),
        Trim(top_db=30, p=0.3)
    ])

    augmented_audio = augment(samples=audio, sample_rate=sampling_rate)
    return augmented_audio

# Define the preprocessing function for the test dataset (without augmentation)
def prepare_dataset(batch):
    audio = batch['audio']
    batch['input_features'] = []
    for a in audio:
        audio_array = a['array'].astype(np.float32)
        input_features = feature_extractor(audio_array, sampling_rate=a['sampling_rate']).input_features[0]
        batch['input_features'].append(input_features)
    
    tokenized = tokenizer(batch['text'], padding='longest', return_attention_mask=True)
    batch['labels'] = tokenized['input_ids']
    batch['decoder_attention_mask'] = tokenized['attention_mask']
    
    return batch

# Preprocess the test dataset
test_dataset = test_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=64,
    num_proc=4  # Adjust based on your CPU cores
)

# Define custom AugmentedDataset class for training dataset
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, feature_extractor, tokenizer, augment_audio_with_clipping,
        initial_augmentation_level=0.5, final_augmentation_level=0.1, epochs=10
    ):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.augment_audio_with_clipping = augment_audio_with_clipping
        self.initial_augmentation_level = initial_augmentation_level
        self.final_augmentation_level = final_augmentation_level
        self.epochs = epochs - 1  # Adjust for zero-based indexing
        self.current_epoch = 0
        self.augmentation_level = initial_augmentation_level

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        # Exponential decay of augmentation level
        decay_rate = (self.final_augmentation_level / self.initial_augmentation_level) ** (1 / self.epochs)
        self.augmentation_level = self.initial_augmentation_level * (decay_rate ** self.current_epoch)
        print(f"Epoch {self.current_epoch}: Augmentation level set to {self.augmentation_level}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        audio = batch['audio']
        audio_array = audio['array'].astype(np.float32)
        augmented_audio = self.augment_audio_with_clipping(
            audio_array, audio['sampling_rate'], self.augmentation_level
        )
        input_features = self.feature_extractor(
            augmented_audio, sampling_rate=audio['sampling_rate']
        ).input_features[0]
        batch['input_features'] = input_features

        tokenized = self.tokenizer(
            batch['text'], padding='longest', return_attention_mask=True
        )
        batch['labels'] = tokenized['input_ids']
        batch['decoder_attention_mask'] = tokenized['attention_mask']
        
        return batch

# Create augmented training dataset with updated initial and final augmentation levels
train_dataset = AugmentedDataset(
    dataset=train_dataset,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    augment_audio_with_clipping=augment_audio_with_clipping,
    initial_augmentation_level=0.5,  # Updated initial level
    final_augmentation_level=0.1,
    epochs=epochs
)

# Callback to update the dataset's current epoch
class UpdateDatasetEpochCallback(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(state.epoch)

# Data Collator for padding input and labels
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors='pt'
        )

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors='pt'
        )

        labels = labels_batch['input_ids'].masked_fill(
            labels_batch['attention_mask'].ne(1), -100
        )
        batch['labels'] = labels

        # Include decoder attention mask
        batch['decoder_attention_mask'] = labels_batch['attention_mask']

        return batch

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# Update model configuration
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.pad_token_id = tokenizer.pad_token_id

# Create data collator instance
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Evaluation metric: Word Error Rate (WER)
metric = evaluate.load('wer')

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}

# Training arguments with updated evaluation strategy
training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    per_device_train_batch_size=16,    # Adjust batch size as needed
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,     # Effective batch size of 32
    learning_rate=1e-5,                # Lower initial learning rate
    warmup_steps=500,                  # Adjusted warmup steps
    num_train_epochs=epochs,
    evaluation_strategy='epoch',       # Evaluate at each epoch
    save_strategy='epoch',             # Save model at each epoch
    logging_strategy='epoch',          # Log metrics at each epoch
    predict_with_generate=True,
    generation_max_length=225,
    report_to=['tensorboard'],
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    dataloader_num_workers=4,          # Adjust based on your CPU cores
    save_total_limit=2,
    lr_scheduler_type='cosine',
    seed=42,
    data_seed=42,
    weight_decay=0.01,                 # Added weight decay for regularization
    bf16=True,                         # Enable mixed precision with bf16 if available
    fp16=False                         # Disable fp16 to prevent numerical instability
)

# Define Trainer with EarlyStoppingCallback and UpdateDatasetEpochCallback
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        UpdateDatasetEpochCallback(train_dataset)
    ]
)

# Start training
trainer.train()

# Perform final evaluation on the test dataset
final_metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Final evaluation metrics:", final_metrics)