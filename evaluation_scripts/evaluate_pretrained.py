import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import soundfile as sf
import os
import jiwer
import string
from num2words import num2words
import re
import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("jacktol/atc-dataset")

# Define the phonetic alphabet for normalization
phonetic_alphabet = {
    'a': 'alfa', 'b': 'bravo', 'c': 'charlie', 'd': 'delta', 'e': 'echo',
    'f': 'foxtrot', 'g': 'golf', 'h': 'hotel', 'i': 'india', 'j': 'juliett',
    'k': 'kilo', 'l': 'lima', 'm': 'mike', 'n': 'november', 'o': 'oscar',
    'p': 'papa', 'q': 'quebec', 'r': 'romeo', 's': 'sierra', 't': 'tango',
    'u': 'uniform', 'v': 'victor', 'w': 'whiskey', 'x': 'x-ray', 'y': 'yankee', 'z': 'zulu'
}

# Normalization functions
def normalize_prediction(text):
    # Function to normalize the prediction to match ground truth format
    def convert_flight_level(match):
        flight_level_number = match.group(1)
        expanded_flight_level = ' '.join([num2words(int(digit)) for digit in flight_level_number])
        return f"flight level {expanded_flight_level}"

    def convert_altitude(match):
        number = match.group(1).replace(',', '')
        feet = match.group(2)
        num = int(number)
        number_in_words = num2words(num)
        return f"{number_in_words} {feet}"

    def convert_hyphenated_numbers(match):
        hyphenated_number = match.group()
        segments = hyphenated_number.split('-')
        result = []
        for segment in segments:
            sub_result = []
            for char in segment:
                if char.isdigit():
                    sub_result.append(num2words(int(char)))
                elif char.isalpha():
                    sub_result.append(phonetic_alphabet[char.lower()])
            result.append(' '.join(sub_result))
        return ' '.join(result)

    def convert_alphanumeric(match):
        token = match.group()
        has_digit = any(char.isdigit() for char in token)
        has_alpha = any(char.isalpha() for char in token)

        runway_match = re.match(r'^(\d{1,2})([LR])$', token, re.IGNORECASE)
        if runway_match:
            number = runway_match.group(1)
            direction = runway_match.group(2).upper()
            expanded_number = ' '.join([num2words(int(digit)) for digit in number])
            expanded_direction = 'left' if direction == 'L' else 'right'
            return f"{expanded_number} {expanded_direction}"

        if has_digit and has_alpha:
            result = []
            for char in token:
                if char.isdigit():
                    result.append(num2words(int(char)))
                elif char.isalpha():
                    result.append(phonetic_alphabet[char.lower()])
            return ' '.join(result)
        return token

    def convert_digits(match):
        number = match.group()
        if re.match(r'^\d+\.\d+$', number):
            parts = number.split('.')
            integer_part = ' '.join([num2words(int(digit)) for digit in parts[0]])
            decimal_part = ' '.join([num2words(int(digit)) for digit in parts[1]])
            return f"{integer_part} decimal {decimal_part}"
        elif number.isdigit():
            return ' '.join([num2words(int(digit)) for digit in number])
        return number

    def convert_single_letters(match):
        letter = match.group(1)
        return phonetic_alphabet[letter.lower()] if len(letter) == 1 else letter

    text = re.sub(r'\bFL(\d+)\b', convert_flight_level, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d{1,3}(?:,\d{3})?)\s*(feet)', convert_altitude, text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b', convert_hyphenated_numbers, text)
    text = re.sub(r'\b(?=[A-Za-z]*\d)[A-Za-z0-9]+\b', convert_alphanumeric, text)
    text = re.sub(r'\d+(\.\d+)?', convert_digits, text)
    text = re.sub(r'(?<=\s)([b-hj-zB-HJ-Z])(?=\s)', convert_single_letters, text)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = text.replace('take off', 'takeoff')
    text = text.translate(str.maketrans('', '', string.punctuation))
    normalized_text = text.lower()

    return normalized_text

def generate_transcription_and_process(model_name):
    # Load Hugging Face Whisper model and processor
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    log_file_path = "whisper-medium.en-evaluation-data-raw.txt"

    with open(log_file_path, "w") as log_file:
        wer_list = []

        for idx in range(len(dataset['test'])):
            sample = dataset['test'][idx]
            audio_array = np.array(sample['audio']['array'])
            audio_sr = sample['audio']['sampling_rate']
            ground_truth = sample['text']

            # Save audio to temporary file
            audio_path = f"temp_audio_{idx}.wav"
            sf.write(audio_path, audio_array, audio_sr)

            # Read and process the audio
            audio_input, _ = sf.read(audio_path)
            inputs = processor(audio_input, return_tensors="pt", sampling_rate=audio_sr)
            inputs = {key: val.to(model.device) for key, val in inputs.items()}

            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Normalize prediction
            normalized_prediction = normalize_prediction(prediction)

            # Compute Word Error Rate (WER)
            wer = jiwer.wer(ground_truth, normalized_prediction)
            wer_list.append(wer)

            # Log the results
            log_file.write(f"--------------------------------------------------\n")
            log_file.write(f"Sample {idx + 1}:\n")
            log_file.write(f"Ground Truth: {ground_truth}\n")
            log_file.write(f"Prediction: {prediction}\n")
            log_file.write(f"Normalized Prediction: {normalized_prediction}\n")
            log_file.write(f"Word Error Rate (WER): {wer * 100:.2f}%\n")
            log_file.write(f"--------------------------------------------------\n")

            if os.path.exists(audio_path):
                os.remove(audio_path)

        avg_wer = np.mean(wer_list) * 100

        log_file.write(f"\nAverage Word Error Rate (WER) across {len(dataset['test'])} samples: {avg_wer:.2f}%\n")
        log_file.write(f"--------------------------------------------------\n")

    process_evaluation_log(log_file_path, avg_wer)

def process_evaluation_log(log_file_path, avg_wer):
    with open(log_file_path, 'r') as file:
        data = file.read()

    pattern = r"Sample\s+(\d+):\s*Ground Truth:\s*(.+?)\s*Prediction:.*?\s*Normalized Prediction:\s*(.+?)\s*Word Error Rate \(WER\):\s*([\d.]+)%"

    matches = re.findall(pattern, data)

    if not matches:
        print("No matches found. Please check the format of the input file or the regex pattern.")
    else:
        df = pd.DataFrame(matches, columns=['Sample', 'Ground Truth', 'Prediction', 'WER'])

        df['Sample'] = df['Sample'].astype(int)
        df['WER'] = df['WER'].astype(float)

        average_wer = df['WER'].mean()

        print(f"Average WER: {average_wer:.2f}%")

        df_sorted = df.sort_values(by='WER', ascending=False)

        wer_str = f"{avg_wer:.2f}"

        csv_filename = f'whisper-medium.en-{wer_str}-WER-evaluation-data.csv'
        df_sorted.to_csv(csv_filename, index=False)

        print(f"Results saved to: {csv_filename}")

        if os.path.exists(log_file_path):
            os.remove(log_file_path)

if __name__ == "__main__":
    model_name = "openai/whisper-medium.en"
    generate_transcription_and_process(model_name=model_name)