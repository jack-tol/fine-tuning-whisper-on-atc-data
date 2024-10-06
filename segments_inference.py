from faster_whisper import WhisperModel

# Load the fine-tuned model from Hugging Face
model_path = "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper"

# If you're running this on a CPU, you can modify the `device` argument to "cpu".
model = WhisperModel(model_path, device="cuda", compute_type="float32")

# Transcribe the audio file (change "audio.mp3" to the path of your audio file)
segments, info = model.transcribe("test.wav", beam_size=5)

# Print detected language and its probability
print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")

# Print the transcription at the segment level
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
