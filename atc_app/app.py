import chainlit as cl
from faster_whisper import WhisperModel
from openai import AsyncOpenAI

# Model path for the fine-tuned Whisper model
model_path = "jacktol/whisper-medium.en-fine-tuned-for-ATC-faster-whisper"

# Initialize the Whisper model and OpenAI client
whisper_model = WhisperModel(model_path, device="cuda", compute_type="float32")
client = AsyncOpenAI()

# System prompt for converting transcript to standard ATC syntax
system_prompt = """Convert the provided transcript into standard pilot-ATC syntax without altering the content.
Ensure that all runway and heading numbers are formatted correctly (e.g., '11L' for 'one one left'). Use standard
aviation phraseology wherever applicable. Maintain the segmentation of the transcript as provided, but exclude the timestamps.
Based on the context and segmentation of each transmission, label it as either 'ATC' or 'Pilot'. At the very beginning of your
response place a horizonal div with "---" and then line-break, and then add a H2 which says "Transciption, and then
proceed with the transciption."""


# Function to transcribe audio and return the concatenated transcript with segment info
def transcribe_audio(file_path):
    segments, info = whisper_model.transcribe(file_path, beam_size=5)
    transcript = []
    
    # Combine all segments with timestamps
    for segment in segments:
        transcript.append(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    
    print('\n'.join(transcript).strip())
    
    return '\n'.join(transcript).strip()

@cl.on_chat_start
async def start_chat():
    # Welcome message
    welcome_message = """
## Welcome to the **ATC Transcription Assistant**

---

### What is this tool for?

This tool transcribes **Air Traffic Control (ATC)** audio using OpenAI’s **Whisper medium.en** model, fine-tuned for ATC communications. Developed as part of a research project, the fine-tuned **Whisper medium.en** model offers significant improvements in transcription accuracy for ATC audio.

---

### Performance

- **Fine-tuned Whisper medium.en WER**: 15.08%
- **Non fine-tuned Whisper medium.en WER**: 94.59%
- **Relative improvement**: 84.06%

While the fine-tuned model performs much better, **we cannot guarantee the accuracy of the transcriptions**. For more details on the fine-tuning process, see the [blog post](https://jacktol.net/posts/fine-tuning_whisper_on_atc_data), or check out the [project repository](https://github.com/jack-tol/fine-tuning-whisper-on-atc-data). Feel free to contact me at [contact@jacktol.net](mailto:contact@jacktol.net).

---

### How to Use

1. **Upload an ATC audio file**: Upload an audio file in **MP3** or **WAV** format containing ATC communications.
2. **View the transcription**: The tool will transcribe the audio and display the text on the screen.
3. **Transcribe another audio**: Click **New Chat** in the top-right to start a new transcription.

---

To get started, upload the audio below.
"""

    await cl.Message(content=welcome_message).send()

    # Prompt user to upload audio file
# Prompt user to upload audio file (MP3 or WAV)
    files = await cl.AskFileMessage(
        content="", 
        accept={
            "audio/wav": [".wav"],
            "audio/mpeg": [".mp3"]
        },
        max_size_mb=50,
        timeout=3600
    ).send()


    if files:
        audio_file = files[0]

        # Get the full segmented transcription with timestamps
        transcription = transcribe_audio(audio_file.path)

        # Send the entire transcription to the LLM for ATC syntax processing
        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcription},
            ],
            stream=True,
            model="gpt-4o",  # Use the appropriate model
            temperature=0,
        )

        # Stream the ATC-processed output
        async for part in stream:
            token = part.choices[0].delta.content or ""
            await msg.stream_token(token)

        await msg.update()
