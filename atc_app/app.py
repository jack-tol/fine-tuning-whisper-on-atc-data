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
Based on the context and segmentation of each transmission, label it as either 'ATC' or 'Pilot'."""


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
## Welcome to the **Aviation Transcription Assistant**

---

### Purpose

This tool uses OpenAI's Whisper model, **finetuned for Air Traffic Control (ATC)**, to transcribe aviation communications into text. It helps improve clarity and reduce communication errors in the aviation industry.

---

### The Problem

Communication errors cause **over 70% of aviation incidents** (NASA). Non-standard language and miscommunication are common, with **44% of pilots** facing these issues on every flight (IATA). This tool helps create a clear record of pilot-ATC communications for review.

---

### The Solution

By converting communications into **standard ATC syntax**, this tool enhances accuracy and acts as a **backup log**, reducing risks from misunderstood instructions.

---

## How to Use

1. Upload an **audio file** of aviation communications.
2. The tool will **transcribe the audio** into standard ATC format.
3. Review the transcription for accuracy.

---
"""

    await cl.Message(content=welcome_message).send()

    # Prompt user to upload audio file
    files = await cl.AskFileMessage(
        content="", 
        accept={"audio/wav": [".wav"]},
        max_size_mb=50
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
