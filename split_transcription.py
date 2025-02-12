import os
import subprocess
import math
from openai import OpenAI
from chunk_text import chunk_gpt_tokens
from upsert_to_pinecone import upsert_to_pinecone
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 1. Configuration
VIDEO_FILE = "videos/data_contracts.mp4"
CHUNK_LENGTH_SECONDS = 60  # for example, 300s = 5 minutes per chunk
TEMP_DIR = "temp_chunks"
LANGUAGE = "en"

# 2. Create a directory for our temporary audio chunks
os.makedirs(TEMP_DIR, exist_ok=True)


# 3. Determine the total duration (in seconds) using ffprobe
def get_video_duration(file_path):
    """Use ffprobe to get the duration of the video file in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    output = subprocess.check_output(cmd).decode().strip()
    return float(output)


total_duration = get_video_duration(VIDEO_FILE)
print(f"Total video duration: {total_duration} seconds")

# 4. Calculate how many chunks we need
num_chunks = math.ceil(total_duration / CHUNK_LENGTH_SECONDS)

# 5. Split the original video into chunks using ffmpeg
chunk_file_paths = []
for i in range(num_chunks):
    start_time = i * CHUNK_LENGTH_SECONDS
    output_file = os.path.join(TEMP_DIR, f"chunk_{i}.mp4")
    chunk_file_paths.append(output_file)

    # ffmpeg command to extract a portion of the video
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if exists
        "-i", VIDEO_FILE,
        "-ss", str(start_time),
        "-t", str(CHUNK_LENGTH_SECONDS),
        "-c", "copy",  # Copy codec to avoid re-encoding (faster split).
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"Created chunk: {output_file}")

# 6. Transcribe each chunk and concatenate results
full_transcription = ""

for idx, chunk_path in enumerate(chunk_file_paths):
    print(f"Transcribing chunk {idx + 1}/{num_chunks}...")
    with open(chunk_path, "rb") as audio_file:
        # Use the new `openai.Audio` methods:
        # https://platform.openai.com/docs/guides/speech-to-text
        # If you're using openai < 0.27.0, adjust accordingly.
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language='en'
        )
        chunk_text = response.text
        full_transcription += chunk_text + " "

# 7. Print or save the final transcription
print("\nFinal Transcription:\n")
print(full_transcription.strip())

chunks = chunk_gpt_tokens(full_transcription.strip(), chunk_size=50, overlap=25)
for chunk in chunks:
  upsert_to_pinecone(chunk['chunk_text'], metadata={'video_id': VIDEO_FILE, 'category': 'Education'})