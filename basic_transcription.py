import os
from openai import OpenAI

# Create an api client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Load audio file
audio_file = open("videos/five_transformations.mp4", "rb")

# Transcribe
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file,
  language='en'
)
# Print the transcribed text
print(transcription.text)