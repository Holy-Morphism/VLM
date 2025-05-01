import tempfile
import os
from groq import Groq
client = Groq()

# Whisper ASR via Groq API
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_bytes.name)[1]) as tmp_file:
                    tmp_file.write(audio_bytes.getvalue())
                    tmp_file_path = tmp_file.name
                
    try:
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3-turbo",
                prompt="Specify context or spelling",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        return transcription.text

    finally:
        # Clean up the temporary audio file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)