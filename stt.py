import tempfile
import os
from groq import Groq
import streamlit as st

# Initialize Groq client with API key check
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("GROQ_API_KEY environment variable not set. Speech recognition may not work.")
    return Groq(api_key=api_key)

client = get_groq_client()

# Whisper ASR via Groq API with improved error handling
def transcribe_audio(audio_bytes):
    if not audio_bytes:
        return None
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_bytes.name)[1]) as tmp_file:
        tmp_file.write(audio_bytes.getvalue())
        tmp_file_path = tmp_file.name
            
    try:
        with open(tmp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(tmp_file_path, file.read()),
                model="whisper-large-v3-turbo",
                prompt="Transcribe clearly with proper punctuation. Focus on questions about images.",
                response_format="json",
                language="en",
                temperature=0.0
            )
        
        text = transcription.text.strip()
        
        return text

    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None
    finally:
        # Clean up the temporary audio file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)