import pyttsx3
import io
import streamlit as st
import tempfile
import os
import base64

# Improved Text-to-Speech with voice options and audio file output
def speak(text, rate=150, volume=1.0, voice_gender="female"):
    if not text:
        return None
        
    try:
        engine = pyttsx3.init()
        
        # Set speech rate (words per minute)
        engine.setProperty('rate', rate)
        
        # Set volume (0.0 to 1.0)
        engine.setProperty('volume', volume)
        
        # Set voice based on gender preference
        voices = engine.getProperty('voices')
        if voice_gender.lower() == "female":
            female_voices = [v for v in voices if 'female' in v.name.lower()]
            if female_voices:
                engine.setProperty('voice', female_voices[0].id)
        else:
            male_voices = [v for v in voices if 'male' in v.name.lower()]
            if male_voices:
                engine.setProperty('voice', male_voices[0].id)
        
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_file_path = tmp_file.name
        
        # Save speech to file instead of playing it
        engine.save_to_file(text, temp_file_path)
        engine.runAndWait()
        
        # Read the audio file and return as bytes
        with open(temp_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return audio_bytes
        
    except Exception as e:
        st.error(f"Error in text-to-speech: {str(e)}")
        return None
        
# Function to generate auto-play audio HTML
def autoplay_audio(audio_bytes):
    if audio_bytes:
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            </audio>
        """
        return audio_html
    return None