import streamlit as st
from PIL import Image
import io
import os
import time
from model import load_model, generate_caption
from stt import transcribe_audio
from tts import speak, autoplay_audio

# Page configuration
st.set_page_config(
    page_title="Pici-Talk: Visual Question Answering",
    page_icon="🧠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .subheader {
        font-size: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #0D47A1;
    }
    .user-message {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #BBDEFB;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton button {
        background-color: #1976D2;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'tts_settings' not in st.session_state:
    st.session_state.tts_settings = {"rate": 150, "volume": 1.0, "voice_gender": "female"}

# Load model
@st.cache_resource
def get_model():
    with st.spinner("Loading VLM model... This might take a minute."):
        return load_model()

# Function to handle image processing
def process_image(uploaded_file):
    try:
        if uploaded_file is None:
            return None
            
        image = Image.open(uploaded_file)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Resize if image is too large to save memory
        max_size = 1000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# VQA inference with better error handling
def answer_question(image, question):
    if image is None:
        return "Please upload an image first."
    if not question:
        return "Please ask a question about the image."
        
    try:
        with st.spinner("Thinking about your question..."):
            return generate_caption(image, question)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't process your question. Please try again."

# Main application UI
st.markdown("<h1 class='main-header'>🧠 Pici-Talk: Interactive Visual QA</h1>", unsafe_allow_html=True)

# Main content area - use columns for better layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='subheader'>🖼️ Image Upload</h2>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    # Process the image when uploaded
    if uploaded_image is not None and (not st.session_state.image_uploaded or 
                                      uploaded_image.name != getattr(st.session_state.get('last_uploaded'), 'name', None)):
        with st.spinner("Processing image..."):
            st.session_state.current_image = process_image(uploaded_image)
            st.session_state.image_uploaded = True
            st.session_state.last_uploaded = uploaded_image
            st.session_state.conversation = []  # Reset conversation for new image
    
    # Display the current image
    if st.session_state.image_uploaded and st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="Current Image", use_container_width=True)
    
    # Camera input option
    st.markdown("<h3>Or take a photo with your camera</h3>", unsafe_allow_html=True)
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None and (not st.session_state.image_uploaded or 
                                    'camera' not in st.session_state or 
                                    st.session_state.camera != camera_input):
        with st.spinner("Processing camera image..."):
            st.session_state.current_image = process_image(camera_input)
            st.session_state.image_uploaded = True
            st.session_state.camera = camera_input
            st.session_state.conversation = []  # Reset conversation for new image

with col2:
    if st.session_state.image_uploaded and st.session_state.current_image is not None:
        st.markdown("<h2 class='subheader'>🎤 Speak your question about the image</h2>", unsafe_allow_html=True)
        
        # Voice input - now the only option
        col2a, col2b = st.columns([3, 1])
        with col2a:
            # Correct implementation of audio_input
            audio_bytes = st.audio_input("Record your question")
        
        with col2b:
            if st.button("Process Voice", key="process_audio"):
                if audio_bytes:
                    with st.spinner("Transcribing audio..."):
                        question_text = transcribe_audio(audio_bytes)
                        if question_text:
                            st.success(f"Transcribed: '{question_text}'")
                            
                            # Add user message to conversation
                            st.session_state.conversation.append(("user", question_text))
                            
                            # Generate answer
                            answer = answer_question(st.session_state.current_image, question_text)
                            
                            # Add assistant response to conversation
                            st.session_state.conversation.append(("assistant", answer))
                            
                            # Generate speech
                            st.session_state.audio_bytes = speak(
                                answer, 
                                rate=st.session_state.tts_settings["rate"],
                                volume=st.session_state.tts_settings["volume"],
                                voice_gender=st.session_state.tts_settings["voice_gender"]
                            )
                        else:
                            st.error("Could not transcribe audio. Please try again.")
                else:
                    st.warning("No audio recorded. Please record audio first.")
        
        # Controls for conversation
        if st.button("Reset Conversation"):
            st.session_state.conversation = []
            st.session_state.audio_bytes = None
        
        # Autoplay the generated audio if available
        if st.session_state.audio_bytes:
            audio_html = autoplay_audio(st.session_state.audio_bytes)
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
            st.audio(st.session_state.audio_bytes)

# Create a container for chat history at the bottom
st.markdown("<h2 class='subheader'>📝 Conversation History</h2>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    if not st.session_state.conversation:
        st.info("Your conversation will appear here. Upload an image and speak your question to get started!")
    
    for i, (role, text) in enumerate(st.session_state.conversation):
        if role == "user":
            st.markdown(f"<div class='user-message'><b>You:</b> {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-message'><b>Assistant:</b> {text}</div>", unsafe_allow_html=True)
            
            # Add a speak button for each assistant response
            col_speak, col_copy = st.columns([1, 9])
            with col_speak:
                if st.button(f"🔊", key=f"speak_{i}"):
                    audio_bytes = speak(
                        text, 
                        rate=st.session_state.tts_settings["rate"],
                        volume=st.session_state.tts_settings["volume"],
                        voice_gender=st.session_state.tts_settings["voice_gender"]
                    )
                    if audio_bytes:
                        st.session_state.audio_bytes = audio_bytes
                        st.experimental_rerun()
