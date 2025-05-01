import streamlit as st
from PIL import Image
from model import load_model, generate_answer
from stt import transcribe_audio
from tts import speak

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'current_image' not in st.session_state:
    st.session_state.current_image = None

# Load model
@st.cache_resource
def get_model():
    return load_model()

processor, model = get_model()

# VQA inference
def answer_question(image: Image.Image, question: str) -> str:
    return generate_answer(image, question)

# Streamlit GUI
st.title("🧠 Pici-Talk")

# Image upload section
st.subheader("🖼️ Upload an Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Process the image when uploaded
if uploaded_image is not None and not st.session_state.image_uploaded:
    st.session_state.current_image = Image.open(uploaded_image).convert("RGB")
    st.session_state.image_uploaded = True
    st.session_state.conversation = []  # Reset conversation for new image
    st.experimental_rerun()

# Display the current image
if st.session_state.image_uploaded and st.session_state.current_image is not None:
    st.image(st.session_state.current_image, caption="Current Image", use_column_width=True)
    
    # Create a container for chat history
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        for i, (role, text) in enumerate(st.session_state.conversation):
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Assistant:** {text}")
                # Add a speak button for each assistant response
                if st.button(f"🔊 Speak", key=f"speak_{i}"):
                    speak(text)
    
    # Question input methods
    st.subheader("Ask a question about the image")
    
    # Method 1: Audio recording
    st.write("🎤 Record your question (up to 10 seconds)")
    audio_bytes = st.audio_recorder(sample_rate=44100, duration=10)
    
    if audio_bytes is not None:
        with st.spinner("Transcribing audio..."):
            question_text = transcribe_audio(audio_bytes)
            if question_text:
                # Add user message to conversation
                st.session_state.conversation.append(("user", question_text))
                
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer = answer_question(st.session_state.current_image, question_text)
                
                # Add assistant response to conversation
                st.session_state.conversation.append(("assistant", answer))
                
                # Rerun to update the UI
                st.experimental_rerun()
    
    # Method 2: Text input
    question_text = st.chat_input("Or type your question here:")
    if question_text:
        # Add user message to conversation
        st.session_state.conversation.append(("user", question_text))
        
        # Generate answer
        with st.spinner("Generating answer..."):
            answer = answer_question(st.session_state.current_image, question_text)
        
        # Add assistant response to conversation
        st.session_state.conversation.append(("assistant", answer))
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    # Option to reset the conversation or upload a new image
    if st.button("Reset conversation"):
        st.session_state.conversation = []
        st.experimental_rerun()
    
    if st.button("Upload a new image"):
        st.session_state.image_uploaded = False
        st.session_state.current_image = None
        st.session_state.conversation = []
        st.experimental_rerun()
