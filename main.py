import streamlit as st
from PIL import Image
from model import load_model, generate_answer
from stt import transcribe_audio
from tts import speak


processor, model = load_model()


# VQA inference
def answer_question(image: Image.Image, question: str) -> str:
    return generate_answer(image, question)

# Streamlit GUI
st.title("🧠 Pici-Talk")

# Audio recording
st.subheader("🎤 Record your question (up to 10 seconds)")
audio_bytes = st.audio_recorder(sample_rate=44100, duration=10)

if audio_bytes is not None:
    with st.spinner("Transcribing audio..."):
        question_text = transcribe_audio(audio_bytes.read())
    st.success(f"Transcribed Question: {question_text}")
else:
    question_text = st.text_input("Or type your question:")

# Image upload
st.subheader("🖼️ Upload an Image")
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None and question_text:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating answer..."):
        answer = answer_question(image, question_text)
    
    st.success(f"Answer: {answer}")

    if st.button("🔊 Speak Answer"):
        speak(answer)
