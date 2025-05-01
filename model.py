import streamlit as st
import torch
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image

# Load model & processor once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    return processor, model

processor, model = load_model()

def generate_caption(image: Image.Image, question: str) -> str:
    try:
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        # Process inputs on CPU
        inputs = processor(image, question, return_tensors="pt")
        
        # Generate caption
        out = model.generate(
            **inputs, 
            max_length=100,  # Increased for more detailed responses
            num_beams=3,     # Beam search for better quality
            temperature=0.7, # Add some variability
            do_sample=True
        )
        
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # If answer is empty or too short, provide a fallback
        if not answer or len(answer) < 5:
            return "I couldn't generate a good answer. Could you rephrase your question?"
            
        return answer
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Sorry, I encountered an error while processing your question."