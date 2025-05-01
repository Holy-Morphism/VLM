import streamlit as st
import torch
from peft import PeftModel
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from PIL import Image

path_to_adapters = './blip2_finetuned'

# Load model & processor once
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Load base model without GPU optimizations
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="cpu",
        load_in_8bit=False
    )
    
    # Load the fine-tuned adapters
    model = PeftModel.from_pretrained(base_model, path_to_adapters)
    
    # Ensure model is on CPU
    model = model.to("cpu")
        
    return processor, model

processor, model = load_model()

def generate_caption(image: Image.Image, question: str) -> str:
    try:
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        # Process inputs on CPU
        inputs = processor(image, question, return_tensors="pt")
        
        # Generate caption with a timeout and parameter tuning
        generated_ids = model.generate(
            **inputs, 
            max_length=100,  # Increased for more detailed responses
            num_beams=3,     # Beam search for better quality
            temperature=0.7, # Add some variability
            do_sample=True
        )
        
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # If answer is empty or too short, provide a fallback
        if not answer or len(answer) < 5:
            return "I couldn't generate a good answer. Could you rephrase your question?"
            
        return answer
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Sorry, I encountered an error while processing your question."