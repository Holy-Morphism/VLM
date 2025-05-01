import streamlit as st
import torch
from transformers import BlipProcessor, Blip2ForConditionalGeneration
from PIL import Image

# Load model & processor once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    # Ensure model is in evaluation mode
    model.eval()
    return processor, model

processor, model = load_model()

def generate_caption(image, question: str) -> str:
    try:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image.Image type for 'image' parameter")
            
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # Ensure consistent image size to avoid shape mismatches
        image = image.resize((512, 512))
            
        # Process inputs on CPU and handle device properly
        inputs = processor(image, question, return_tensors="pt")
        
        # Check if inputs contain valid data
        if inputs['pixel_values'].shape[0] == 0 or inputs['input_ids'].shape[0] == 0:
            return "Error: Failed to process the image or question. Please try again."
        
        # Generate caption with proper error handling
        with torch.no_grad():  # Disable gradient calculation for inference
            try:
                out = model.generate(
                    **inputs, 
                    max_length=100,  # Increased for more detailed responses
                    num_beams=3,     # Beam search for better quality
                    temperature=0.7, # Add some variability
                    do_sample=True
                )
            except RuntimeError as e:
                if "shape mismatch" in str(e):
                    return "Sorry, there was an issue processing this image. Try a different image or resize it."
                raise e
        
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # If answer is empty or too short, provide a fallback
        if not answer or len(answer) < 5:
            return "I couldn't generate a good answer. Could you rephrase your question?"
            
        return answer
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# Make functions available at module level
__all__ = ['load_model', 'generate_caption']