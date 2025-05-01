import streamlit as st
import torch
from peft import PeftModel
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from PIL import Image

path_to_adapters = './blip2_finetuned'

# Load model & processor once
@st.cache_resource
def load_model():
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    try:
        if cuda_available:
            # Use 8-bit quantization with CUDA
            base_model = Blip2ForConditionalGeneration.from_pretrained(
                "ybelkada/blip2-opt-2.7b-fp16-sharded",
                device_map="auto",
                load_in_8bit=True
            )
        else:
            # Fall back to CPU with regular precision
            base_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",  # Using the standard model for CPU
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
        # Load the fine-tuned adapters if available
        try:
            model = PeftModel.from_pretrained(
                base_model,
                path_to_adapters,
                is_trainable=False
            )
            st.success("Loaded model with fine-tuned adapters")
        except Exception as e:
            st.warning(f"Could not load adapters: {e}. Using base model.")
            model = base_model
            
    except RuntimeError as e:
        st.error(f"Error loading model with 8-bit quantization: {e}")
        st.warning("Falling back to CPU with regular precision")
        # Fall back to CPU with regular precision
        base_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        model = base_model
        
    return processor, model

processor, model = load_model()

def generate_answer(image: Image.Image, question: str) -> str:
    if not image.mode == "RGB":
        image = image.convert("RGB")

    inputs = processor(image, question, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**inputs, max_length=50)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return answer