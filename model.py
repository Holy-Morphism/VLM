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
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
            "ybelkada/blip2-opt-2.7b-fp16-sharded",
            device_map="auto",
            load_in_8bit=True
        )
   
        
    return processor, model

processor, model = load_model()

def generate_answer(image: Image.Image, question: str) -> str:
    if not image.mode == "RGB":
        image = image.convert("RGB")

    inputs = processor(image, question, return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**inputs, max_length=50)
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return answer