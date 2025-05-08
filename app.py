import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

# Title
st.set_page_config(page_title="Academic Research Chatbot", layout="centered")
st.title("🎓 AI Chatbot for Academic Research")
st.markdown("Ask research-related questions like summarizing abstracts, generating research questions, or APA citations.")

# Load model and tokenizer efficiently
@st.cache_resource
def load_model():
    base_model = "mistralai/Mistral-7B-Instruct-v0.1"
    adapter_repo = "Dishara/mistral-finetuned-academic"
    HF_TOKEN = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=False, token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_repo,
        token=HF_TOKEN,
    )

    model.eval()
    return tokenizer, model

# Load once and reuse
tokenizer, model = load_model()

# User Input
prompt = st.text_area("📘 Enter your academic prompt:")

if st.button("Generate Response") and prompt.strip():
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    st.markdown("### 🧠 Model Response:")
    st.write(response)


