# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

st.title("📚 Academic Research Assistant (LLM)")
st.markdown("Use this chatbot to summarize abstracts, generate research questions, or format APA citations.")

@st.cache_resource
def load_model():
    base_model = "mistralai/Mistral-7B-Instruct-v0.1"
    adapter_repo = "Dishara/mistral-finetuned-academic"

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_repo, is_trainable=False)
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Enter your research prompt:", height=200)

if st.button("💬 Generate Response") and prompt:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.9)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    st.success("🧠 Model Response:")
    st.write(decoded)

