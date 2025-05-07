import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
os.environ["HF_TOKEN"] = "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"

st.set_page_config(page_title="Academic Research Chatbot", layout="centered")
st.title("📚 Academic Research Chatbot (Mistral + LoRA)")

@st.cache_resource
def load_model():
    base_model = "mistralai/Mistral-7B-Instruct-v0.1"
    adapter_repo = "Dishara/mistral-finetuned-academic"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_repo, adapter_name="default")

    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_area("Enter your academic question or abstract:")

if st.button("Get Answer") and user_input:
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    st.markdown("### 🧠 Response:")
    st.write(decoded_output)
