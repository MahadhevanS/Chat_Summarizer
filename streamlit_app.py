import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Helper Function for Model Loading ---
@st.cache_resource
def load_model_and_tokenizer():
    """
    Loads the fine-tuned model and tokenizer.
    Uses Streamlit's cache to avoid reloading on every rerun.
    """
    try:
        # Load the fine-tuned model and tokenizer from the saved directory
        tokenizer = AutoTokenizer.from_pretrained("models/chat_summarizer/fine-tuned-moderator")
        model = AutoModelForSeq2SeqLM.from_pretrained("models/chat_summarizer/fine-tuned-moderator")
        return tokenizer, model
    except OSError:
        st.error("Model files not found. Please ensure the fine-tuning script has been run and the 'fine-tuned-moderator' directory exists.")
        return None, None


# --- Main Logic for Summary Generation ---
def generate_summary(chat_log, loaded_tokenizer, loaded_model):
    """
    Generates a summary for a given chat log using the loaded model.
    """
    if not loaded_tokenizer or not loaded_model:
        return "Model not loaded. Please check the logs."

    inputs = loaded_tokenizer(chat_log, return_tensors="pt", max_length=1024, truncation=True)
    
    # Check for GPU availability and move tensors
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        loaded_model.to('cuda')

    summary_ids = loaded_model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return loaded_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Streamlit UI Components ---
def text_summarizer():

    tokenizer, model = load_model_and_tokenizer()
    
    # Text area for user input
    chat_input = st.text_area(
        "Paste the chat log below:",
        height=300,
        placeholder="Paste a chat conversation about a civic issue here..."
    )

    # Button to trigger summary generation
    if st.button("Generate Summary"):
        if chat_input:
            with st.spinner("Generating summary..."):
                summary_output = generate_summary(chat_input, tokenizer, model)
                st.markdown("---")
                st.success("Summary Generated!")
                st.markdown("### Generated Summary:")
                st.write(summary_output)
        else:
            st.warning("Please paste a chat log to generate a summary.")