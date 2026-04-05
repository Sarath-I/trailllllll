import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text Snapshot Studio", page_icon="📝", layout="centered")

st.title("Text Snapshot Studio")
st.write("Paste any long text below and get a neat summary instantly.")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def split_text_into_chunks(text, chunk_size=350):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_long_text(text, min_len, max_len):
    chunks = split_text_into_chunks(text)

    partial_summaries = []
    for chunk in chunks:
        output = summarizer(
            chunk,
            min_length=min_len,
            max_length=max_len,
            do_sample=False,
            truncation=True
        )
        partial_summaries.append(output[0]["summary_text"])

    merged_text = " ".join(partial_summaries)

    if len(partial_summaries) > 1:
        final_output = summarizer(
            merged_text,
            min_length=min_len,
            max_length=max_len,
            do_sample=False,
            truncation=True
        )
        return final_output[0]["summary_text"]

    return merged_text

user_text = st.text_area(
    "Enter your text here",
    height=250,
    placeholder="Paste article text, notes, or any long paragraph..."
)

min_length = st.slider("Minimum summary length", 20, 100, 30)
max_length = st.slider("Maximum summary length", 50, 200, 80)

if st.button("Generate Summary"):
    clean_text = user_text.strip()

    if len(clean_text) < 50:
        st.warning("Please enter a longer text for a better summary.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_long_text(clean_text, min_length, max_length)

        st.subheader("Summary")
        st.success(summary)
