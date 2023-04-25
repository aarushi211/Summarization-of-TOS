import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch


@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = PegasusTokenizer.from_pretrained('nsi319/legal-pegasus')
    model = PegasusForConditionalGeneration.from_pretrained(
        "arjav/TOS-Pegasus", use_cuda = False)
    model.args.use_multiprocessing = False

    return tokenizer, model


tokenizer, model = get_model()

user_input = st.text_area('Enter Terms of Service to summarize')
button = st.button("Summarize")

if user_input and button:
    input_tokenized = tokenizer.encode(
        [user_input], return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=50,
                                 max_length=150,
                                 early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for g in summary_ids][0]
    st.write(summary)
