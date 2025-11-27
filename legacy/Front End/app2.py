import gradio as gr
from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
import re

# Load model and tokenizer just once (outside the function)
model_name = "aarushi-211/TOS-Longformer"
model = LEDForConditionalGeneration.from_pretrained(model_name)
tokenizer = LEDTokenizer.from_pretrained(model_name)

def summarize_in_points(Terms):
    # Tokenize input
    input_tokenized = tokenizer.encode(
        Terms, return_tensors='pt', max_length=4096, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=50,
                                 max_length=150,
                                 early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Split into sentences using simple regex
    points = re.split(r'(?<=[.?!])\s+', summary.strip())
    points = [f"• {point}" for point in points if point]  # format as bullet points

    return "\n".join(points)

# Gradio interface
description = "Enter a Terms of Service document to summarize"
title = "Terms of Service Summarization"

interface = gr.Interface(
    fn=summarize_in_points,
    inputs=gr.Textbox(label="Terms of Service", lines=10, placeholder="Paste TOS text here..."),
    outputs=gr.Textbox(label="Summary in Bullet Points", lines=10),
    description=description,
    title=title,
    examples=[
        ["account termination policy youtube will terminate a user s access to the service if under appropriate circumstances the user is determined to be a repeat infringer. youtube reserves the right to decide whether content violates these terms of service for reasons other than copyright infringement such as but not limited to pornography obscenity or excessive length. youtube may at any time without prior notice and in its sole discretion remove such content and or terminate a user s account for submitting such material in violation of these terms of service."]
    ],
    allow_flagging='never'
)

interface.launch()
