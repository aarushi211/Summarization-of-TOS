import gradio as gr
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import re


def summarize(Terms):
    tokenizer = PegasusTokenizer.from_pretrained('nsi319/legal-pegasus')
    model = PegasusForConditionalGeneration.from_pretrained(
        "arjav/TOS-Pegasus")
    input_tokenized = tokenizer.encode(
        Terms, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(input_tokenized,
                                 num_beams=9,
                                 no_repeat_ngram_size=3,
                                 length_penalty=2.0,
                                 min_length=50,
                                 max_length=150,
                                 early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False) for g in summary_ids][0]
    
    points = re.split(r'(?<=[.?!])\s+', summary.strip())
    points = [f"• {point}" for point in points if point]  # format as bullet points

    return "\n".join(points)

    return summary


description = "Enter a Terms of Service document to summarize"
title = "Terms of Service Summarization"
interface = gr.Interface(fn=summarize,
                         inputs=gr.Textbox(
                             label="Terms of Service", lines=2, placeholder="Enter Terms of Service"),
                         outputs=gr.Textbox(label="Summary in Bullet Points"),
                         description=description,
                         title=title,
                         examples=[['account termination policy youtube will terminate a user s access to the service if under appropriate circumstances the user is determined to be a repeat infringer. youtube reserves the right to decide whether content violates these terms of service for reasons other than copyright infringement such as but not limited to pornography obscenity or excessive length. youtube may at any time without prior notice and in its sole discretion remove such content and or terminate a user s account for submitting such material in violation of these terms of service.']],
                         allow_flagging='never'
                         )


interface.launch()
