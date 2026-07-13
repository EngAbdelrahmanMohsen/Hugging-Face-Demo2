from transformers import pipeline
import gradio as gr

model = pipeline("summarization")

# In app.py
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct")

text_to_summarize = "Your long text goes here..."

messages = [
    {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
    {"role": "user", "content": f"Summarize the following text:\n\n{text_to_summarize}"}
]

# 3. Generate response
outputs = pipe(messages, max_new_tokens=150)
summary = outputs[0]["generated_text"][-1]["content"]

print(summary)

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

demo.launch()