import gradio as gr
from transformers import pipeline

# Safe to use the real model now—Hugging Face infrastructure can handle it!
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

def predict(user_input):
    messages = [
        {
            "role": "system", 
            "content": "You are a strict text-summarization bot. Output ONLY the concise summary. Do not write introductions or explanations."
        },
        {
            "role": "user", 
            "content": f"Summarize this text in one or two sentences: {user_input}"
        }
    ]
    
    outputs = pipe(messages, max_new_tokens=150)
    
    # Extract the clean assistant response text
    conversation = outputs[0]['generated_text']
    assistant_reply = conversation[-1]['content']
    
    return assistant_reply.strip()

textbox = gr.Textbox(placeholder="Enter text to summarize...", lines=4)

gr.Interface(
    fn=predict, 
    inputs=textbox, 
    outputs="text"
).launch()