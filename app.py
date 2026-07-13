from transformers import pipeline
import gradio as gr

model = pipeline("summarization")


# 1. Initialize your pipeline
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct")

# 2. DEFINE THE FUNCTION FIRST (Make sure it is named 'predict')
def predict(user_input):
    # If using a chat model structure:
    messages = [{"role": "user", "content": user_input}]
    outputs = pipe(messages, max_new_tokens=150)
    
    # Extract the text output depending on your model's format
    return outputs[0]['generated_text']

# 3. CALL THE INTERFACE AT THE BOTTOM
textbox = gr.Textbox(placeholder="Enter text to summarize...", lines=4)

gr.Interface(
    fn=predict,  # This matches the function name above perfectly now!
    inputs=textbox, 
    outputs="text"
).launch()