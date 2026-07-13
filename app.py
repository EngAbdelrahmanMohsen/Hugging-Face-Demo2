from transformers import pipeline
import gradio as gr

model = pipeline("summarization")


# 1. Initialize your pipeline
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct")

# 2. DEFINE THE FUNCTION FIRST (Make sure it is named 'predict')
def predict(user_input):
    # 1. Structure messages with a strict System Prompt telling it to summarize
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
        {"role": "user", "content": f"Summarize this text: {user_input}"}
    ]
    
    # 2. Generate the response
    outputs = pipe(messages, max_new_tokens=100)
    
    # 3. FIX: Extract ONLY the assistant's content from the response
    # The output structure is: [{'generated_text': [...]}]
    conversation = outputs[0]['generated_text']
    
    # Grab the very last message in the conversation (which is the assistant's reply)
    assistant_reply = conversation[-1]['content']
    
    return assistant_reply

# 3. CALL THE INTERFACE AT THE BOTTOM
textbox = gr.Textbox(placeholder="Enter text to summarize...", lines=4)

gr.Interface(
    fn=predict,  # This matches the function name above perfectly now!
    inputs=textbox, 
    outputs="text"
).launch()