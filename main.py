# chatbot_app.py

# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=None, force_download=True)

# Chatbot function
def ask_chatbot(question, age, symptom):
    if int(age) < 18:
        return "Sorry, this chatbot is for users 18 and older."

    prompt = f"I am a {age}-year-old experiencing {symptom}. {question}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.strip()

# Define Gradio UI
age_input = gr.Number(label="Your Age", minimum=18)
symptom_input = gr.Dropdown(["Fever", "Cough", "Fatigue", "Rash", "Headache", "Other"], label="Select a Symptom")
question_input = gr.Textbox(label="Your Question")

gr.Interface(
    fn=lambda age, symptom, question: ask_chatbot(question, age, symptom),
    inputs=[age_input, symptom_input, question_input],
    outputs="text",
    title="AI Medical Assistant Chatbot"
).launch()
