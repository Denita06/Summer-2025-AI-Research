import requests
import gradio as gr

def ask_chatbot(age, gender, symptoms, other_symptom, question):
    if age < 18:
        return "Sorry, this chatbot is only for users 18 and older."

    if "Other" in symptoms and other_symptom:
        symptoms.append(other_symptom)

    symptoms_str = ", ".join(symptoms)

    prompt = (
        f"The user is a {age}-year-old {gender} experiencing {symptoms_str}."
        f"They ask: {question}"
    )

    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error contacting Deepseek model: {e}"



with gr.Blocks(title="AI Medical Assistant Chatbot (DeepSeek)") as demo:
    gr.Markdown("### AI Medical Assistant Chatbot (DeepSeek)")

    age_input = gr.Number(label="Your Age", minimum=18, value=18)
    gender_input = gr.Radio(["Male", "Female"], label="Your Gender")

    symptom_input = gr.CheckboxGroup(
        ["Fever","Cough","Headache","Fatigue","Rash","Other"],
        label="Select Symptoms"
    )

    other_symptom_input = gr.Textbox(label="Describe other symptoms", visible=False)
    question_input = gr.Textbox(label="Your Question")
    output = gr.Textbox(label="Chatbot Response")

    def show_other_symptom(symptoms):
        return gr.update(visible="Other" in symptoms)

    symptom_input.change(
        show_other_symptom,
        inputs=symptom_input,
        outputs=other_symptom_input
    )

    submit_button = gr.Button("Submit")
    submit_button.click(
        ask_chatbot,
        inputs=[age_input, gender_input, symptom_input, other_symptom_input, question_input],
        outputs=output
    )


demo.launch()
