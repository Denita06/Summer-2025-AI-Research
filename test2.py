import requests
import gradio as gr
import json
import re

# Chatbot function that processes user inputs and calls the DeepSeek model
def ask_chatbot(age, gender, symptoms, other_symptom, question):
    if age < 18:
        return "Sorry, this chatbot is only for users 18 and older."

    if "Other" in symptoms and other_symptom:
        symptoms.append(other_symptom)

    symptoms_str = ", ".join(symptoms)

    # Stronger prompt enforcing strict JSON output
    prompt = (
        f"The user is a {age}-year-old {gender} experiencing {symptoms_str}. "
        f"They ask: {question}\n\n"
        "Please provide your output ONLY in valid strict JSON format, with no additional text, no tags, and no explanations. "
        "The format MUST be:\n"
        "{\n"
        "  \"thought\": \"Your internal reasoning here\",\n"
        "  \"answer\": \"Your final short and clear answer here\"\n"
        "}\n"
        "If your output is not in this exact JSON format, it will be rejected and not used."
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
        raw_output = response.json()["response"].strip()

        # Try parsing as JSON first
        try:
            output_json = json.loads(raw_output)
            thought = output_json["thought"]
            answer = output_json["answer"]
        except json.JSONDecodeError:
            # Fallback to regex extraction if not valid JSON
            thought_match = re.search(r'"thought":\s?"(.*?)"', raw_output, re.DOTALL)
            answer_match = re.search(r'"answer":\s?"(.*?)"', raw_output, re.DOTALL)
            if thought_match and answer_match:
                thought = thought_match.group(1)
                answer = answer_match.group(1)
            else:
                return f"Error: Could not parse LLM output as JSON or extract using regex. Output was:\n{raw_output}"

        # Optional: log thought process to a file
        with open("thought_logs.txt", "a") as log_file:
            log_file.write(f"Thought: {thought}\n\n")

        return answer

    except Exception as e:
        return f"Error contacting DeepSeek model: {e}"


# Builds the Gradio user interface
with gr.Blocks(title="AI Medical Assistant Chatbot (DeepSeek)") as demo:
    gr.Markdown("### AI Medical Assistant Chatbot (DeepSeek)")

    # Input for user's age (must be >= 18)
    age_input = gr.Number(label="Your Age", minimum=18, value=18)
    gender_input = gr.Radio(["Male", "Female"], label="Your Gender")

    # Checkbox to allow user to select multiple symptoms
    symptom_input = gr.CheckboxGroup(
        ["Fever", "Cough", "Headache", "Fatigue", "Rash", "Other"],
        label="Select Symptoms"
    )

    # If the user selects 'Other' in the symptoms list, show textbox
    other_symptom_input = gr.Textbox(label="Describe other symptoms", visible=False)
    question_input = gr.Textbox(label="Your Question")
    output = gr.Textbox(label="Chatbot Response")

    # Show 'other' textbox if 'Other' is checked
    def show_other_symptom(symptoms):
        return gr.update(visible="Other" in symptoms)

    symptom_input.change(
        show_other_symptom,
        inputs=symptom_input,
        outputs=other_symptom_input
    )

    # Submit Button
    submit_button = gr.Button("Submit")
    submit_button.click(
        ask_chatbot,
        inputs=[age_input, gender_input, symptom_input, other_symptom_input, question_input],
        outputs=output
    )

demo.launch()
