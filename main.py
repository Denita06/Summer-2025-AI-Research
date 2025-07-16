import requests
import gradio as gr # creates web interface
import re
from datetime import datetime
from medquad_loader import load_medquad, retrieve_relevant_answers


# Load MedQuAD data at startup
medquad_df = load_medquad("Kaggle - MedQuAD/medquad.csv")

# Get nearby healthcare providers using OpenStreetMap Nomination API
def get_healthcare_providers(city, max_results=5):
    facility_types = {"hospital", "clinic", "urgent care", "doctor office", "pharmacy"}
    headers = {"User-Agent": "AI-Medical-Chatbot"}
    providers = []

    for facility in facility_types:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{facility} in {city}",
            "format": "json",
            "limit": max_results
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            for place in data:
                name = place.get("display_name")
                if name not in providers:
                    providers.append(name)
        except Exception as e:
            print(f"Error getting {facility}: {e}")
    return providers

# chatbot function that processes user inputs and calls the DeepSeek model
def ask_chatbot(history, age, gender, symptoms, other_symptom, question, city):
    if age < 18:
        history.append({"role": "assistant", "content": "Sorry, this chatbot is only for users 18 and older."})
        return history, history, ""

    if "Other" in symptoms and other_symptom:
        symptoms.append(other_symptom)

    symptoms_str = ", ".join(symptoms)
    query_text = question + "" + symptoms_str

    relevant_answers = retrieve_relevant_answers(medquad_df, query_text, symptoms)
    retrieval_context = "\n".join(relevant_answers) if relevant_answers else "No relevant MedQuAD data found."

    conversation = ""
    for msg in history:
        conversation += f"{msg['role']}: {msg['content']}\n"

    prompt = (
        f"The user is a {age}-year-old {gender} experiencing {symptoms_str}."
        f"They ask: {question}. "
        f"Here is some relevant medical information retrieved from MedQuAD:\n{retrieval_context}"
        "Only use this MedQuAD information to answer the user's question."
        "If the information does the answer their question, reply: 'I could not find a specific answer to your question in the MedQuAD dataset."
        "Please think step by step to arrive at the best answer internally. "
        "Do not show your internal reasoning or thought process. "
        "Speak directly to the user in your response with detailed, clear, and actionable medical advice including possible causes, what they can do today, and when to see a doctor."
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

        thoughts = re.findall(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        user_answer = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

        for thought in thoughts:
            print("AI Internal Thought:", thought.strip())
            try:
                with open("ai_thought_log.txt", "a") as log_file:
                    log_file.write("\n" + "="*60 + "\n")
                    log_file.write(f"Timestamp: {datetime.now()}\n")
                    log_file.write("AI Internal Thought Process:\n")
                    log_file.write(thought.strip() + "\n")
                    log_file.write("="*60 + "\n")
            except Exception as e:
                print("Logging error:", e)

        bot_reply = user_answer if user_answer else "Sorry, I could not generate a clear response."

    except Exception as e:
        bot_reply = f"Error contacting Deepseek model: {e}"

    providers_text = ""
    if city.strip():
        providers = get_healthcare_providers(city)
        if providers:
            providers_text = "\n\n**Nearby healthcare providers based on your city:**\n"
            for i, p in enumerate(providers, 1):
                providers_text += f"{i}. {p}\n"
        else:
            providers_text = "\n\n**Nearby Healthcare Providers:**\nSorry, no nearby healthcare providers could be found for your city."

    final_reply = f"{bot_reply}{providers_text}"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": final_reply})
    return history, history, ""


# Builds the Gradio user interface
if __name__ == "__main__":
    with gr.Blocks(title="AI Medical Assistant Chatbot (DeepSeek)") as demo:

        chatbot = gr.Chatbot(label="AI Medical Assistant Chatbot", type="messages")
        state = gr.State([]) # stores conversation history

        # Input for user's age (must be >= 18)
        with gr.Row():
            age_input = gr.Number(label="Your Age (18+)", minimum=18, value=18)
            gender_input = gr.Radio(["Male", "Female"], label="Your Gender")
            city_input = gr.Textbox(label="Your City", placeholder="Enter your city or location")

        # Checkbox to allow user to select multiple symptoms if needed
        symptom_input = gr.CheckboxGroup(
            ["Fever","Cough","Headache","Fatigue","Rash","Other"],
            label="Select Symptoms"
        )

        # If the user selects 'other' in the symptoms list, they have the option to describe their symptoms
        other_symptom_input = gr.Textbox(label="Describe other symptoms", visible=False)
        question_input = gr.Textbox(label="Your Question", placeholder="Describe your main concern today...")

        # if "Other" in symptoms is checked, then a textbox will appear
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
            inputs=[state, age_input, gender_input, symptom_input, other_symptom_input, question_input, city_input],
            outputs=[chatbot, state, question_input]
        )

    demo.launch()
