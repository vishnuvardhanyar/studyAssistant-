
import os
from google import genai
from google.genai import types
import gradio as gr

client = genai.Client(api_key=os.getenv("GEMINI_APIKEY"))

personalities = {
    "Friendly": """
You are a friendly, enthusiastic, and highly encouraging Study Assistant.
Your goal is to break down complex concepts into simple, beginner-friendly explanations.
Use analogies and real-world examples that beginners can relate to.
Always ask a follow-up question to check understanding.
""",

    "Academic": """
You are a strictly academic, highly detailed, and professional university Professor.
Use precise, formal terminology and structure your response.
Break down complex concepts into simple explanations.
Use analogies and real-world examples when helpful.
Always ask a follow-up question to check understanding.
"""
}


def study_assistant(question, persona):
    try:
        system_prompt = personalities[persona]

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.4,
                max_output_tokens=500,
            ),
            contents=question,
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.Interface(
    fn=study_assistant,
    inputs=[
        gr.Textbox(
            label="Question",
            lines=4,
            placeholder="Ask a question..."
        ),
        gr.Radio(
            choices=list(personalities.keys()),
            value="Friendly",
            label="Personality"
        )
    ],
    outputs=gr.Textbox(
        label="Explanation",
        lines=10
    ),
    title="Study Assistant",
    description="Ask a question to get simple explanations with analogies and real-world examples."
)

if __name__ == "__main__":
    demo.launch(debug=True)
