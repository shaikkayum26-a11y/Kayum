import google.generativeai as genai
import os

API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

model_gemini = genai.GenerativeModel("gemini-2.5-flash")
def get_gemini_response(field, problem, emotion, confidence):

    try:

        prompt = f"""
You are a helpful learning assistant.

A student studying {field} is feeling {emotion}
(confidence: {confidence:.1%}).

Student problem:

"{problem}"

Provide:

1. Acknowledge the student's emotion.
2. Give one subject-specific learning tip.
3. Encourage the student.

Keep the response short and simple.
"""

        response = model_gemini.generate_content(prompt)

        return response.text.strip()

    except Exception as e:

        return f"AI response unavailable: {e}"