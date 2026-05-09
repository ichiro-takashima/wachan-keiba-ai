from openai import OpenAI
import google.generativeai as genai


TARGET_MODEL = "gemini-2.5-flash"


def ask_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(TARGET_MODEL)
    response = model.generate_content(prompt)
    return response.text


def ask_chatgpt(prompt, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
