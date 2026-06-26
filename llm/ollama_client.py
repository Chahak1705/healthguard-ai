# llm/ollama_client.py

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi3:mini"
SYSTEM_PROMPT = """
You are HealthGuard AI, an intelligent medical assistant created by a student team.
You are NOT developed by Microsoft, Google, or OpenAI.
You help users with:
1. Disease prediction from symptoms
2. Lab report interpretation
3. General medical Q&A

Rules:
- Never diagnose definitively
- Always say "may indicate" or "could suggest"
- Keep responses clear and easy to understand
- Always remind users to consult a real doctor
- Be empathetic and supportive
- Detect the language the user is writing in and reply in the same language
- If user writes in Hindi, respond in Hindi
- If user writes in Hinglish, respond in Hinglish
- If user writes in English, respond in English
"""


class OllamaClient:

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []

    def is_running(self) -> bool:
        try:
            requests.get("http://localhost:11434", timeout=3)
            return True
        except:
            return False

    def chat(self, user_message: str) -> str:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            data = response.json()
            if "error" in data:
                return user_message
            reply = data["message"]["content"]
            return reply
        except Exception as e:
            return f"LLM error: {str(e)}"

    def extract_intent(self, user_message: str) -> dict:
        prompt = f"""
Analyze this user message and return ONLY a valid JSON object, nothing else.
No explanation. No markdown. Just raw JSON.

User message: "{user_message}"

Return exactly this format:
{{
    "module": "disease_predictor" or "lab_analyzer" or "qa_engine" or "general",
    "symptoms": ["symptom1", "symptom2"] or [],
    "lab_test": "test_name" or null,
    "lab_value": 0.0 or null,
    "lab_sex": "male" or "female" or null,
    "question": "extracted question" or null
}}

Rules:
- Use disease_predictor if user mentions body symptoms like fever, pain, headache
- Use lab_analyzer if user mentions a lab test name with a numeric value
- Use qa_engine if user asks a medical question
- Use general for greetings or unclear messages
"""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            data = response.json()
            if "error" in data:
                return {
                    "module": "general",
                    "symptoms": [],
                    "lab_test": None,
                    "lab_value": None,
                    "lab_sex": None,
                    "question": None
                }
            raw = data["message"]["content"]
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except:
            return {
                "module": "general",
                "symptoms": [],
                "lab_test": None,
                "lab_value": None,
                "lab_sex": None,
                "question": None
            }

    def format_disease_result(self, result: dict, original_message: str) -> str:
        try:
            diseases = result.get("diseases", [])
            if not diseases:
                return "Could not identify a specific condition. Please consult a doctor."
            top = diseases[0]
            name = top.get("name", "unknown").title()
            conf = round(top.get("confidence", 0) * 100, 1)
            others = ", ".join(d.get("name", "").title() for d in diseases[1:3])
            response = f"Based on your symptoms, the most likely condition is **{name}** (confidence: {conf}%)."
            if others:
                response += f" Other possibilities include {others}."
            response += " Please consult a doctor for proper diagnosis and treatment."
            return response
        except:
            return "Analysis complete. Please consult a doctor for proper diagnosis."

    def format_lab_result(self, result: dict, original_message: str) -> str:
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"""
User said: "{original_message}"
Lab analysis returned:
{json.dumps(result, indent=2)}
Write a clear, caring 3-4 sentence response explaining what this result means.
If status is CRITICALLY LOW or CRITICALLY HIGH, use urgent but calm language.
End with: "Please consult your doctor to discuss these results."
"""}],
                "stream": False
            }
            data = requests.post(OLLAMA_URL, json=payload, timeout=60).json()
            if "error" in data:
                return "Lab analysis complete. Please consult your doctor to discuss these results."
            return data["message"]["content"]
        except:
            return "Lab analysis complete. Please consult your doctor to discuss these results."

    def format_qa_result(self, answer: str, original_message: str) -> str:
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": f"""
User asked: "{original_message}"
Retrieved answer: "{answer}"
Rephrase this as a warm, helpful 2-3 sentence response.
Make it easy to understand for a non-medical person.
"""}],
                "stream": False
            }
            data = requests.post(OLLAMA_URL, json=payload, timeout=60).json()
            if "error" in data:
                return answer
            return data["message"]["content"]
        except:
            return answer

