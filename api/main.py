from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from api.routes import disease, lab, qa, ocr
from core.orchestrator import HealthOrchestrator
from llm.ollama_client import OllamaClient
import easyocr

app = FastAPI(title="HealthGuard AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(disease.router)
app.include_router(lab.router)
app.include_router(qa.router)
app.include_router(ocr.router)

_orchestrator = HealthOrchestrator()
_llm = OllamaClient()

print("Loading EasyOCR...")
_ocr_reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR ready")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok", "modules": {"disease_predictor": "ready", "lab_analyzer": "ready", "medical_qa": "ready", "ocr": "ready"}}

@app.post("/reset")
def reset():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = _orchestrator.handle(req.message)
        return {"response": result.get("response", ""), "module_used": result.get("module_used", "General Assistant"), "confidence": result.get("confidence", 0.0), "raw_result": result.get("raw_result", {}), "disclaimer": "Not a substitute for professional medical advice."}
    except Exception as e:
        return {"response": f"Error: {str(e)}", "module_used": "Error", "confidence": 0.0, "raw_result": {}, "disclaimer": "Not a substitute for professional medical advice."}

@app.post("/chat/upload")
async def chat_with_file(message: str = Form(default=""), file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower()
        extracted_text = ""

        if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                import numpy as np
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                results = _ocr_reader.readtext(np.array(image))
                lines = [text for (_, text, conf) in results if conf > 0.1]
                extracted_text = "\n".join(lines) or "[Could not extract text]"
            except Exception as ex:
                extracted_text = f"[EasyOCR error: {str(ex)}]"

        elif filename.endswith(".pdf"):
            try:
                import pdfplumber, io
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    extracted_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as ex:
                extracted_text = f"[PDF error: {str(ex)}]"

        elif filename.endswith((".txt", ".csv")):
            extracted_text = file_bytes.decode("utf-8", errors="ignore")

        else:
            extracted_text = "[Unsupported file type]"

        if extracted_text.startswith("["):
            return {"response": f"Could not read the file.\n\n{extracted_text}", "module_used": "OCR Reader", "confidence": 0.0, "raw_result": {}, "disclaimer": "Not a substitute for professional medical advice."}

        clean_text = "\n".join([l.strip() for l in extracted_text.split("\n") if l.strip()][:50])
        prompt = f"You are a medical assistant. A patient uploaded their prescription. Explain in simple English: what condition they likely have, each medicine and its use, when/how to take each medicine, and any special instructions.\n\nPRESCRIPTION TEXT:\n{clean_text}\n\nGive a clear, friendly explanation."
        response_text = _llm.chat(prompt)

        if not response_text or response_text.startswith("LLM error"):
            response_text = f"Extracted text from prescription:\n\n{clean_text}\n\nPlease show this to your pharmacist."

        return {"response": response_text, "module_used": "EasyOCR + Prescription Reader", "confidence": 0.95, "raw_result": {"extracted_text": extracted_text[:500]}, "disclaimer": "Not a substitute for professional medical advice."}

    except Exception as e:
        return {"response": f"Error: {str(e)}", "module_used": "Error", "confidence": 0.0, "raw_result": {}, "disclaimer": "Not a substitute for professional medical advice."}
