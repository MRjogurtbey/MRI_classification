"""
Medical report generation with multi-backend support.
Priority: Ollama (local) → Groq (cloud) → Gemini (cloud) → fallback message
"""

import os
import requests

_DESCRIPTIONS = {
    "glioma":      "Glioma (beyin/omurilikte glial hücre kaynaklı malign tümör)",
    "meningioma":  "Meningioma (beyin zarlarından kaynaklanan genellikle iyi huylu tümör)",
    "notumor":     "Tümör tespit edilmedi (normal beyin dokusu görünümü)",
    "pituitary":   "Pituitary tümör (hipofiz bezi kaynaklı tümör)",
}

_PROMPT = """Sen bir radyoloji asistanısın. Yapay zeka MRI sınıflandırma sonucuna dayanarak doktora yönelik kısa, profesyonel bir ön rapor hazırlıyorsun.

Model Tahmini: {description}

Kurallar:
- 3-4 cümle, Türkçe
- Tıbbi terminoloji kullan ama anlaşılır ol
- Kesin tanı koyma, "ön değerlendirme" olduğunu belirt

Ön Rapor:"""


def _get_api_key(key_name: str) -> str | None:
    """Check Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        return st.secrets.get(key_name)
    except Exception:
        return os.environ.get(key_name)


def _via_ollama(prompt: str, model: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _via_groq(prompt: str, model: str = "llama3-8b-8192") -> str:
    from groq import Groq
    api_key = _get_api_key("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY bulunamadı")
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()


def _via_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str:
    import google.generativeai as genai
    api_key = _get_api_key("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY bulunamadı")
    genai.configure(api_key=api_key)
    response = genai.GenerativeModel(model).generate_content(prompt)
    return response.text.strip()


def generate_report(
    predicted_class: str,
    ollama_model: str = "llama3",
    backend: str = "auto",
) -> str:
    """
    Generate a Turkish medical pre-report.

    backend: "auto" | "ollama" | "groq" | "gemini"
    - "auto": tries Ollama → Groq → Gemini in order
    """
    key = predicted_class.lower().replace(" ", "")
    description = _DESCRIPTIONS.get(key, predicted_class)
    prompt = _PROMPT.format(description=description)

    backends = (
        ["ollama", "groq", "gemini"] if backend == "auto"
        else [backend]
    )

    errors = []
    for b in backends:
        try:
            if b == "ollama":
                return _via_ollama(prompt, ollama_model)
            elif b == "groq":
                return _via_groq(prompt)
            elif b == "gemini":
                return _via_gemini(prompt)
        except requests.exceptions.ConnectionError:
            errors.append("Ollama: servis çalışmıyor")
        except Exception as e:
            errors.append(f"{b}: {e}")

    return (
        "AI raporu şu an kullanılamıyor.\n"
        "• Yerel demo için: `ollama serve` çalıştırın\n"
        "• Bulut için: Streamlit Secrets'a GROQ_API_KEY veya GEMINI_API_KEY ekleyin\n"
        f"(Detay: {' | '.join(errors)})"
    )
