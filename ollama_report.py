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


def generate_report(predicted_class: str, ollama_model: str = "llama3") -> str:
    key = predicted_class.lower().replace(" ", "")
    description = _DESCRIPTIONS.get(key, predicted_class)
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": _PROMPT.format(description=description), "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Ollama servisi çalışmıyor. Terminalde `ollama serve` komutunu çalıştırın."
    except requests.exceptions.Timeout:
        return "Ollama yanıt vermedi. Model yükleniyor olabilir, tekrar deneyin."
    except Exception as e:
        return f"Rapor üretilemedi: {e}"
