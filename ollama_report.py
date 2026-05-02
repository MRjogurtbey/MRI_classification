import requests

CLASS_DESCRIPTIONS = {
    "glioma": "Glioma (beyin ve omurilikte bulunan glial hücre kaynaklı tümör)",
    "meningioma": "Meningioma (beyin zarlarından kaynaklanan genellikle iyi huylu tümör)",
    "notumor": "Tümör tespit edilmedi (normal beyin dokusu görünümü)",
    "pituitary": "Pituitary tümör (hipofiz bezi kaynaklı tümör)",
}

PROMPT_TEMPLATE = """Sen bir radyoloji asistanısın. Bir yapay zeka MRI sınıflandırma modelinin çıktısına dayanarak doktora yönelik kısa ve profesyonel bir ön rapor özeti hazırlıyorsun.

Model Tahmini: {description}

Lütfen şu kurallara uy:
- 3-4 cümle yaz.
- Tıbbi terminoloji kullan ama anlaşılır ol.
- Kesin tanı koyma, bu bir ön değerlendirmedir.
- Türkçe yaz.

Ön Rapor:"""


def generate_report(predicted_class: str, ollama_model: str = "llama3") -> str:
    description = CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)
    prompt = PROMPT_TEMPLATE.format(description=description)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": ollama_model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "Ollama servisi çalışmıyor. Lütfen terminalde `ollama serve` komutunu çalıştırın."
    except requests.exceptions.Timeout:
        return "Ollama yanıt süresi aşıldı. Model yükleniyor olabilir, tekrar deneyin."
    except Exception as e:
        return f"Rapor oluşturulamadı: {e}"
