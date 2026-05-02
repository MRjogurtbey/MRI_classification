# 🚀 NeuroBridge AI - Deployment Guide

## Jüri için Online Demo Hazırlama Kılavuzu

---

## 🎯 Seçenek 1: Hugging Face Spaces (ÖNERİLEN)

### Avantajlar:
- ✅ Tamamen ÜCRETSIZ
- ✅ GPU desteği (ücretsiz!)
- ✅ Streamlit otomatik çalışır
- ✅ 5 dakikada hazır
- ✅ Public link otomatik oluşur

### Adımlar:

#### 1. Hugging Face Hesabı Aç
- https://huggingface.co/join
- Ücretsiz hesap oluştur

#### 2. New Space Oluştur
- https://huggingface.co/spaces
- "Create new Space" tıkla
- Space name: `neurobridge-ai-demo`
- SDK: **Streamlit** seç
- Hardware: **CPU basic** (ücretsiz)
- Visibility: **Public**

#### 3. Dosyaları Yükle
```bash
# Space repo'yu klonla
git clone https://huggingface.co/spaces/YOUR_USERNAME/neurobridge-ai-demo
cd neurobridge-ai-demo

# Gerekli dosyaları kopyala
cp ../MRI_classification/app.py .
cp ../MRI_classification/config.py .
cp ../MRI_classification/requirements.txt .
cp -r ../MRI_classification/utils .
cp -r ../MRI_classification/checkpoints .

# README ekle
echo "# NeuroBridge AI - MRI Tumor Classification Demo" > README.md

# Push et
git add .
git commit -m "Initial deployment"
git push
```

#### 4. Space Otomatik Build Olur
- 2-3 dakika bekle
- Link örneği: `https://YOUR_USERNAME-neurobridge-ai-demo.hf.space`

#### 5. Jüriye Link Ver
```
Streamlit Demo: https://YOUR_USERNAME-neurobridge-ai-demo.hf.space
```

---

## 🎯 Seçenek 2: Railway (FastAPI için)

### Avantajlar:
- ✅ Ücretsiz $5 credit (aylık)
- ✅ FastAPI çalışır
- ✅ Otomatik HTTPS
- ✅ GitHub entegrasyonu

### Adımlar:

#### 1. Railway Hesabı
- https://railway.app
- GitHub ile giriş yap

#### 2. New Project
- "Deploy from GitHub repo" seç
- `MRI_classification` repo'yu seç
- Branch: `feature/model-v3-dataset-expansion`

#### 3. Dockerfile Oluştur
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 4. Deploy
- Railway otomatik build eder
- Link örneği: `https://neurobridge-ai.railway.app`

---

## 🎯 Seçenek 3: Ngrok (Hızlı Geçici Çözüm)

### Avantajlar:
- ✅ 2 dakikada hazır
- ✅ Lokal server'ı internete açar
- ✅ Kurulum kolay

### Dezavantajlar:
- ⚠️ Bilgisayar açık kalmalı
- ⚠️ Geçici link (her seferinde değişir)

### Adımlar:

#### 1. Ngrok İndir
```bash
# Windows
https://ngrok.com/download

# Zip'i çıkar ve PATH'e ekle
```

#### 2. Streamlit için Tunnel Aç
```bash
# Terminal 1: Streamlit'i başlat
cd MRI_classification
.venv\Scripts\activate
streamlit run app.py

# Terminal 2: Ngrok tunnel aç
ngrok http 8501
```

#### 3. API için Tunnel Aç
```bash
# Terminal 3: API'yi başlat
python api.py

# Terminal 4: Ngrok tunnel aç
ngrok http 8000
```

#### 4. Jüriye Link Ver
```
Streamlit: https://abc123.ngrok.io
API: https://def456.ngrok.io
```

⚠️ **DİKKAT:** Ngrok linkler geçici! Demo sırasında bilgisayar açık olmalı.

---

## 🎯 Seçenek 4: Video Demo (Yedek Plan)

### Adımlar:

#### 1. Ekran Kaydı Yap
- OBS Studio veya Windows Game Bar kullan
- 3-5 dakikalık demo çek:
  - Streamlit UI göster
  - MRI yükle ve tahmin göster
  - API Swagger UI göster
  - Canlı API call yap

#### 2. YouTube'a Yükle
- Unlisted video olarak yükle
- Jüriye link ver

#### 3. Loom ile Hızlı Kayıt
- https://loom.com (ücretsiz)
- Tarayıcı eklentisi ile 1 tıkla kayıt
- Otomatik link oluşur

---

## 📋 ÖNERİLEN DEPLOYMENT STRATEJİSİ

### Jüri İçin Submit Edeceğiniz:

1. **Hugging Face Space Link** → Streamlit Demo
   - `https://YOUR_USERNAME-neurobridge-ai-demo.hf.space`

2. **GitHub Repository** → Kaynak Kod
   - `https://github.com/MRjogurtbey/MRI_classification`

3. **Video Demo** → Loom/YouTube
   - `https://loom.com/share/YOUR_VIDEO_ID`

4. **API Documentation** → README
   - Repo içinde detaylı README.md var

---

## 🔧 Hugging Face için Özel requirements.txt

Hugging Face'de GPU yoksa, PyTorch CPU versiyonu kullan:

```txt
# requirements_hf.txt
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.0.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
```

**Not:** `opencv-python` yerine `opencv-python-headless` kullan!

---

## 🎯 Hızlı Başlangıç (Jüri için hazır link)

### ŞUAN İÇİN EN HIZLI: Ngrok

1. Ngrok indir ve yükle
2. Her iki servisi başlat:
   ```bash
   # Terminal 1
   streamlit run app.py
   
   # Terminal 2
   ngrok http 8501
   ```

3. Ngrok link'ini kopyala → Jüriye gönder

4. Demo sırasında bilgisayarı açık tut

---

## 📧 Submit Form Örneği

**Demo Link:** `https://abc123.ngrok.io` (Streamlit)
**API Link:** `https://def456.ngrok.io/docs` (Swagger)
**GitHub:** `https://github.com/MRjogurtbey/MRI_classification`
**Video:** `https://loom.com/share/YOUR_VIDEO` (Yedek)

**Not:** Ngrok kullanıyorsanız demo saatinde bilgisayarın açık olması gerektiğini belirtin.

---

## ⚡ Acil Durum Planı

Eğer hiçbiri çalışmazsa:

1. **Video demo hazır olsun** (3-5 dk)
2. **Ekran görüntüleri al** (confusion matrix, training curves, API swagger)
3. **README.md detaylı olsun** (zaten var!)
4. **Local demo için hazır ol** (kendi laptop'undan göster)

---

## 🎯 Özetle: NE YAPMALIYIZ?

### En Hızlı Çözüm (Bugün):
1. **Ngrok indir** (2 dk)
2. **Tunnel aç** (1 dk)
3. **Linki test et** (1 dk)
4. **Jüriye gönder**

### En İyi Çözüm (Yarın için):
1. **Hugging Face Space oluştur** (10 dk)
2. **Dosyaları deploy et** (5 dk)
3. **Kalıcı link al**
4. **Jüriye gönder**

---

**🚀 Her iki yöntemi de yapmanızı öneririm! Ngrok hızlı yedek, HF Space kalıcı çözüm.**
