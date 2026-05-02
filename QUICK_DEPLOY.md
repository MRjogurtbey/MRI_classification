# 🚀 Quick Deploy Guide - Streamlit Cloud

## ⚡ 2 Dakikada Online Demo!

### Adım 1: GitHub'a Push (ZATen YAPILDI ✅)
```bash
# Repo zaten güncel: feature/model-v3-dataset-expansion
```

### Adım 2: Streamlit Cloud'a Git
1. **Website:** https://streamlit.io/cloud
2. **Sign in with GitHub** tıkla
3. Repo'ya erişim izni ver

### Adım 3: New App Deploy
1. **"New app"** butonuna tıkla
2. **Repository:** `MRjogurtbey/MRI_classification` seç
3. **Branch:** `feature/model-v3-dataset-expansion` seç
4. **Main file path:** `app.py`
5. **Deploy!** tıkla

### Adım 4: Bekle (2-3 dakika)
- Streamlit otomatik build eder
- Requirements yükler
- Model yükler

### Adım 5: HAZIR! 🎉
Link örneği: `https://neurobridge-ai.streamlit.app`

---

## 📋 Jüriye Vereceğiniz Link:

```
Live Demo: https://YOUR_APP_NAME.streamlit.app
GitHub: https://github.com/MRjogurtbey/MRI_classification
```

---

## ⚠️ Önemli Notlar:

### Model Dosyası Sorunu
Eğer `checkpoints/best_model.pth` çok büyükse (>100MB), GitHub LFS kullanmalısınız:

```bash
# LFS kur
git lfs install

# Model dosyasını track et
git lfs track "*.pth"
git add .gitattributes
git add checkpoints/best_model.pth
git commit -m "Add model with LFS"
git push
```

**Veya:** Google Drive'dan yükle:

```python
# app.py'nin başına ekle
import gdown

if not Path("checkpoints/best_model.pth").exists():
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    gdown.download(url, "checkpoints/best_model.pth", quiet=False)
```

---

## 🎯 Alternatif: Render (Eğer Streamlit Cloud çalışmazsa)

### Adım 1: Render'a Git
https://render.com → Sign up with GitHub

### Adım 2: New Web Service
- Repository: `MRI_classification`
- Branch: `feature/model-v3-dataset-expansion`
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Adım 3: Deploy
Link: `https://neurobridge-ai.onrender.com`

---

## 🔥 HIZLI TEST:

```bash
# Local test (son kontrol)
streamlit run app.py

# Tarayıcıda aç: http://localhost:8501
# Çalışıyorsa, cloud'da da çalışır!
```

---

## 📊 Dosya Boyutu Kontrolü:

```bash
# Model dosyası boyutunu kontrol et
cd checkpoints
ls -lh best_model.pth

# Eğer >100MB ise git LFS kullan!
```

---

## ✅ Hazırlanan Dosyalar:

- ✅ `.streamlit/config.toml` - Streamlit ayarları
- ✅ `packages.txt` - Sistem paketleri (OpenCV için)
- ✅ `requirements.txt` - opencv-python-headless (cloud için)

---

## 🎯 ŞİMDİ NE YAPMALIYIZ?

1. **Push et** (yeni dosyaları GitHub'a)
2. **Streamlit Cloud'a git** (https://streamlit.io/cloud)
3. **Sign in with GitHub**
4. **New app → Deploy**
5. **Link'i jüriye gönder**

**Toplam süre: 5 dakika! 🚀**
