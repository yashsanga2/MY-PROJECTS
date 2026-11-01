# 🪧 Signboard Translator

**Problem:** India has 22 official languages. Travel one state over and you can't read bus stands, shop boards, or hospital signs. Millions of travellers, migrant workers, and truck drivers face this daily.

**Solution:** Photo of a signboard → **text extracted (OCR) → translated → spoken aloud**. Built for regional languages: Hindi, Tamil, Kannada, Bengali, Marathi, Telugu.

## 🧠 The pipeline
```
image → grayscale/pre-process → Tesseract OCR (per-language pack)
      → token cleanup → offline dictionary translation → text-to-speech
```

- **OCR:** Tesseract with Indic language packs (the same engine Google uses in OCR apps)
- **Translation:** works **offline** via a dictionary (extend it!) — no API keys, no cost
- **TTS:** `pyttsx3` reads the translation aloud — designed for low literacy contexts too

## ▶️ Setup
```bash
# 1. Install Tesseract
sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam  # Linux (add -kan, -ben as needed)
brew install tesseract                                            # Mac
# Windows: installer from https://github.com/UB-Mannheim/tesseract/wiki

# 2. Python side
pip install -r requirements.txt

# 3. Run
python main.py signboard.jpg hin
```

## 📊 Data
`MINI_DICTIONARY` in `main.py` holds seed vocabulary — **add the words you actually encounter** (street names, common signage). For full coverage, plug in an offline model like `IndicTrans2` (AI4Bharat).

## 🚀 Extension ideas
- Webcam live-translation mode
- Swap dictionary for AI4Bharat IndicTrans2 (state-of-the-art Indian translation)
- History + favorites for frequent phrases
