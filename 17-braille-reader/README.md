# ⠿ Braille & TTS Reader

**Problem:** Screen readers solved digital screens — but the physical world is still a wall for blind people. Restaurant menus, letters, notice boards, medicine labels: almost nothing daily-life is printed in braille.

**Solution:** Photograph any printed text → get **two accessible outputs**: spoken audio (offline TTS) and **Unicode braille** ready for a refreshable braille display.

## 🧠 What's inside
```
photo → Tesseract OCR → text ─┬→ pyttsx3 text-to-speech (offline)
                              └→ custom braille encoder (from scratch)
```
- **The braille encoder is the crown jewel:** written from scratch, mapping English to the Unicode braille cell block (U+2800 + dot bits), including the number-sign prefix rule and punctuation cells — proof you understand the *system*, not just an API
- Fully **offline** — designed for areas with patchy internet
- Bonus: `python main.py --test` prints braille directly in the terminal — a 10-second wow demo

## ▶️ Run
```bash
# Setup: install Tesseract OCR (tesseract-ocr on apt/brew), then:
pip install -r requirements.txt

python main.py --test        # instant braille demo, no image needed
python main.py menu.jpg      # full pipeline: OCR → braille → voice
```

## 🌍 Portfolio talking point
Accessibility tech is one of the most respected project categories there is — and the from-scratch braille encoder (with the dot-bit math documented in comments) shows engineering depth most portfolios lack. Mention: "the number sign rule exists because braille digits reuse the letters a–j."

## 🚀 Extension ideas
- Hindi braille (Bharati braille standard) — a genuine gap in most accessibility tools
- Live webcam reading mode with a physical "scan" button
- Output to an actual braille emulator display via serial/Bluetooth
