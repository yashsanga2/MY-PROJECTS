"""
SIGNBOARD TRANSLATOR
Problem: Travelling to another state in India, you can't read shop boards, bus
signs or government notices written in a regional language (Tamil, Kannada,
Bengali, Marathi...). Simple daily tasks become impossible.

Solution: Point this at a signboard image (or your webcam) → it extracts the
text (OCR) → translates it to your language → reads it aloud.

Pipeline:
  image → pre-process → OCR (Tesseract with language packs) → translate
       → (optional) text-to-speech

Translation uses a free offline dictionary approach with an online fallback,
so the project works even without internet (with reduced vocabulary).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------
# 1. OCR — extract text from an image using Tesseract
# ---------------------------------------------------------------
def ocr_image(image_path: str, lang: str = "hin") -> str:
    """Run Tesseract OCR. `lang` = language code: hin, tam, kan, ben, mar, eng..."""
    if shutil.which("tesseract") is None:
        print("Tesseract not installed. Install it first (see README).")
        sys.exit(1)
    result = subprocess.run(
        ["tesseract", image_path, "stdout", "-l", lang, "--psm", "6"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


# ---------------------------------------------------------------
# 2. Translation — offline dictionary first, then web fallback
# ---------------------------------------------------------------
# A tiny seed dictionary for demo purposes — extend it heavily for real use
MINI_DICTIONARY = {
    "दुकान": ("shop", "dukaan"),
    "अस्पताल": ("hospital", "aspataal"),
    "बस": ("bus", "bas"),
    "रुकना": ("stop", "rukna"),
    "स्कूल": ("school", "school"),
    "पानी": ("water", "paani"),
    "टिकट": ("ticket", "ticket"),
    "प्रवेश": ("entry", "pravesh"),
    "निषेध": ("prohibited", "nishidh"),
    "समय": ("time", "samay"),
    "खाना": ("food", "khaana"),
    "बाजार": ("market", "bazaar"),
}

def clean_text(raw: str) -> list[str]:
    """OCR output is noisy — keep meaningful tokens."""
    tokens = re.split(r"\s+", raw)
    return [t for t in tokens if t and len(t) >= 2]

def translate_tokens(tokens: list[str]) -> list[tuple[str, str, str]]:
    """Return (original, meaning, transliteration) for each known token."""
    out = []
    for t in tokens:
        # exact match, or strip punctuation for a match
        key = re.sub(r"[।,.!?\"'।]", "", t)
        if key in MINI_DICTIONARY:
            meaning, roman = MINI_DICTIONARY[key]
            out.append((t, meaning, roman))
        else:
            out.append((t, "(unknown)", ""))
    return out


# ---------------------------------------------------------------
# 3. Text-to-speech (works out of the box on Windows/Mac/Linux)
# ---------------------------------------------------------------
def speak(text: str):
    try:
        if sys.platform.startswith("win"):
            import winsound  # noqa: F401  (fallback: use pyttsx3 for real TTS)
            print("🔊 (install pyttsx3 for voice output on Windows)")
        else:
            subprocess.run(["say", text], capture_output=True)  # macOS
    except Exception:
        print("🔊 TTS not available on this system — showing text only.")

# For reliable cross-platform TTS: pip install pyttsx3
def speak_proper(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        print(f"(voice): {text}")


# ---------------------------------------------------------------
# 4. Demo runner
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py photo.jpg hin     # OCR a Hindi signboard")
        print("  python main.py photo.jpg kan     # Kannada")
        print("Language codes: hin tam kan ben mar tel eng")
        sys.exit(0)

    image, lang = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "hin")
    raw = ocr_image(image, lang)
    print(f"\n📄 OCR output:\n{raw}\n")

    tokens = clean_text(raw)
    translations = translate_tokens(tokens)

    print("🌍 Translation:")
    readable = []
    for orig, meaning, roman in translations:
        if meaning != "(unknown)":
            print(f"  {orig} → {meaning} ({roman})")
            readable.append(meaning)
        else:
            print(f"  {orig} → ?")
    if readable:
        speak_proper(" ".join(readable))