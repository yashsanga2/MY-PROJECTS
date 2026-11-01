"""
BRAILLE & TTS READER
Problem: Visually impaired people can't read printed menus, letters, notices,
or shop labels. Screen readers solve screens; paper is still a wall.
Braille books exist but almost nothing daily-life is printed in braille.

Solution: A camera/photograph → text (OCR) → TWO accessible outputs:
  1. Voice (text-to-speech) — for anyone
  2. Refreshable braille pattern (Unicode braille) — for braille-display output

Pipeline:  image → OCR (Tesseract) → text-to-speech (pyttsx3, offline)
           → braille Unicode mapping (custom encoder, from scratch)

The braille encoder maps English text to Unicode braille dots
(U+2800 block) — the same 6-dot cell standard used on physical displays.
Writing that encoder from scratch demonstrates you understand the *system*,
not just the API.

  python main.py photo.jpg
"""

import re
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------
# 1. OCR — photo to text
# ---------------------------------------------------------------------
def ocr_image(image_path: str, lang: str = "eng") -> str:
    if shutil.which("tesseract") is None:
        raise SystemExit("Tesseract OCR not installed — see README for setup.")
    result = subprocess.run(
        ["tesseract", image_path, "stdout", "-l", lang, "--psm", "6"],
        capture_output=True, text=True)
    text = result.stdout.strip()
    if not text:
        raise SystemExit("No text detected in image — try better lighting/focus.")
    return text

# ---------------------------------------------------------------------
# 2. Braille encoder — English → Unicode braille (from scratch)
# ---------------------------------------------------------------------
# Unicode braille cell: U+2800 base + dot bits.
# Dot layout:  dot1 dot4   → bits: 1, 8
#             dot2 dot5      bits: 2, 16
#             dot3 dot6      bits: 4, 32

LETTERS = {
    'a': 0x01, 'b': 0x03, 'c': 0x09, 'd': 0x19, 'e': 0x11,
    'f': 0x0B, 'g': 0x1B, 'h': 0x13, 'i': 0x0A, 'j': 0x1A,
    'k': 0x05, 'l': 0x07, 'm': 0x0D, 'n': 0x1D, 'o': 0x15,
    'p': 0x0E, 'q': 0x1E, 'r': 0x16, 's': 0x0E, 't': 0x1C,
    'u': 0x25, 'v': 0x27, 'w': 0x3A, 'x': 0x2D, 'y': 0x3D, 'z': 0x35,
}
NUMBERS = {  # number sign (0x3C) precedes digit using letters j..a pattern
    '1': 'a', '2': 'b', '3': 'c', '4': 'd', '5': 'e',
    '6': 'f', '7': 'g', '8': 'h', '9': 'i', '0': 'j',
}
PUNCT = {'.': 0x04, ',': 0x0C, ';': 0x1C, '!': 0x15, '?': 0x2C,
         '-': 0x20, '(': 0x36, ')': 0x36, "'": 0x00, ':': 0x14}

def to_braille(text: str) -> str:
    out = []
    prev_capital = False
    for ch in text.lower():
        if ch == ' ':
            out.append(' ')
            continue
        if ch in NUMBERS:
            out.append(chr(0x2800 + 0x3C))          # number sign
            out.append(chr(0x2800 + LETTERS[NUMBERS[ch]]))
            continue
        if ch in LETTERS and ch in PUNCT and ch in '.,!?':  # ambiguous chars
            out.append(chr(0x2800 + PUNCT[ch]))
            continue
        if ch in LETTERS:
            out.append(chr(0x2800 + LETTERS[ch]))
        elif ch in PUNCT:
            out.append(chr(0x2800 + PUNCT[ch]))
        else:
            out.append(chr(0x2800 + 0x00))          # blank cell for unknowns
    return "".join(out)

# ---------------------------------------------------------------------
# 3. Text-to-speech (offline, cross-platform)
# ---------------------------------------------------------------------
def speak(text: str):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)     # slightly slower for comprehension
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        # graceful fallback: macOS `say`, Linux `spd-say`
        if sys.platform == "darwin":
            subprocess.run(["say", text])
        else:
            print(f"🔊 (install pyttsx3 for voice): {text}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py photo.jpg [ocr-language]")
        print("Braille test:  python main.py --test")
        sys.exit(0)

    if sys.argv[1] == "--test":
        sample = "hello world 123"
        print(f"English : {sample}")
        print(f"Braille : {to_braille(sample)}")
        print("(Each character is a 6-dot braille cell in Unicode)")
        sys.exit(0)

    lang = sys.argv[2] if len(sys.argv) > 2 else "eng"
    text = ocr_image(sys.argv[1], lang)
    print(f"\n📄 OCR text:\n{text}\n")

    print(f"⠿ Braille output:")
    print(to_braille(text))

    print("\n🔊 Reading aloud...")
    speak(text)