# 🤟 Sign Language → Text Converter

**Problem:** ~70 million deaf people worldwide use sign language, but almost nobody else understands it. Everyday conversations — at shops, hospitals, schools — become a barrier.

**Solution:** A webcam app that reads hand signs in real time and converts them into on-screen text, sentence by sentence.

## 🧠 The ML inside
1. **MediaPipe Hands** tracks 21 hand landmarks (x, y, z) per frame
2. **Normalization** makes each gesture invariant to position, distance, and tilt:
   - translate wrist to origin
   - scale by hand size
   - rotate so the palm axis points up
3. **k-NN classifier** (written from scratch, no sklearn) matches the normalized vector to collected training samples and votes on the sign

## ▶️ Run
```bash
pip install -r requirements.txt

# Step 1 — teach it your signs (repeat for each: hello, thanks, yes, ...)
python main.py --collect hello
python main.py --collect thanks

# Step 2 — recognize live
python main.py
```
Keys while collecting: `s` = save current frame's landmarks, `q` = quit.
Keys while recognizing: `c` = clear sentence, `q` = quit.

## 📊 Data
Training samples are appended to `sign_data.csv` (63 features + label per row). ~30–50 samples per sign gives solid accuracy.

## 🚀 Extension ideas
- Alphabet gestures (A–Z) instead of whole words
- Swap k-NN for a small neural network (higher accuracy)
- Text-to-speech output so signs become voice
