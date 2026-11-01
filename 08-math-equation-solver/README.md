# ✍️ Handwritten Math Equation Solver

**Problem:** Students get stuck on handwritten equations — typing math into a calculator is slow and demotivating, so most never check their work.

**Solution:** Photo of a handwritten linear equation (e.g. `2x + 5 = 13`) → **reads the characters → solves it → shows the answer.**

## 🧠 The pipeline (3 real ML/CV stages)
```
photo → binarize + contour segmentation → glyph classification → solve
```
1. **Segmentation:** Otsu thresholding + contour bounding boxes, sorted left→right (OpenCV)
2. **Classification:** a small **CNN trained on MNIST** (60k handwritten digit images, ~99% accuracy in 3 epochs) + shape heuristics for `+`, `-`, `=` (wide-short = equals, flat = minus, tiny = plus)
3. **Solver:** linear equation parser + algebra **written from scratch** — `ax + b = c → x = -b/a` — handles `2x+5=13`, `4x-7=21`, even `12=3x+3`

## ▶️ Run
```bash
pip install -r requirements.txt

python main.py --train               # 1. train digit CNN once (~2 min CPU)
python main.py --text "2x+5=13"      # 2. test the solver brain alone
python main.py equation.jpg          # 3. solve from a photo
```

**Testing tip:** write `2x+5=13` large and clear on white paper, good lighting.

## 🚀 Extension ideas
- Quadratic support (x² glyph → discriminant solver)
- Webcam live mode
- Show step-by-step working (great for exam revision)
