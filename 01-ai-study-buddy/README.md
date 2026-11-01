# 📖 AI Study Buddy

**Problem:** Students waste hours manually making revision notes and practice quizzes from textbooks.

**Solution:** Feed this tool your study notes (any `.txt` file) and it instantly generates:
- A **summary** of the most important sentences
- A **fill-in-the-blank quiz** from key concepts

## 🧠 The ML inside
Uses **TF-IDF (Term Frequency × Inverse Document Frequency)** — the same core idea behind search engines — to rank which words and sentences matter most:
- **TF:** how often a word appears in a sentence
- **IDF:** how rare that word is across all sentences (rare = informative)
- Sentence score = average TF-IDF of its words → top sentences become the summary
- Top keywords become quiz blanks

Zero dependencies — pure Python.

## ▶️ Run
```bash
python main.py        # runs the built-in demo (photosynthesis notes)
```

Use your own notes:
```python
from main import summarize, make_quiz
text = open("my-notes.txt").read()
print(summarize(text, 5))
for q in make_quiz(text, 10):
    print(q["question"], "→", q["answer"])
```

## 🚀 Extension ideas
- PDF input with `pypdf`
- MCQ generation (add distractor options from similar TF-IDF words)
- Web UI with Flask
