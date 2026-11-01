# 🏛️ Government Scheme Eligibility Chatbot

**Problem:** India runs **3,000+ welfare schemes** — scholarships, pensions, housing subsidies, health cover. But crores of eligible citizens never claim them because rules are buried in PDFs and jargon. The people who need help most lose out silently.

**Solution:** A chatbot that asks 7 simple questions (age, income, student?, farmer?...) and lists **exactly which schemes you qualify for** — with benefit amounts, required documents, and where to apply. It also understands free-text questions via NLP intent matching.

## 🧠 The two engines
```
free-text question → [TF-IDF + cosine similarity intent matcher] → topic
profile answers    → [structured eligibility rules engine] → matched schemes
```
- **NLP intent matching from scratch** — TF-IDF vectors + cosine similarity, the same math that powers real chatbots, built manually in ~40 lines
- **Rules engine with real schemes:** PM-KISAN, NMMS scholarship, Sukanya Samriddhi, PM Awas Yojana, Ayushman Bharat, Atal Pension — each with eligibility lambda, documents list, and official apply link
- Honest disclaimer: "rules change — always confirm on the official portal" (real portals like myscheme.gov.in are referenced)

## ▶️ Run
```bash
python main.py
```
Zero dependencies. First watch the NLP demo map 4 free-text questions to schemes, then answer the 7-question profile and get your matches.

## 🌍 Portfolio talking point
Digital-government + inclusion is a powerful essay theme: *"₹ thousands of crores of welfare go unclaimed every year because eligibility is unreadable. I built a tool that makes it readable."* Also shows you handle real-world complexity (rules change, edge cases) gracefully.

## 🚀 Extension ideas
- Add state-level schemes (varies by state — great data-modelling exercise)
- Hindi input support (the users who need this mostly don't type English)
- Voice input for low-literacy users, echoing the braille reader project
