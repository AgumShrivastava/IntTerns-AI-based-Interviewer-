# 🤖 AI Interview Evaluator (NexaHire)

An intelligent AI-based interview system that evaluates candidate answers using Natural Language Processing (NLP), semantic similarity, and adaptive difficulty.

This project simulates a real interview environment where candidates answer questions, and the system evaluates their performance automatically.

---

## 🚀 Features

* 🎤 AI-powered interview simulation
* 🧠 NLP-based answer evaluation
* 📊 Score generation (out of 10)
* 🔁 Adaptive difficulty (Easy → Medium → Hard)
* 💬 Interactive chat-based UI
* ⚡ Real-time feedback system
* 🧾 Final HR-style report generation

---

## 🧠 How It Works

### 1. Answer Evaluation

* Uses **Sentence Transformers (`all-MiniLM-L6-v2`)** for semantic similarity
* Compares:

  * Ideal Answer vs Candidate Answer
* Uses **Cosine Similarity** to calculate score

👉 Code Reference: 

---

### 2. Scoring Logic

* Similarity Score → converted to scale of 10
* Penalty for filler words like:

  * *um, uh, like, basically*
* Behavioral answers scored based on:

  * Word count
  * Fluency

---

### 3. Backend API

Built using **Flask**

* Endpoint: `/evaluate`
* Method: POST
* Input:

```json
{
  "ideal_answer": "...",
  "candidate_answer": "...",
  "is_behavioral": false
}
```

👉 Code Reference: 

---

### 4. Frontend

* Built using HTML, CSS, JavaScript
* Chat-based UI simulating real interview
* Voice input support 🎤
* Live score tracking

👉 Code Reference: 

---

## 📦 Tech Stack

* **Backend:** Flask, Python
* **Frontend:** HTML, CSS, JavaScript
* **ML/NLP:** Sentence Transformers
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn

👉 Requirements: 

---

## 📊 Dataset

* CSV file containing:

  * Questions
  * Ideal Answers
  * Difficulty Levels

Example:

| Question     | Answer | Difficulty |
| ------------ | ------ | ---------- |
| What is OOP? | ...    | Medium     |

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Backend

```bash
python api.py
```

### 4. Run Frontend

* Open `index.html` in browser
* OR use Live Server

---

## 📸 Project Flow

1. User enters name
2. AI asks questions
3. User responds (text/voice)
4. AI evaluates answer
5. Score updates dynamically
6. Final report generated

---

## 🎯 Use Cases

* Interview preparation
* Skill assessment platforms
* EdTech tools
* HR automation

---

## 📈 Future Improvements

* Add GPT-based evaluation
* Speech-to-text backend integration
* More advanced scoring metrics
* Database integration
* User authentication system

---

## 👨‍💻 Author

**Agum**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
