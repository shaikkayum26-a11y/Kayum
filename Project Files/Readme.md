project executable files
An AI-powered networking assistant that helps users generate personalized conversation starters for professional events and conferences.

Built using FastAPI, Streamlit, Natural Language Processing (NLP), and Wikipedia API.

📌 Features
🔍 Analyze networking event descriptions
🧠 Detect important topics using NLP
💬 Generate personalized conversation starters
📚 Verify topics using Wikipedia
📝 Save conversation history
👍👎 Collect user feedback on suggestions
📄 Download conversation reports as PDF
🧪 Unit testing with PyTest
📖 Interactive API documentation using Swagger UI
🏗️ Project Architecture
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Services Layer
 ├── Event Analyzer
 ├── Topic Generator
 ├── Fact Checker
 ├── History Logger
 └── Feedback Logger
  ↓
JSON Storage + Wikipedia API
📂 Project Structure
personalized-networking-assistant/
│
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   ├── routers/
│   └── services/
│
├── frontend/
│   └── streamlit_app.py
│
├── tests/
│   ├── conftest.py
│   ├── test_event_analyzer.py
│   ├── test_fact_checker.py
│   └── test_routes.py
│
├── images/
├── history.json
├── feedback.json
├── requirements.txt
├── README.md
└── .env
⚙️ Technologies Used
Technology	Purpose
FastAPI	Backend API
Streamlit	Frontend UI
Transformers	NLP Processing
Wikipedia API	Fact Verification
ReportLab	PDF Generation
PyTest	Unit Testing
Git & GitHub	Version Control
🚀 Installation
1. Clone Repository
git clone [https://github.com/kottehemanthkumarreddy/Emotion-Detection-Learning-Support-Engine.git]

cd personalized-networking-assistant
2. Create Virtual Environment
python -m venv .venv
Activate environment:

Windows

.venv\Scripts\activate
Linux / macOS

source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
▶️ Run Backend
python -m uvicorn app.main:app --reload
Backend runs at:

http://127.0.0.1:8000
Swagger API Documentation:

http://127.0.0.1:8000/docs
▶️ Run Frontend
streamlit run frontend/streamlit_app.py
Frontend runs at:

http://localhost:8501
🧪 Running Tests
Run all test cases:

pytest -v
Example output:

================ 5 passed =================
📷 Application Screenshots
🏠 Home Page
Home Page

📖 Swagger UI
Swagger UI

🔎 Fact Checker
Fact Checker

🌟 Future Enhancements
Gemini API integration
User authentication system
Cloud deployment
Database integration (MongoDB/PostgreSQL)
Analytics dashboard
Dark mode support
👨‍💻 Author
KOTTE HEMANTH KUMAR REDDY

Email : 21013ee003@gmail.com

Roll No. : 24HM5A0226

B.Tech EEE

Course : Google Cloud Generative AI

AITS Kadapa

GitHub: (https://github.com/kottehemanthkumarreddy/Emotion-Detection-Learning-Support-Engine.git)

📜 License
This project was developed as part of an internship project for educational purposes.
