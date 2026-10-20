import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Registration Assistant", page_icon="🎓", layout="wide")

# -----------------------------
# Sample knowledge base
# Replace these sample rules with your institution's official information.
# -----------------------------
FAQS = [
    {
        "question": "How do I register for courses?",
        "answer": "Typical steps: 1) Sign in to your student portal. 2) Open Course Registration. 3) Select the semester. 4) Choose eligible courses. 5) Check credits and timetable clashes. 6) Submit and save the confirmation. Follow your college's official process."
    },
    {
        "question": "Which documents are required for registration?",
        "answer": "Commonly requested documents may include your student ID, previous semester marksheet, fee receipt, and any institution-specific forms. The exact list depends on your college, so verify it with the official notice or registrar."
    },
    {
        "question": "What if I cannot log in?",
        "answer": "Check your student ID and password, use the portal's password-reset option, and verify that your account is active. If the issue continues, contact your college IT helpdesk or registrar. Never share your password in the chatbot."
    },
    {
        "question": "What if a course is not visible?",
        "answer": "Check whether registration is open, whether the course is offered to your program and semester, and whether you meet its prerequisites. If it still does not appear, contact your department or registrar."
    },
    {
        "question": "How can I check registration status?",
        "answer": "Open the student portal and look for Registration Status, Registered Courses, or Confirmation. Save a copy of the confirmation. If your status remains pending, contact the registrar."
    },
    {
        "question": "What are prerequisites?",
        "answer": "Prerequisites are courses or requirements that must be completed before you can enrol in another course. The official course catalogue determines the prerequisites."
    },
    {
        "question": "What should I do if I miss the deadline?",
        "answer": "Contact your department or registrar as soon as possible and ask whether late registration or an extension is available. Do not assume an extension will be granted."
    },
    {
        "question": "How do I choose courses?",
        "answer": "Review your degree requirements, prerequisites, credit limits, timetable, and academic plan. Confirm your choices with your academic adviser if you are unsure."
    },
    {
        "question": "How do I contact the registrar?",
        "answer": "Use the official contact details listed on your college website or student portal. This demo does not contain your institution's actual contact information."
    },
]

COURSES = pd.DataFrame([
    {"Course Code": "CS501", "Course Name": "Introduction to Machine Learning", "Semester": 5, "Credits": 4, "Prerequisite": "Python Programming"},
    {"Course Code": "CS502", "Course Name": "Natural Language Processing", "Semester": 5, "Credits": 3, "Prerequisite": "Python Programming"},
    {"Course Code": "CS503", "Course Name": "Database Management Systems", "Semester": 5, "Credits": 4, "Prerequisite": "None"},
    {"Course Code": "CS504", "Course Name": "Cybersecurity Fundamentals", "Semester": 5, "Credits": 3, "Prerequisite": "Computer Networks"},
    {"Course Code": "CS401", "Course Name": "Python Programming", "Semester": 4, "Credits": 4, "Prerequisite": "None"},
])

def normalize(text):
    """Basic NLP preprocessing: lowercase, remove punctuation, normalize spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

faq_questions = [normalize(item["question"]) for item in FAQS]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
faq_matrix = vectorizer.fit_transform(faq_questions)

def chatbot_reply(user_text):
    """Match the student's question to the closest FAQ using TF-IDF and cosine similarity."""
    cleaned = normalize(user_text)
    if not cleaned:
        return "Please type a question about registration, eligibility, courses, or documents."

    query_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(query_vector, faq_matrix).flatten()
    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.12:
        return ("I am not confident I understood that question. Try asking about course registration, "
                "required documents, login issues, prerequisites, deadlines, or registration status.")
    return FAQS[best_index]["answer"]

def check_eligibility(cgpa, completed_credits, semester, prerequisites_completed):
    """Example eligibility rule only; replace with official institution rules."""
    reasons = []
    if semester < 1 or semester > 8:
        reasons.append("Enter a semester between 1 and 8.")
    if cgpa < 0 or cgpa > 10:
        reasons.append("CGPA must be between 0 and 10.")
    if completed_credits < 0:
        reasons.append("Completed credits cannot be negative.")

    if reasons:
        return False, reasons

    # Demonstration policy, NOT a universal academic rule.
    if cgpa < 5.0:
        reasons.append("Sample rule: minimum CGPA for this demo is 5.0.")
    if completed_credits < (semester - 1) * 20:
        reasons.append(f"Sample rule: a Semester {semester} student should have at least {(semester - 1) * 20} completed credits.")
    if not prerequisites_completed:
        reasons.append("Some required prerequisite courses are marked incomplete.")

    if reasons:
        return False, reasons
    return True, ["The student meets this demo's sample checks. Confirm final eligibility with the official college system."]

st.title("🎓 AI-Powered Registration Assistant")
st.caption("A Python + NLP chatbot prototype for course registration, eligibility checks, and student guidance.")

with st.sidebar:
    st.header("Project overview")
    st.write("**Technologies:** Python, Streamlit, scikit-learn, TF-IDF, cosine similarity")
    st.info("This is a demonstration. The courses, credit thresholds, and eligibility rules are sample data—not official college policy.")
    st.markdown("**Try asking:**")
    st.markdown("- How do I register for courses?\n- Which documents are required?\n- What if I cannot log in?\n- What are prerequisites?")

tab_chat, tab_eligibility, tab_courses, tab_about = st.tabs([
    "💬 Chatbot", "✅ Eligibility Checker", "📚 Course Catalogue", "ℹ️ About the Project"
])

with tab_chat:
    st.subheader("Ask the Registration Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can help with registration steps, documents, deadlines, prerequisites, and common portal problems. What would you like to know?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Type your registration question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        answer = chatbot_reply(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

with tab_eligibility:
    st.subheader("Sample eligibility checker")
    st.warning("These rules are placeholders for demonstration. Replace them with your institution's approved regulations before real use.")
    with st.form("eligibility_form"):
        col1, col2 = st.columns(2)
        with col1:
            cgpa = st.number_input("Current CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
            semester = st.selectbox("Semester applying for", options=list(range(1, 9)), index=4)
        with col2:
            completed_credits = st.number_input("Credits completed", min_value=0, max_value=300, value=80, step=1)
            prerequisites_completed = st.checkbox("I have completed the required prerequisites", value=True)
        submitted = st.form_submit_button("Check eligibility")

    if submitted:
        eligible, messages = check_eligibility(cgpa, completed_credits, semester, prerequisites_completed)
        if eligible:
            st.success("Demo result: eligible based on the sample rules.")
        else:
            st.error("Demo result: one or more sample checks were not met.")
        for message in messages:
            st.write(f"- {message}")
        st.caption("This result is not an official admission, progression, or registration decision.")

with tab_courses:
    st.subheader("Example course catalogue")
    selected_semester = st.selectbox("Filter by semester", ["All"] + sorted(COURSES["Semester"].unique().tolist()), key="course_semester")
    view = COURSES if selected_semester == "All" else COURSES[COURSES["Semester"] == selected_semester]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.caption("Edit the COURSES table in app.py to match your real curriculum.")

with tab_about:
    st.subheader("How the project works")
    st.markdown("""
    **Python:** runs the app, stores the sample FAQ/course data, and applies the eligibility rules.

    **NLP:** cleans text by converting it to lowercase and removing punctuation. TF-IDF turns the FAQ questions into numerical vectors. Cosine similarity compares a student's question with those vectors.

    **Chatbot:** shows the best matching answer in a conversational interface. If the match score is too low, it asks the student to try a more specific question.

    **Streamlit:** creates the web interface without requiring separate HTML, CSS, and JavaScript files.

    **Current limitations:** this prototype does not connect to a college database, authenticate students, submit registrations, or use official institutional rules. Those features require approved data, secure authentication, and integration with the college's systems.
    """)
    st.subheader("Suggested future improvements")
    st.markdown("- Store official courses and rules in a database.\n- Add login through the institution's approved authentication system.\n- Add a deadline calendar and email notifications.\n- Log unanswered questions to improve the FAQ knowledge base.\n- Add multilingual support and a larger NLP model if needed.")
