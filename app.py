from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
import re
from fpdf import FPDF
import io

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="🧠 AI Mental Health Assistant", layout="wide")
st.title("🧠 MindCare AI")

# Ultra-detailed, expert-level prompt
gemini_prompt = """
You are a world-class, compassionate, and clinically trained AI mental health expert and sentiment analysis specialist. Your role is to deeply analyze the user’s input—whether text, images, or a combination—to provide a comprehensive emotional and psychological assessment, focusing on mental wellbeing, distress signals, and suicide risk detection.

Your response must strictly follow the structure below and include all details asked, using clear, precise, and empathetic language appropriate for sensitive topics.

---

1. **Sentiment Breakdown**

- Calculate the percentage distribution of overall sentiment in the input text or image analysis: Positive, Neutral, and Negative sentiments. Present each as a percentage rounded to the nearest whole number.
- Briefly explain the dominant sentiment and what it suggests about the user’s current emotional state.

2. **Emotional Profile and Intensity**

- Identify key emotions present in the input. Examples include: sadness, anxiety, hopelessness, fear, anger, guilt, shame, loneliness, calm, joy, relief, optimism.
- For each detected emotion, provide an intensity score between 0 (not present) to 100 (extremely strong).
- Summarize how these emotions interact or dominate the user’s mental state.

3. **Risk Assessment Score**

- Generate a comprehensive risk score from 0 to 100 quantifying the likelihood of emotional distress, crisis, suicidal ideation, or self-harm risk.
- Categorize the risk as LOW (0-33), MODERATE (34-66), or HIGH (67-100).
- Provide a detailed rationale explaining the basis for the risk score, citing specific emotional indicators, language patterns, or visual cues.

4. **Warning Signals and Risk Indicators**

- List explicit warning signs or concerning phrases found in the input, such as hopeless statements, self-harm references, feelings of worthlessness, isolation, or expressions of severe anxiety.
- Highlight any patterns or repetitions that increase concern.
- If applicable, identify indirect signals like withdrawal, fatigue, or drastic mood changes inferred from context.

5. **Empathetic and Supportive Response**

- Address the user directly with a warm, understanding, and non-judgmental message.
- Validate their feelings, acknowledge their pain or struggle, and emphasize that help is available.
- Encourage hope and the importance of reaching out for support.

6. **Personalized Recommendations for Mental Wellbeing**

- Provide tailored, actionable suggestions based on the emotional and risk profile:
  - Daily mindfulness or relaxation exercises
  - Social engagement or counseling recommendations
  - Healthy lifestyle advice (sleep, nutrition, exercise)
  - Cognitive behavioral techniques for managing distressing thoughts
  - Crisis resources and emergency contacts if HIGH risk is detected
- Offer reputable resources (hotlines, websites, apps) for mental health support and crisis intervention.

7. **Structured Output Formatting**

- Clearly separate each section with bold headings.
- Use numbered lists and bullet points for readability.
- Include exact percentages and numeric scores for each metric.
- Output in plain text with no extraneous commentary or meta-information.
- Ensure the tone is consistently professional, sensitive, and empathetic.

---

**Important:**

- Never minimize or dismiss any signs of distress.
- Prioritize user safety by flagging any HIGH risk with clear advice to seek immediate help.
- Maintain confidentiality and neutrality, avoiding assumptions beyond the data provided.
- Be thorough but concise, ensuring the output is actionable and suitable for integration into user-facing applications.

Begin your detailed mental health and sentiment analysis now.
"""

def get_gemini_response(user_text, image_parts, prompt, image_uploaded):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if image_uploaded:
        response = model.generate_content([user_text, image_parts[0], prompt])
    else:
        response = model.generate_content([user_text, prompt])
    return response.text

def input_image_setup(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return [{"mime_type": uploaded_file.type, "data": bytes_data}]

def parse_sentiments(text):
    matches = re.findall(r"(Positive|Neutral|Negative).*?(\d+)%", text, re.IGNORECASE)
    return {label: int(score) for label, score in matches}

def parse_risk_score(text):
    match = re.search(r"Risk.*?(\d{1,3})", text, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        return max(0, min(val, 100))
    return 0

def display_risk_level(score):
    if score >= 67:
        st.markdown(f"### 🔴 High Risk ({score}/100)")
        st.markdown("[🆘 Crisis Support Resources](https://www.opencounseling.com/suicide-hotlines)")
    elif 34 <= score <= 66:
        st.markdown(f"### 🟠 Moderate Risk ({score}/100)")
    else:
        st.markdown(f"### 🟢 Low Risk ({score}/100)")

def plot_emotion_pie(sentiments):
    if sentiments:
        labels, values = zip(*sentiments.items())
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.0f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

def generate_pdf_report(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(0, 10, txt=line.strip(), ln=True)
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# UI Inputs
user_input = st.text_area("✍️ Enter your text or journal here:")

uploaded_file = st.file_uploader("📷 Upload an image (optional)", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if st.button("🧠 Analyze Mental Health"):
    if not user_input.strip() and not uploaded_file:
        st.warning("Please provide text input or upload an image to analyze.")
    else:
        with st.spinner("Analyzing your emotional and mental state..."):
            try:
                image_uploaded = uploaded_file is not None
                image_data = input_image_setup(uploaded_file) if image_uploaded else None
                response = get_gemini_response(user_input, image_data, gemini_prompt, image_uploaded)

                st.subheader("🧾 AI Mental Health Sentiment Report")
                st.write(response)

                # Risk Assessment
                st.subheader("📊 Risk Assessment")
                risk_score = parse_risk_score(response)
                display_risk_level(risk_score)

                # Sentiment Pie Chart
                sentiments = parse_sentiments(response)
                if sentiments:
                    st.subheader("📉 Emotion Sentiment Breakdown")
                    plot_emotion_pie(sentiments)

                # Download PDF Report
                pdf_report = generate_pdf_report(response)
                st.download_button("📥 Download Full Report as PDF", data=pdf_report, file_name="mental_health_report.pdf")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
