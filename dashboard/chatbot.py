# ======== chatbot.py ========
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()



# System prompt with full Pulsar dashboard context
SYSTEM_PROMPT = (
    "You are a helpful assistant for a Pulsar Star Classification dashboard. "
    "The dashboard predicts whether a signal is a pulsar (1) or not (0) based on astronomy features. "
    "Features include integrated profile mean, standard deviation, excess kurtosis, skewness, DM-SNR curve mean, etc. "
    'Models used include Logistic Regression, LightGBM, XGBoost, Decision Tree, and SVM. '
    "SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the training data. "
    "You can explain metrics like accuracy, precision, recall, F1 score, AUC, thresholding, and how the models work. "
    "Provide clear, concise explanations about results, EDA, model behavior, feature distributions, and predictions."
)

def ask_question(user_query, context=None):
    """
    Sends the question + chat history + system prompt to OpenAI.
    """

    # Check for API Key dynamically (force reload to pick up hot-swapped .env updates)
    load_dotenv(override=True)
    api_key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass

    if not api_key or api_key.strip() == "" or api_key.startswith("sk-dummy"):
        return "⚠️ Chatbot is currently unavailable (Backend API Key is missing)."

    local_client = OpenAI(api_key=api_key)

    # Build full message history
    messages = []

    # System prompt
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT if context is None else context
    })

    # Past conversation from Streamlit session
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history:
            messages.append(msg)

    # New user message
    messages.append({"role": "user", "content": user_query})

    try:
        # Call OpenAI Chat API
        response = local_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"*(Error: {e})*"
