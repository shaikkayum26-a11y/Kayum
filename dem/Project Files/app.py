import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from src.gemini_helper import get_gemini_response
from src.bilstm_predictor import EmotionPredictor
from src.bert_model import BERTEmotionClassifier
from src.gemini_helper import get_gemini_response
from src.emotion_responses import EMOTION_RESPONSES
from src.mixed_emotion import MixedEmotionDetector
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🤖",
    layout="wide"
)
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

# -------------------------------------------------------
# Load Models (Cached)
# -------------------------------------------------------

@st.cache_resource
def load_models():

    try:
        bilstm_model = EmotionPredictor()

        bert_model = BERTEmotionClassifier()
        bert_model.load_model()

        return bilstm_model, bert_model, "✅ Models Loaded"

    except Exception as e:

        return None, None, f"❌ Error: {e}"


bilstm_model, bert_model, status = load_models()

st.success(status)
detector = MixedEmotionDetector(threshold=0.15)

# -------------------------------------------------------
# Save Prediction to CSV
# -------------------------------------------------------

def save_to_csv(
    field,
    problem,
    emotion,
    confidence,
    ai_response
):

    new_row = {

        "text": problem,
        "emotion": emotion,
        "confidence": confidence,
        "response": ai_response,
        "field": field,
        "timestamp": datetime.now().isoformat()

    }

    ####################################################
    # Save interaction history
    ####################################################

    filename = "emotion_response_examples.csv"

    if os.path.exists(filename):

        df = pd.read_csv(filename)

        df = pd.concat(
            [df, pd.DataFrame([new_row])],
            ignore_index=True
        )

    else:

        df = pd.DataFrame([new_row])

    df.to_csv(filename, index=False)

    ####################################################
    # Save emotion-response mapping
    ####################################################

    mapping_file = "emotion_response_mapping.csv"

    if os.path.exists(mapping_file):

        mapping_df = pd.read_csv(mapping_file)

    else:

        mapping_df = pd.DataFrame(
            columns=["emotion", "response"]
        )

    if emotion not in mapping_df["emotion"].values:

        mapping_df = pd.concat(
            [
                mapping_df,
                pd.DataFrame([{
                    "emotion": emotion,
                    "response": ai_response
                }])
            ],
            ignore_index=True
        )

    mapping_df.to_csv(mapping_file, index=False)

    return True
def add_to_history(
    field,
    problem,
    emotion,
    confidence,
    ai_response,
    bilstm_scores,
    bert_result=None
):
    st.session_state.emotion_history.append({

        "timestamp": datetime.now(),

        "field": field,

        "problem": problem,

        "emotion": emotion,

        "confidence": confidence,

        "ai_response": ai_response,

        "all_scores": bilstm_scores,

        "model": "BiLSTM"

    })
    if bert_result:

        st.session_state.emotion_history.append({

            "timestamp": datetime.now(),

            "field": field,

            "problem": problem,

            "emotion": bert_result["emotion"],

            "confidence": bert_result["confidence"],

            "ai_response": ai_response,

            "all_scores": bert_result["scores"],

            "model": "BERT"

        })
def get_mixed_emotions(scores, threshold=0.15):

    if not scores:
        return []

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    primary = sorted_scores[0]

    mixed = [primary]

    for emotion, score in sorted_scores[1:]:

        if primary[1] - score <= threshold:
            mixed.append((emotion, score))

    return mixed
col1, col2 = st.columns([3, 1])

# LEFT COLUMN
with col1:

    st.subheader("📚 Tell us about your learning challenge")

    field = st.selectbox(
        "What field are you studying?",
        [
            "Computer Science",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Engineering",
            "Business",
            "Literature",
            "History",
            "Psychology",
            "Other"
        ]
    )

    user_text = st.text_area(
    f"Describe your {field} problem:",
    height=150
)
# RIGHT COLUMN
with col2:

    st.subheader("⚙️ Settings")

    use_ai = st.checkbox("Use Gemini AI", value=True)

    save_data = st.checkbox("Save to CSV", value=True)

    show_details = st.checkbox("Show Analysis Details", value=False)
    st.markdown("---")

    st.write("📁 Predict from Saved Data")

    use_csv_prediction = st.checkbox(
        "Use CSV-based prediction",
        value=False
    )

    examples_df = pd.DataFrame()

    if os.path.exists("emotion_response_examples.csv"):
        examples_df = pd.read_csv("emotion_response_examples.csv")

    if use_csv_prediction and len(examples_df) > 0:
        st.info(
            f"Using {len(examples_df)} saved examples for prediction"
        )
bert_result = None
scores = {}
emotion = ""
confidence = 0.0
ai_response = ""
if st.button("Predict Emotion"):

        if user_text.strip() == "":
            st.warning("Please enter some text.")

        else:
            result = bilstm_model.predict(user_text)
            emotion = result["emotion"]
            confidence = result["confidence"]
            scores = result["scores"]
            mixed = detector.get_mixed_emotions(scores)
            mixed_result = detector.format_output(mixed)
            emotion = mixed_result["emotion"]
            confidence = mixed_result["confidence"]
            bert_result=None
            if bert_model:
                bert_result = bert_model.predict(user_text)
            bilstm_mixed = get_mixed_emotions(scores)
            print(bert_result)
           
            if bert_result:
                bert_mixed = get_mixed_emotions(
                    bert_result["scores"]
            )
            emotion = emotion.strip().lower()
            use_ai = True

            if use_ai:

                ai_response = get_gemini_response(
                field,
                user_text,
                emotion,
                confidence
            )

            if ai_response is None:

                ai_response = EMOTION_RESPONSES[emotion]["response"]

            else:

                ai_response = EMOTION_RESPONSES[emotion]["response"]
            if save_data:
                save_to_csv(
                    field,
                    user_text,
                    emotion,
                    confidence,
                    ai_response
                )
            add_to_history(
                field,
                user_text,
                emotion,
                confidence,
                ai_response,
                scores,
                bert_result
            )
            st.subheader("⚖️ Model Predictions Comparison")
            if bert_result:

                col1, col2 = st.columns(2)

            else:

                col1 = st.columns(1)[0]
            with col1:

                st.write("### BiLSTM")

                st.metric(
                    "Emotion",
                    emotion,
                    f"{confidence:.1%}"
                )

                st.write("Scores")

                for emotion_name, score in sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):

                    st.progress(score)

                    st.caption(f"{emotion_name}: {score:.2%}")
            if bert_result is not None:

                with col2:

                    st.write("### BERT")

                    st.metric(
                        "Emotion",
                        bert_result["emotion"],
                        f"{bert_result['confidence']:.1%}"
                    )

                    st.write("Scores")

                    for emotion_name, score in sorted(
                        bert_result["scores"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):

                        st.progress(score)

                        st.caption(f"{emotion_name}: {score:.2%}")
            # ============================================
            response_data = EMOTION_RESPONSES.get(
                emotion,
                {
                     "emoji": "🙂",
                    "response": ai_response,
                    "action": "Keep learning!"
                }
            )

            st.subheader("🤖 AI Learning Assistant")

            st.write(
                response_data["emoji"],
                ai_response
            )

            st.info(
                "Suggested Action: "
                + response_data["action"]
            )  
if st.session_state.emotion_history:

    st.markdown("---")
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.emotion_history)

    # Convert timestamp column
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    tab1, tab2, tab3 = st.tabs(
        ["Emotions", "Fields", "Summary"]
    )   
    with tab1:

        col1, col2 = st.columns(2)

    # Pie Chart
        with col1:

            emotion_counts = df["emotion"].value_counts()

            fig1 = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="Emotion Distribution"
        )

        st.plotly_chart(fig1, use_container_width=True)

    # Timeline
        with col2:

            df_copy = df.copy()

            df_copy["time"] = df_copy["timestamp"].dt.strftime("%H:%M:%S")

            fig2 = px.line(
            df_copy,
            x="time",
            y="confidence",
            color="emotion",
            markers=True,
            title="Emotional Journey"
        )

        st.plotly_chart(fig2, use_container_width=True)
    with tab2:

        if "model" in df.columns:

            field_emotion = (
            df.groupby(["field", "emotion", "model"])
            .size()
            .reset_index(name="count")
        )

            fig3 = px.bar(
            field_emotion,
            x="field",
            y="count",
            color="emotion",
            facet_col="model",
            title="Emotions by Study Field & Model"
        )
        else:
            field_emotion = (
            df.groupby(["field", "emotion"])
            .size()
            .reset_index(name="count")
        )

            fig3 = px.bar(
            field_emotion,
            x="field",
            y="count",
            color="emotion",
            title="Emotions by Study Field"
        )

        st.plotly_chart(fig3, use_container_width=True)
    with tab3:

        st.metric(
        "Total Predictions",
        len(df)
    )

        st.metric(
        "Unique Emotions",
        df["emotion"].nunique()
    )

        st.metric(
        "Study Fields",
        df["field"].nunique()
    )

        st.subheader("Recent Records")

    st.dataframe(df.tail(10))
with st.sidebar:

    st.header("📊 Dashboard")

    st.write(f"Models : {status}")

    st.write(
        f"Total Interactions : {len(st.session_state.emotion_history)}"
    )

    csv_count = 0

    if os.path.exists("emotion_response_examples.csv"):

        csv_count = len(
            pd.read_csv("emotion_response_examples.csv")
        )

    st.write(f"CSV Examples : {csv_count}")

    if st.button("Clear History"):

        st.session_state.emotion_history = []

        st.rerun()

    if st.session_state.emotion_history:

        st.subheader("Recent Sessions")

        recent = st.session_state.emotion_history[-3:]

        for item in reversed(recent):

            st.write(
                f"• {item['field']} : "
                f"{item['emotion']} "
                f"({item['confidence']:.1%})"
            )
