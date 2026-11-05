
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(
    page_title="LeafCare AI | Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.txt"

# ---------------------------------------------
# BOTANICAL THEME AND CUSTOM CSS
# ---------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --forest: #174C35;
        --green: #287A4D;
        --light-green: #EAF5EC;
        --cream: #F7FAF5;
        --text: #20372A;
        --muted: #68796D;
    }

    .stApp {
        background: linear-gradient(180deg, #F5FAF3 0%, #FFFFFF 48%, #F5FAF3 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--forest) !important;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 15px;
        padding: 12px 0 25px 0;
        border-bottom: 1px solid #E2EDE1;
        margin-bottom: 28px;
    }

    .brand {
        color: #174C35;
        font-family: 'Manrope', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .brand span {
        color: #47965D;
    }

    .top-tag {
        background: #E8F4E8;
        color: #287A4D;
        border: 1px solid #D5E9D5;
        padding: 7px 13px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .hero {
        background: linear-gradient(120deg, #E7F4E6, #F6FAEE);
        border: 1px solid #DDEBDD;
        border-radius: 25px;
        padding: 35px 34px;
        margin-bottom: 25px;
    }

    .eyebrow {
        color: #287A4D;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #174C35;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.15rem);
        font-weight: 800;
        letter-spacing: -1.7px;
        line-height: 1.13;
        margin-bottom: 15px;
    }

    .hero-copy {
        color: #526B58;
        font-size: 1.05rem;
        line-height: 1.8;
        max-width: 680px;
    }

    .section-label {
        color: #287A4D;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-heading {
        color: #174C35;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .section-copy {
        color: #68796D;
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E2EDE1;
        border-radius: 18px;
        padding: 20px;
        min-height: 135px;
        box-shadow: 0 5px 20px rgba(27, 75, 43, 0.035);
    }

    .feature-icon {
        font-size: 1.55rem;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #174C35;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .feature-copy {
        color: #68796D;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    div[data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #A8CCAC;
        border-radius: 17px;
        padding: 14px;
    }

    div[data-testid="stFileUploader"] section {
        background: #F7FBF5;
        border-radius: 12px;
    }

    div[data-testid="stButton"] > button {
        background: #287A4D;
        color: white;
        border: 1px solid #287A4D;
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div[data-testid="stButton"] > button:hover {
        background: #174C35;
        border-color: #174C35;
        color: white;
    }

    .result-card {
        background: #F0F8EF;
        border: 1px solid #D8EAD6;
        border-radius: 18px;
        padding: 22px;
        margin: 12px 0;
    }

    .result-label {
        color: #60806A;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .result-name {
        color: #174C35;
        font-family: 'Manrope', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .confidence-number {
        color: #287A4D;
        font-family: 'Manrope', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2EDE1;
        padding: 17px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #68796D;
    }

    div[data-testid="stProgress"] > div {
        background: #E5EFE3;
        border-radius: 10px;
    }

    div[data-testid="stProgress"] div[role="progressbar"] {
        background: #47965D;
        border-radius: 10px;
    }

    .notice {
        background: #FFF8E8;
        color: #795C22;
        border: 1px solid #F0E1B6;
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 0.9rem;
        line-height: 1.65;
        margin-top: 15px;
    }

    .footer {
        border-top: 1px solid #E2EDE1;
        margin-top: 45px;
        padding-top: 22px;
        text-align: center;
        color: #748477;
        font-size: 0.85rem;
        line-height: 1.8;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 25px 20px;
        }

        .topbar {
            align-items: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# LOAD THE EXISTING TRAINED MODEL
# ---------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}"
        )
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found at: {CLASS_PATH}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        return file.read().splitlines()


try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Unable to load the model or class names: {error}")
    st.stop()


# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown(
    """
    <div class="topbar">
        <div class="brand">🌿 LeafCare<span> AI</span></div>
        <div class="top-tag">AI · PLANT HEALTH</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# HERO SECTION
# ---------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMARTER PLANT CARE</div>
        <div class="hero-title">
            Healthy leaves.<br>Smarter decisions.
        </div>
        <div class="hero-copy">
            Explore an AI-powered way to analyze tomato leaf images.
            Upload a photo to receive a predicted category and view
            the model's confidence scores in seconds.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# FEATURE CARDS
# ---------------------------------------------
feature_columns = st.columns(3)

features = [
    ("🌱", "Leaf analysis", "Upload a clear image of a tomato leaf."),
    ("🧠", "Deep learning", "Predictions from your trained CNN model."),
    ("📊", "Score breakdown", "Compare scores across all six categories.")
]

for column, (icon, title, description) in zip(feature_columns, features):
    with column:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-copy">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("")

# ---------------------------------------------
# IMAGE UPLOAD SECTION
# ---------------------------------------------
st.markdown(
    """
    <div class="section-label">01 · UPLOAD</div>
    <div class="section-heading">Analyze a tomato leaf</div>
    <div class="section-copy">
        Choose a JPG, JPEG or PNG image. A clear, well-lit leaf
        photograph usually provides a better input for the model.
    </div>
    """,
    unsafe_allow_html=True
)

left_column, right_column = st.columns([1, 1], gap="large")

with left_column:
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"],
        help="This model was trained on six tomato leaf categories."
    )

    st.caption("Supported formats: JPG, JPEG and PNG")

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(
                uploaded_image,
                caption="Your uploaded tomato leaf image",
                use_container_width=True
            )
        except Exception:
            st.error("This image could not be opened. Try another file.")
            st.stop()

with right_column:
    st.markdown(
        """
        <div class="feature-card">
            <div class="section-label">HOW IT WORKS</div>
            <div class="feature-title">Three simple steps</div>
            <br>
            <div class="feature-copy">
                <b>1.</b> Upload a tomato leaf image.<br><br>
                <b>2.</b> Select the prediction button.<br><br>
                <b>3.</b> Review the predicted category and scores.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "This model recognizes six tomato leaf categories only. "
        "It is not designed to identify every plant species."
    )

    predict_button = st.button(
        "🌿  Detect Disease",
        disabled=uploaded_file is None,
        use_container_width=True
    )

# ---------------------------------------------
# PREDICTION AND RESULTS
# ---------------------------------------------
if uploaded_file is not None and predict_button:
    try:
        with st.spinner("Analyzing your image..."):
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((128, 128))

            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]

            if len(predictions) != len(class_names):
                st.error(
                    "The model output does not match the saved class names."
                )
                st.stop()

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        st.markdown("---")

        st.markdown(
            """
            <div class="section-label">02 · RESULTS</div>
            <div class="section-heading">Your analysis results</div>
            <div class="section-copy">
                The following results show the model's predicted category
                and its relative scores for the supplied image.
            </div>
            """,
            unsafe_allow_html=True
        )

        result_column, confidence_column = st.columns([1.5, 1])

        with result_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted category</div>
                    <div class="result-name">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with confidence_column:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Model confidence score</div>
                    <div class="confidence-number">{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Category-wise scores")

        st.caption(
            "Higher scores indicate which categories the model favors "
            "for this image. These scores are not guaranteed probabilities "
            "of real-world correctness."
        )

        sorted_indices = np.argsort(predictions)[::-1]

        for index in sorted_indices:
            category = class_names[index]
            score = float(predictions[index])

            score_column, percentage_column = st.columns([5, 1])

            with score_column:
                st.write(category)
                st.progress(max(0.0, min(1.0, score)))

            with percentage_column:
                st.write(f"**{score * 100:.2f}%**")

        st.markdown(
            """
            <div class="notice">
                <b>Important:</b> This is an academic AI project.
                A high confidence score does not guarantee a correct
                diagnosis. Image quality, lighting, background and
                differences from the training dataset can affect results.
                Consult an agricultural expert before making treatment
                decisions.
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown(
    """
    <div class="footer">
        <b>LeafCare AI</b> · Plant Disease Detection<br>
        Built with Python, TensorFlow and Streamlit<br>
        Academic project · Experimental predictions only
    </div>
    """,
    unsafe_allow_html=True
)

