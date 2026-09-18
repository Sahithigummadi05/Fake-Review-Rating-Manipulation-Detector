"""
Streamlit demo: paste a review, get a deceptiveness score plus the
interpretable feature breakdown behind it.

    python -m src.real_text_model   # once, to train + save the model
    streamlit run app_streamlit.py
"""

from pathlib import Path

import streamlit as st

from src.real_text_model import MODEL_PATH, explain_review, top_indicative_terms

st.set_page_config(page_title="Fake Review Detector", page_icon="🔍")
st.title("🔍 Fake Review Detector")
st.caption(
    "Text-suspicion model trained on the Ott Deceptive Opinion Spam Corpus "
    "(1,600 gold-labeled hotel reviews). ~89% CV accuracy."
)

if not Path(MODEL_PATH).exists():
    st.error("Model not found. Run `python -m src.real_text_model` first.")
    st.stop()

review = st.text_area(
    "Paste a review",
    height=160,
    placeholder="We had the most amazing luxurious experience, definitely the best hotel ever...",
)

if st.button("Analyze") and review.strip():
    result = explain_review(review)
    prob = result["prob_deceptive"]

    st.metric("Probability deceptive", f"{prob:.0%}", result["verdict"].upper())
    st.progress(prob)

    st.subheader("Why — linguistic signals")
    st.table(
        {"feature": list(result["linguistic"].keys()),
         "value": [f"{v:.3f}" for v in result["linguistic"].values()]}
    )

with st.expander("Words the model finds most indicative"):
    terms = top_indicative_terms()
    col1, col2 = st.columns(2)
    col1.markdown("**Deceptive**\n\n" + ", ".join(terms["deceptive"]))
    col2.markdown("**Truthful**\n\n" + ", ".join(terms["truthful"]))
