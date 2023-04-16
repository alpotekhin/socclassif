"""Runs the app"""

import streamlit as st
import shap

from transformers import pipeline
from streamlit_shap import st_shap
from spacy_streamlit import visualize_ner
from huggingface_hub import login
from supabase import create_client


@st.cache_resource
def init_database_connection():
    """Connects to database"""
    url = st.secrets["database_url"]
    key = st.secrets["database"]
    return create_client(url, key)


@st.cache_data(ttl=600)
def run_database_query(data):
    """Inserts new data into the database"""
    database = init_database_connection()
    return database.table("socclassif").insert(data).execute()


@st.cache_resource(show_spinner=True)
def load_seq_classifier():
    """Loads pre-trained model from huggingface hub and wraps
    it into a sequence classification pipeline

    Returns:
        Pipeline (object): huggingface Pipeline class
    """
    classifier = pipeline(
        "text-classification",
        model="alpotekhin/rubert-tiny2-socclassif",
        tokenizer="cointegrated/rubert-tiny2",
        return_all_scores=True,
    )
    return classifier


@st.cache_resource(show_spinner=True)
def load_ner():
    """Loads pre-trained model from huggingface hub and wraps
    it into a token classification pipeline

    Returns:
        Pipeline (object):  huggingface Pipeline class
    """
    ner_model = pipeline(
        "ner",
        model="viktoroo/sberbank-rubert-base-collection3",
        tokenizer="viktoroo/sberbank-rubert-base-collection3",
        aggregation_strategy="simple",
    )
    return ner_model


def plot_shap(input, seq_classifier):
    """Plots interpretation of classification result based on SHAP values

    Args:
        input (str): user input
        seq_classifier (object): huggingface Pipeline
    """
    shap_model = shap.models.TransformersPipeline(
        seq_classifier, rescale_to_logits=False
    )
    explainer2 = shap.Explainer(shap_model)
    shap_values2 = explainer2([input])[0]
    st_shap(shap.plots.text(shap_values2), height=200)


def plot_ner(input, ner_result):
    """Plots interpretation of classification result based on SHAP values
    from https://github.com/slundberg/shap

    Args:
        input (str): user input
        ner_result (list[dict[str]])): NER pipeline output
    """
    ner_spans = [
        {
            "text": input,
            "ents": [
                {
                    "start": span["start"],
                    "end": span["end"],
                    "label": span["entity_group"],
                }
                for span in ner_result
            ],
        }
    ]
    visualize_ner(
        ner_spans,
        show_table=False,
        manual=True,
        labels=["ORG", "LOC", "PER"],
        colors={"ORG": "#9cc3ff", "LOC": "#ff7070", "PER": "#dddddd"},
        title=None,
    )


def show_classif_res(input, seq_classifier):
    """Shows result of sequence classification
    and gets the feedback

    Args:
        input (str): user input
        seq_classifier (object): huggingface Pipeline
    """
    st.markdown("**Classification results**")
    with st.spinner("Performing classification..."):
        plot_shap(input, seq_classifier)

    label = st.radio(
        "Please select the right label to feed model more data",
        ("other", "housing", "work"),
        horizontal=True,
    )
    submit_btn = st.button("Submit")

    if submit_btn:
        st.success(f"**{input}** is labeled as **{label}**, thank you!")
        # run_database_query({"sequence": input, "label": label}) 


def show_ner_res(input, ner_result):
    """Shows result of sequence classification
    and gets the feedback

    Args:
        input (str): user input
        ner_result (list[dict[str]])): NER pipeline output
    """
    with st.spinner("Performing NER..."):
        st.markdown("**NER results**")
        plot_ner(input, ner_result)


def run():
    st.set_page_config(
        layout="centered", page_icon=":clipboard:", page_title="Social classification"
    )
    login(st.secrets["huggingface_login"])

    seq_classifier = load_seq_classifier()
    token_classifier = load_ner()

    st.title("Social classification")
    st.markdown(
        "This mini-application solves the tasks of named entity recognition \
            and classification of a text query into the 'work' and 'housing' labels"
    )

    input = st.text_input("Write text here")
    run_button = st.button("Get prediction", type="primary")

    if run_button or input:
        if input == "":
            st.error("Please input the text")
            return

        ner_result = token_classifier(input)

        if ner_result:  # if no NER rusult -> use only 1 tab
            tab1, tab2 = st.tabs(["Classification", "NER"])
            with tab1:
                show_classif_res(input, seq_classifier)

            with tab2:
                show_ner_res(input, ner_result)
        else:
            show_classif_res(input, seq_classifier)


run()
