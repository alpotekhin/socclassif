# Social classification demo

**This is a demo of the project to create a sociolinguistic classifier for the Russian language. At this stage, the possibility of recognizing the labels "work" and "housing" is embodied, as well as Named Entity Recognition task. Feel free to [try it out!](https://alpotekhin-socclassif.streamlit.app/)**

## Classification

We use [tiny Rubert](https://huggingface.co/cointegrated/rubert-tiny2) with classification head and [SHAP](https://github.com/slundberg/shap) for prediction interpretation

![classif_example](https://github.com/alpotekhin/socclassif/blob/master/imgs/class_pic.png)

## NER

NER-model is [Rubert-collection3](https://huggingface.co/viktoroo/sberbank-rubert-base-collection3) — fine-tuned version of [sberbank-ai/ruBert-base](https://huggingface.co/ai-forever/ruBert-base)


![ner_example](https://github.com/alpotekhin/socclassif/blob/master/imgs/ner_pic.png)
