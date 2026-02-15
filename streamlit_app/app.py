import os
import streamlit as st

st.set_page_config(
    page_title="Org Relevance",
    layout="wide",
)

st.title("DLS Yamaps Relevance - Streamlit UI")

st.markdown(
"""
Это оболочка над пайплайном.

Откройте страницу **Single Query** слева (в меню Pages), чтобы:
- загрузить датасет
- выбрать строку датасета по ID
- увидеть исходные данные и метку класса
- запустить `graph.invoke(basic_state)`
- посмотреть `relevance` и `reason`

Откройте страницу **Batch Run** слева (в меню Pages), чтобы:
- загрузить файл с данными .jsonl
- выбрать необходимые строки
- запустить агента
- после получить файл с ответами
- посмотреть полученные метрики
"""
)

