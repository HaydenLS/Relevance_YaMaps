## Запуск по отдельным примерам.

import os
import json
import streamlit as st
import pandas as pd
# from contextlib import redirect_stdout, redirect_stderr
import io

from langchain_openai import ChatOpenAI

from org_relevance.config import CONFIG
from org_relevance.agent.graph import build_graph
from org_relevance.data.dataset import get_template_from_row


LLM_PROVIDERS = {
    "OpenRouter": {
        "default_model": "openrouter/aurora-alpha",
        "default_base_url": "https://openrouter.ai/api/v1",
        "env_key_name": "OPENROUTER_API_KEY",
    },
    "OpenAI": {
        "default_model": "gpt-4o-mini",
        "default_base_url": "https://api.openai.com/v1",
        "env_key_name": "OPENAI_API_KEY",
    }
}

# Загрузка документа (будет в сайдбаре)
@st.cache_data(show_spinner=False)
def load_df_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_json(io.BytesIO(file_bytes), lines=True)

def load_df(uploaded_file) -> pd.DataFrame:
    """
    uploaded_file: st.file_uploader output (UploadedFile) or None
    """
    file_bytes = uploaded_file.getvalue()
    return load_df_from_bytes(file_bytes)



@st.cache_resource(show_spinner=False)
def get_graph_cached(provider_name: str, model_name: str, api_key: str, base_url: str):
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    graph = build_graph(llm)
    return graph


def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


st.title("Single Query")

with st.sidebar:
    st.header("Настройки")

    # --- Datset ---
    uploaded_dataset = st.file_uploader(
        "Загрузить датасет (jsonl)",
        type=["jsonl", "json"],
        accept_multiple_files=False,
    )

    # --- LLM provider ---
    provider_name = st.selectbox("LLM provider", list(LLM_PROVIDERS.keys()), index=0)
    provider = LLM_PROVIDERS[provider_name]

    model_name = st.text_input("Model", value=provider["default_model"])
    base_url = st.text_input("Base URL", value=provider["default_base_url"])

    env_key_name = provider["env_key_name"]
    api_key = st.text_input(
        env_key_name,
        value=os.getenv(env_key_name, ""),
        type="password",
    )

    web_provider = st.selectbox("Web provider", ["duckduckgo", "ollama"], index=0)
    CONFIG.web_provider = web_provider


# ---------- ВАЛИДАЦИЯ ---------
if uploaded_dataset is None:
    st.error("Не загружен файл")
    st.stop()
try:
    df = load_df(uploaded_dataset)
except Exception as e:
    st.exception(e)
    st.stop()


if df.empty:
    st.error("Датасет пустой или не прочитался.")
    st.stop()

max_id = len(df) - 1

col_a, col_b = st.columns([1, 2])
with col_a:
    row_id = st.number_input("ID (индекс строки)", min_value=0, max_value=max_id, value=0, step=1)

with col_b:
    st.write(f"Размер датасета: **{len(df)}** строк. Выбран ID: **{row_id}**")

row = df.iloc[int(row_id)]
basic_state = get_template_from_row(row)


# ---------- ПОКАЗ ИСХОДНЫХ ДАННЫХ ----------
st.subheader("Исходные данные / шаблон состояния")
tabs = st.tabs(["Row (raw)", "basic_state"])

# показать строку в сыров виде и в вииде AgentState
with tabs[0]:
    st.json(row.to_dict())
with tabs[1]:
    st.code(safe_json(basic_state), language="json")


# ---------- ИНИЦИАЛИЗАЦИЯ ГРАФА ----------
if not api_key:
    st.warning("Api пуст. Задайте ключ в sidebar или через переменную окружения.")
    st.stop()

graph = get_graph_cached(provider_name, model_name, api_key, base_url)

if CONFIG.DEBUG:
    st.caption("DEBUG включён в CONFIG")


# ---------- ЗАПУСК ----------
st.subheader("Запуск")

run = st.button("Run graph.invoke()", type="primary")

if run:
    with st.spinner("Выполняю граф..."):
        try:
            
            result = graph.invoke(basic_state)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("Готово")

    # Результаты
    st.subheader("Результат")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Реальная метка", value=str(result.get("relevance_true")))
    with col2:
        st.metric("Метка модели", value=str(result.get("relevance_model")))

    st.markdown("**Вердикт модели (reason):**")
    st.write(result.get("reason", ""))

    with st.expander("Полный result"):
        st.code(safe_json(result), language="json")