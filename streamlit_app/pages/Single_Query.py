## Запуск по отдельным примерам.


import os
import json
import streamlit as st
import pandas as pd
from contextlib import redirect_stdout, redirect_stderr
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


@st.cache_data(show_spinner=False)
def load_df(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


@st.cache_resource(show_spinner=False)
def get_graph_cached(provider_name: str, model_name: str, api_key: str, base_url: str):
    # Важно: cache_resource кэшируется по аргументам функции.
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

    dataset_path = st.text_input("Путь к датасету (jsonl)", value=str(CONFIG.train_path))

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


# ---------- ВАЛИДАЦИЯ ----------
if not dataset_path:
    st.error("Не задан путь к датасету")
    st.stop()

df = load_df(dataset_path)

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
basic_state, relevance_true = get_template_from_row(row)


# ---------- ПОКАЗ ИСХОДНЫХ ДАННЫХ ----------
st.subheader("Исходные данные / шаблон состояния")
tabs = st.tabs(["Row (raw)", "basic_state", "relevance_true"])

with tabs[0]:
    # Показать строку как dict
    st.json(row.to_dict())

with tabs[1]:
    st.code(safe_json(basic_state), language="json")

with tabs[2]:
    st.write(relevance_true)


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
    log_buf = io.StringIO()
    err_buf = io.StringIO()

    with st.spinner("Выполняю граф..."):
        try:
            with redirect_stdout(log_buf), redirect_stderr(err_buf):
                result = graph.invoke(basic_state)
        except Exception as e:
            st.exception(e)
            # покажем то, что успело залогироваться
            logs = log_buf.getvalue()
            errs = err_buf.getvalue()
            if logs:
                st.subheader("Logs")
                st.text_area("stdout", logs, height=250)
            if errs:
                st.subheader("Errors")
                st.text_area("stderr", errs, height=250)
            st.stop()

    st.success("Готово")

    # Результаты
    st.subheader("Результат")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Реальная метка", value=str(relevance_true))
    with col2:
        st.metric("Метка модели", value=str(result.get("relevance")))

    st.markdown("**Вердикт модели (reason):**")
    st.write(result.get("reason", ""))

    with st.expander("Полный result"):
        st.code(safe_json(result), language="json")

    # Логи
    logs = log_buf.getvalue()
    errs = err_buf.getvalue()

    with st.expander("Logs (stdout)"):
        st.text_area("stdout", logs, height=300)

    if errs.strip():
        with st.expander("Errors (stderr)"):
            st.text_area("stderr", errs, height=200)