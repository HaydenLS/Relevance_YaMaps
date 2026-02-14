## Запуск по батчам

import os
import json
import pandas as pd
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr
import io

from langchain_openai import ChatOpenAI

from org_relevance.config import CONFIG
from org_relevance.agent.graph import build_graph
from org_relevance.data.dataset import get_template_from_row, prepare_dataset
from org_relevance.evaluation.evaluate import calculate_metrics


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


st.title("Batch Run")


with st.sidebar:
    st.header("Настройки")

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
uploaded_file = st.file_uploader("Загрузите файл .jsonl", type="jsonl")

if uploaded_file is None:
    st.error("Не выбран файл для загрузки")
    st.stop()

# Загружаем файл
df = load_df(uploaded_file)
if df.empty:
    st.error("Датасет пустой или не прочитался.")
    st.stop()

max_id = len(df) - 1

# ---------- ВЫБОР ДИАПАЗОНА СТРОК ДЛЯ ОБРАБОТКИ ----------
st.subheader("Выберите диапазон строк для обработки")

start_idx = st.slider("Начальный индекс", 0, max_id, 0)
end_idx = st.slider("Конечный индекс", start_idx, max_id, max_id)

st.text(f"Выбрано {end_idx - start_idx + 1} строк")

# Подготовим данные для выбранного диапазона
selected_df = df.iloc[start_idx:end_idx+1]

prepared_data = prepare_dataset(selected_df)


# ---------- ИНИЦИАЛИЗАЦИЯ ГРАФА ----------
if not api_key:
    st.warning("Api пуст. Задайте ключ в sidebar или через переменную окружения.")
    st.stop()

graph = get_graph_cached(provider_name, model_name, api_key, base_url)

# ---------- ПРОГРЕСС-БАР И ЗАПУСК АГЕНТА ----------
st.subheader("Запуск агента")

progress_bar = st.progress(0)
results = []

for idx, (basic_state, _) in prepared_data.iterrows():
    result = None
    try:
        result = graph.invoke(basic_state)
        results.append(result)
    except Exception as e:
        st.error(f"Ошибка на строке {idx}: {e}")
        continue
    progress_bar.progress(int((idx + 1) / len(prepared_data) * 100))  # обновление прогресса

# ---------- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ ----------
st.subheader("Сохранение результатов")

# Преобразуем результаты в DataFrame для сохранения
result_df = pd.DataFrame(results)

output_file = "results.jsonl"
result_df.to_json(output_file, orient="records", lines=True)

st.download_button(
    label="Скачать результаты",
    data=open(output_file, "r").read(),
    file_name=output_file,
    mime="application/json",
)

# ---------- ВЫВОД МЕТРИК ----------
st.subheader("Метрики")

calculate_metrics(result_df)