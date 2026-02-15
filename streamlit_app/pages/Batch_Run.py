## Запуск по батчам

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from contextlib import redirect_stdout, redirect_stderr
import io

from langchain_openai import ChatOpenAI

from org_relevance.config import CONFIG
from org_relevance.agent.graph import build_graph
from org_relevance.data.dataset import get_template_from_row
from org_relevance.evaluation.evaluate import calculate_metrics, calculate_confusion_matrix


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


# ------- Боковая панель --------
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



# ------- Загрузка файла ---------
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



# --------- Выбор диапазона строк -----------
st.subheader("Диапазон строк для обработки")

col1, col2 = st.columns(2)
with col1:
    start_text = st.text_input("Начальный индекс", value="0")
with col2:
    end_text = st.text_input("Конечный индекс", value=str(max_id))

def parse_int(text, default):
    try:
        return int(str(text).strip())
    except:
        return default
    
start_idx = parse_int(start_text, 0)
end_idx = parse_int(end_text, max_id)

if start_idx < 0:
    start_idx = 0
if end_idx > max_id:
    end_idx = max_id
if end_idx < start_idx:
    st.error("Конечный индекс должен быть >= начального.")
    st.stop()

st.text(f"Выбрано {end_idx - start_idx + 1} строк (индексы {start_idx}..{end_idx})")

# Подготовим данные для выбранного диапазона
selected_df = df.iloc[start_idx:end_idx+1]
# prepared_data = prepare_dataset(selected_df) # prepared data - датафрейм их tuplов



# ---------- Создание графа -------------
if not api_key:
    st.warning("Api пуст. Задайте ключ в sidebar или через переменную окружения.")
    st.stop()

graph = get_graph_cached(provider_name, model_name, api_key, base_url)



# --------- Запуск программы --------
st.subheader("Запуск агента")

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ----------- Состяния для правильного сохранения---------
if "batch_ready" not in st.session_state:
    st.session_state.batch_ready = False
if "jsonl_data" not in st.session_state:
    st.session_state.jsonl_data = None
if "full_result_df" not in st.session_state:
    st.session_state.full_result_df = None
if "classes_df" not in st.session_state:
    st.session_state.classes_df = None
if "nan_count" not in st.session_state:
    st.session_state.nan_count = 0
if "cm_df" not in st.session_state:
    st.session_state.cm_df = None


# --------- Создание удобных кнопок --------
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    start_clicked = st.button("Запуск агента", type="primary", disabled=st.session_state.is_running)
with btn_col2:
    stop_clicked = st.button("Остановить", disabled=not st.session_state.is_running)

if stop_clicked:
    st.session_state.stop_requested = True

# ----------- Нажата кнопка старта -----------
if start_clicked:
    st.session_state.stop_requested = False
    st.session_state.is_running = True

    results = []
    
    # Progress bar
    total = len(selected_df)
    progress_bar = st.progress(0)
    processed = 0

    for idx, row in selected_df.iterrows():
        # Проверка на стоп
        if st.session_state.stop_requested:
            st.warning("Остановка.")
            break

        try:
            # делам из строки agent_state
            basic_state = get_template_from_row(row)
            # старт агента
            agent_result = graph.invoke(basic_state)

            # получение результата
            result = {"relevance_model": agent_result.get('relevance_model', np.nan),
                      "reason": agent_result.get('reason', np.nan)}

        except Exception as e:
            st.error(f"Ошибка на строке {idx}: {e}")
            result = {"relevance_model": np.nan,
                      "reason": np.nan}


        results.append(result)
        
        processed += 1
        progress_bar.progress(int(processed / total * 100))

    st.session_state.is_running = False
    if not results:
        st.warning("Нет результатов (все упали или остановлено до первого шага).")
        st.stop()

    # --------- Сохранение результатов в файл -----------
    st.subheader("Сохранение результатов")

    # Преобразуем результаты в DataFrame для сохранения
    result_df = pd.DataFrame(results)
    # Объединяем
    full_result_df = pd.concat([selected_df, result_df], axis=1)

    jsonl_data = full_result_df.to_json(orient="records", lines=True, force_ascii=False)

    nan_count = int(full_result_df["relevance_model"].isna().sum())
    
    df_for_metrics = full_result_df.dropna(subset=["relevance_model"]).copy()

    # --- Рассчет метрик в вывод их --
    st.subheader("Метрики")
    # 1) Подсчитаем пропуски и выкинем их
    st.text(f"Всего строк с Nan: {full_result_df['relevance_model'].isna().sum()}")
    full_result_df.dropna(subset=['relevance_model'], inplace=True)

    # --- Classification report ---
    classes_df = calculate_metrics(
        relevance_true=full_result_df["relevance"],
        relevance_model=full_result_df["relevance_model"],
    )
    

    cm_df = calculate_confusion_matrix(
        relevance_true=full_result_df["relevance"],
        relevance_model=full_result_df["relevance_model"],
    )

    # Обновляем состояния
    st.session_state.jsonl_data = jsonl_data
    st.session_state.full_result_df = full_result_df
    st.session_state.classes_df = classes_df
    st.session_state.cm_df = cm_df
    st.session_state.nan_count = nan_count
    st.session_state.batch_ready = True


# ---- Конечный вывол в программу ----
if st.session_state.batch_ready:
    st.subheader("Сохранение результатов")

    st.download_button(
        label="Скачать результат",
        data=st.session_state.jsonl_data,
        file_name="results.jsonl",
        mime="application/json",
    )

    st.subheader("Метрики")
    st.text(f"Всего строк с Nan: {st.session_state.nan_count}")
    st.subheader("Classification Report")
    st.dataframe(st.session_state.classes_df)
    st.subheader("Confusion Matrix")
    st.dataframe(st.session_state.cm_df)