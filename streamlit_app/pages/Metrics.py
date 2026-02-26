import io
import json
import pandas as pd
import streamlit as st

from org_relevance.evaluation.evaluate import calculate_metrics, calculate_confusion_matrix

def safe_json(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


@st.cache_data(show_spinner=False)
def read_jsonl_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_json(io.BytesIO(file_bytes), lines=True)


st.title("Metrics from batches")

st.write("Загрузите несколько файлов батчей (.jsonl). После объединения будут посчитаны метрики.")

uploaded_files = st.file_uploader(
    "Drag & drop batch jsonl files here",
    type=["jsonl", "json"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Файлы не загружены.")
    st.stop()

dfs = []
file_summaries = []

for f in uploaded_files:
    try:
        df_part = read_jsonl_bytes(f.getvalue())
        df_part["__source_file"] = f.name
        dfs.append(df_part)
        file_summaries.append((f.name, len(df_part), list(df_part.columns)))
    except Exception as e:
        st.error(f"Не удалось прочитать файл {f.name}: {e}")
        st.stop()

st.subheader("Файлы")
for name, nrows, cols in file_summaries:
    st.write(f"- {name}: {nrows} строк, колонки: {', '.join(cols[:15])}" + (" ..." if len(cols) > 15 else ""))

full_df = pd.concat(dfs, ignore_index=True)

# Проверка обязательных колонок
required = {"relevance", "relevance_model"}
missing = required - set(full_df.columns)
if missing:
    st.error(f"В объединённых данных не хватает колонок: {', '.join(sorted(missing))}")
    st.stop()

# Сортировка, если есть row_index (ваш новый формат)
if "row_index" in full_df.columns:
    # если вдруг row_index строковый/с NaN — аккуратно приводим
    full_df["row_index"] = pd.to_numeric(full_df["row_index"], errors="coerce")
    # сортируем только если есть хоть какие-то валидные индексы
    if full_df["row_index"].notna().any():
        full_df = full_df.sort_values("row_index", kind="stable").reset_index(drop=True)

st.subheader("Объединённый датафрейм")
st.write(f"Всего строк: {len(full_df)}")

# Фильтрация NaN для метрик
nan_true = int(full_df["relevance"].isna().sum())
nan_pred = int(full_df["relevance_model"].isna().sum())

st.write(f"NaN в y_true (relevance): {nan_true}")
st.write(f"NaN в y_pred (relevance_model): {nan_pred}")

df_for_metrics = full_df.dropna(subset=["relevance", "relevance_model"]).copy()

if df_for_metrics.empty:
    st.warning("После удаления NaN не осталось строк для расчёта метрик.")
    st.stop()

# Метрики
st.subheader("Classification report")
classes_df = calculate_metrics(
    relevance_true=df_for_metrics["relevance"],
    relevance_model=df_for_metrics["relevance_model"],
)
st.dataframe(classes_df, use_container_width=True)

st.subheader("Confusion matrix")
cm_df = calculate_confusion_matrix(
    relevance_true=df_for_metrics["relevance"],
    relevance_model=df_for_metrics["relevance_model"],
)
st.dataframe(cm_df, use_container_width=True)

# Опционально: выгрузка объединённых данных
with st.expander("Скачать объединённый файл"):
    jsonl_data = full_df.to_json(orient="records", lines=True, force_ascii=False)
    st.download_button(
        label="Скачать merged.jsonl",
        data=jsonl_data,
        file_name="merged.jsonl",
        mime="application/json",
    )
