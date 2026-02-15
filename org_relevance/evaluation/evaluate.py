import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

IND2LABEL = {0:0.0, 1:1.0, 2:0.1}
LABEL2IND = {0.0:0, 1.0:1, 0.1:2}
LABEL_ORDER = [0.0, 1.0, 0.1]

def calculate_metrics(relevance_true: pd.Series, relevance_model: pd.Series) -> pd.DataFrame:
    """
    Рассчет метрики по датасету
    Вернет DataFrame
    """

    y_trues = relevance_true.map(LABEL2IND)
    y_preds = relevance_model.map(LABEL2IND)

    report = classification_report(y_trues, y_preds, output_dict=True)
    # Таблица по классам
    classes_df = pd.DataFrame(report).T.rename_axis("label")

    return classes_df
    
def calculate_confusion_matrix(relevance_true: pd.Series, relevance_model: pd.Series) -> pd.DataFrame:
    """
    Возвращает confusion matrix как DataFrame
    Строки = истинные метки, столбцы = предсказанные
    """
    y_trues = relevance_true.map(LABEL2IND)
    y_preds = relevance_model.map(LABEL2IND)


    cm = confusion_matrix(
        y_trues,
        y_preds,
        labels=[0, 1, 2], 
    )

    labels = [str(x) for x in LABEL_ORDER]
    return pd.DataFrame(cm, index=pd.Index(labels, name="true"), columns=pd.Index(labels, name="pred"))