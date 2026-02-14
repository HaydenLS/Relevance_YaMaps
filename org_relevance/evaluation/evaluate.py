import pandas as pd
from sklearn.metrics import classification_report


def calculate_metrics(result_dataframe: pd.DataFrame) -> None:
    """
    Рассчет метрики по датасету
    
    """

    ind2label = {0:0.0, 1:1.0, 2:0.1}
    label2ind = {0.0:0, 1.0:1, 0.1:2}

    y_trues = result_dataframe['relevance'].map(label2ind)
    y_preds = result_dataframe['relevance_llm'].map(label2ind)


    report = classification_report(y_trues, y_preds)

    print(report)
