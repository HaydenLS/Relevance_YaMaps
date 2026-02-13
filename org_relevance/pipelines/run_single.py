import numpy as np
import pandas as pd
import os

from langchain_openai import ChatOpenAI


from org_relevance.config import CONFIG
from org_relevance.agent.graph import build_graph

from org_relevance.data.dataset import get_template_from_row


if __name__ == "__main__":

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = 'openrouter/aurora-alpha'

    llm = ChatOpenAI(
        model=model_name, # Specify a model available on OpenRouter
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0
    )

    graph = build_graph(llm)

    if CONFIG.DEBUG:
        print("[run_single] Graph is builded successfuly")



    data_train = pd.read_json(CONFIG.train_path, lines=True)

    
    wait = 0
    while not wait:
        check_id = int(input("ID:"))
        basic_state, relevance_true = get_template_from_row(data_train.iloc[check_id])
        print(basic_state)
        print(relevance_true)

        wait = int(input("Start? 1- yes, 0- no: "))


    result = graph.invoke(basic_state)
    print(result)

    print('\n\n Итог:')
    print(f"Реальная метка: {relevance_true}")
    print(f"Метка модели:{ result['relevance']}")
    print(f"Вердикт модели: {result['reason']}")
    print("\n")







