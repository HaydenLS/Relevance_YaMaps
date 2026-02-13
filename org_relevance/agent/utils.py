# -----
# в этом коде можно будет убрать метод _clamp_relevance, но пока не критично
# -----

from typing import Dict, Any
import json

from org_relevance.common.types import Relevance

def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Минимальная и практичная вырезка JSON-объекта из ответа LLM.
    Ожидаем, что модель вернёт JSON-объект.
    """
    text = text.strip()
    # Попытка прямого парсинга
    try:
        return json.loads(text)
    except Exception:
        pass

    # Попытка вырезать по первым или последним фигурным скобкам
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        return json.loads(chunk)

    raise ValueError("Не удалось распарсить JSON из ответа модели")



def _clamp_relevance(x: Any) -> Relevance:
    # Допускаем, что LLM вернёт число или строку.
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v == 1.0:
        return 1.0
    if v == 0.1:
        return 0.1
    return 0.0