# -----
# в этом коде можно будет убрать метод _clamp_relevance, но пока не критично
# -----

from typing import Dict, Any
import json
import time
import random

from org_relevance.common.types import Relevance
from org_relevance.web.providers import ollama_web_search, duckduckgo_web_search

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


## Методы для обработки ошибок LLM

def retry_call(fn, *, retries=2, base_delay=0.8, max_delay=10.0, retry_on=(Exception,)):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except retry_on as e:
            last = e
            if attempt >= retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay = delay * (0.7 + 0.6 * random.random())  # jitter
            time.sleep(delay)
    raise last

# Ошибка при достижении Rate Limit
class RateLimitExhausted(RuntimeError):
    """Ошибка появится, когда у api больше не будет запросов (кончились деньги например)"""
    pass

def safe_llm_invoke(llm, msg: str, retries=2):
    def _call():
        return llm.invoke(msg).content
    
    try:
        return retry_call(_call, retries=retries)
    except Exception as e:
        if "rate" in str(e).lower() or "429" in str(e):
            raise RateLimitExhausted(f"LLM rate limit: {e}") from e 
        raise

## Метод для обработки ошибок WebSearch
def safe_web_search(q: str, provider: str, retries=2):
    def _call():
        if "ollama" in provider.lower():
            return ollama_web_search(q)
        if "duck" in provider.lower():
            return duckduckgo_web_search(q)
        else:
            raise ValueError("Unknown web provider")
    return retry_call(_call, retries=retries)


## Json Repair Prompt

REPAIR_PROMPT = "!!! Верни СТРОГО ТОЛЬКО валидный JSON без текста вокруг !!!"

def parse_json_or_repair(raw: str, llm=None, retries=1):
    # 1) обычная попытка
    try:
        return _extract_json_object(raw)
    except Exception:
        pass

    # 2) repair-попытка через LLM
    if llm is None or retries <= 0:
        raise ValueError("JSON parse failed")

    for _ in range(retries):
        repaired = safe_llm_invoke(llm, REPAIR_PROMPT + raw, retries=1)
        try:
            return _extract_json_object(repaired)
        except Exception:
            continue

    raise ValueError("JSON parse failed after repair")
