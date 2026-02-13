import re
import html
import json
from typing import Literal


from org_relevance.common.types import AgentState, Organization, Relevance
from org_relevance.agent.utils import _extract_json_object, _clamp_relevance
from org_relevance.prompts.all_promtps import  ROUTER_PROMPT, MAKE_SEARCH_QUERY_PROMPT, CLASSIFIER_PROMPT_SINGLE

from org_relevance.web.providers import ollama_web_search, duckduckgo_web_search
from org_relevance.web.retrieve import retrieve_top_chunks_on_the_fly

from org_relevance.config import CONFIG

def router_node(state: AgentState, llm) -> dict:
    """
    Router node
    """

    # если лимит веба исчерпан - принудительно решаем
    if state.get("web_tries", 0) >= state.get("max_web_tries", 2):
        if CONFIG.DEBUG:
            print(f"\t[LOG] Web Limit. Forcing model to classify.")\
        
        return {
        "can_decide_now": True,
        "router_reason": "Исчерпан лимит веб поиска. Выполни классификацию по всем имеющимся данным"
        }

    # если нет - спрашиваем у модели ответ
    org = state["organization"]
    payload = {
        "query": state["query"],
        "organization": org,
        "prev_router_reason": state.get("router_reason", ""),
        "web_query": state.get("web_search_query", ""),
        "web_evidence": state.get("org_web_evidence", "")
    }

    msg = ROUTER_PROMPT + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
    if CONFIG.DEBUG:
        print(f"[LOG] Node: Router.  Prompting: {payload}")
    raw = llm.invoke(msg).content  # ожидаем объект с can_decide_now/reason

    obj = _extract_json_object(raw)

    can = bool(obj.get("can_decide_now", False))
    if CONFIG.DEBUG:
        print(f"\t[LOG] Model verdict: can decide now - {can}")
        print(f"\t[LOG] Model verdict: {str(obj.get('reason', ''))}")


    return {
        "can_decide_now": can,
        "router_reason": str(obj.get("reason", "")),
    }



def make_search_query_node(state: AgentState, llm) -> dict:
    """
    Generate a search query to get information

    """

    if CONFIG.DEBUG:
        print(f"[LOG] Node: Make search query.")
    org = state["organization"]
    payload = {
        "query": state["query"],
        "organization": {
            "name": org.get("name", ""),
            "category": org.get("category", ""),
            "address": org.get("address", ""),
        },
        "router_reason": state.get("router_reason", "")
    }

    msg = MAKE_SEARCH_QUERY_PROMPT + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
    raw = llm.invoke(msg).content
    obj = _extract_json_object(raw)


    q = str(obj.get("web_search_query", "")).strip()
    s = str(obj.get("search_phrase", "")).strip()

    if CONFIG.DEBUG:
        print(f"\t[LOG] Model generated Query: {q}")
        print(f"\t[LOG] Model generated Phrase: {s}")

    if not q:
        # fallback: простая эвристика
        q = f'{org.get("name","")} {org.get("address","")} {state["query"]}'.strip()

    return {"web_search_query": q,
            "search_phrase": s}




def web_search_node(state: AgentState) -> dict:
    """
    Нода выполняет запрос к LLM для формирования правильного запроса.
    В конце увеличивает счетчик попыток поиска.
    """
    q = state["web_search_query"]

    if CONFIG.DEBUG:
        print(f"[LOG] Node: web_search_node.")
        print(f"\t[LOG] Starting searching")

    result = ollama_web_search(q)

    if CONFIG.DEBUG:
        print(f"\t[LOG] Ending searching.")
        print(f"\t[LOG] Finded info: {result}"[:1000])

    tries = int(state.get("web_tries", 0)) + 1
    return {
        "web_search_result": result,
        "web_tries": tries,
    }


def augment_context_node(state: AgentState, model_embeddings) -> dict:
    """
    Добавляем веб-результат в накопленное поле org_web_evidence.
    Внимание! Тут старый результат поиска если он есть удаляется.
    Причина: старый результат нам не нужен, видимо запрос модели был неккоректен.
    """
    if CONFIG.DEBUG:
        print(f"[LOG] Node: augment_context_node")
        print(f"\t[LOG] Starting finding chunks...")


    web_results = state.get("web_search_result") or []
    query = state.get("search_phrase", "")
    if CONFIG.DEBUG:
        print(f"\t[LOG] Phrase: {query}")

    top_chunks = retrieve_top_chunks_on_the_fly(
        query,
        web_results=web_results,
        embedding_model=model_embeddings,
        top_k=5,
        min_chunk_len = 50,
        chunk_size=365,
        chunk_overlap=60,
        batch_size=64,
        max_total_chunks=2000,
        lexical_prefilter_topn=0 # отключим пока что фильтр
    )

    if CONFIG.DEBUG:
        print(f"\t[LOG] Chunks finded: {top_chunks}")

    return {"org_web_evidence": top_chunks}


def classify_node(state: AgentState, llm) -> dict:

    if CONFIG.DEBUG:
        print(f"[LOG] Node: classify_node")

    org = dict(state["organization"])
    org["web_query"] = state.get("web_search_query", "")
    org["web_evidence"] = state.get("org_web_evidence", "")

    current_case = {
        "id": state["id"],
        "query": state["query"],
        "organization": org
    }

    msg = CLASSIFIER_PROMPT_SINGLE + "\n\nINPUT:\n" + json.dumps(current_case, ensure_ascii=False)
    raw = llm.invoke(msg).content

    obj = _extract_json_object(raw)

    rel = _clamp_relevance(obj.get("relevance"))
    reason = str(obj.get("reason", "")).strip()


    if CONFIG.DEBUG:
        print(f"\t[LOG] Node: classify_node. Model verdict: relevance {rel}, reason {reason}")

    return {
        "relevance": rel,
        "reason": reason,
    }



def route_after_router(state: AgentState) -> Literal["classify", "make_search_query"]:
    return "classify" if state.get("can_decide_now", False) else "make_search_query"