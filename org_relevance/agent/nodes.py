import re
import html
import json
from typing import Literal
import logging

from org_relevance.common.types import AgentState, Organization, Relevance
from org_relevance.agent.utils import _extract_json_object, _clamp_relevance, safe_llm_invoke, safe_web_search, parse_json_or_repair, \
get_logger

from org_relevance.prompts.all_promtps import  ROUTER_PROMPT, ROUTER_FEW_SHOT, CLASSIFIER_PROMPT, CLASSIFIER_FEW_SHOT

from org_relevance.web.providers import ollama_web_search, duckduckgo_web_search
from org_relevance.web.retrieve import retrieve_top_chunks_on_the_fly

from org_relevance.config import CONFIG



def router_node(state: AgentState, llm) -> dict:
    """
    Router node
    """
    # get logger
    logger = get_logger()

    # ---- если лимит веба исчерпан - принудительно решаем
    if state.get("web_tries", 0) >= state.get("max_web_tries", 2):
        
        logger.info("[Router node] Web Limit. Forcing model to classify.")
        
        return {
        "can_decide_now": True,
        "router_reason": "Исчерпан лимит веб поиска. Выполни классификацию по всем имеющимся данным."
        }

    # ---- если нет спрашиваем у модели ответ
    org = state["organization"]
    payload = {
        "query": state["query"],
        "organization": org,
        "prev_router_reason": state.get("router_reason", ""),
        "web_query": state.get("web_search_query", ""),
        "web_evidence": state.get("org_web_evidence", "")
    }

    msg = ROUTER_PROMPT + "\n\nFEW-SHOT:\n" + ROUTER_FEW_SHOT + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)
    logger.info(f"[Router node] Node: Router.  Prompting: {payload}")


    # Запрос модели
    try:
        raw = safe_llm_invoke(llm, msg, retries=2)
    except Exception as e:
        logger.error(f"[Router node] Ошибка llm  \n{e}.")
        raise


    # Запрос чтения json
    try:
        obj = parse_json_or_repair(raw, llm=llm, retries=1)
    except Exception as e:
        logger.error(f"[Router node] Json Parse error \n{e}")
        raise
    

    can = bool(obj.get("can_decide_now", False))
    q = str(obj.get("web_search_query", "")).strip()
    s = str(obj.get("search_phrase", "")).strip()

    # Logging 
    logger.info(f"[Router node] Model verdict: can decide now - {can}")
    logger.info(f"[Router node] Model verdict: {str(obj.get('reason', ''))}")
    logger.info(f"[Router node] Model generated Query: {q}")
    logger.info(f"[Router node] Model generated Phrase: {s}")
    

    return {
        "can_decide_now": can,
        "web_search_query": q,
        "search_phrase": s,
    }


def web_search_node(state: AgentState) -> dict:
    """
    Нода выполняет запрос к LLM для формирования правильного запроса.
    В конце увеличивает счетчик попыток поиска.
    """
    logger = get_logger() # get logger
    logger.info(f"[web_search_node] Starting searching...")

    # запрос
    q = str(state.get("web_search_query", "")).strip()
    # всегда увеличиваем счетчик обращений к вебу
    tries = int(state.get("web_tries", 0)) + 1
    
    try:
        result = safe_web_search(q, CONFIG.web_provider, retries=2)
    except Exception as e:
        logger.error(f"[web_search_node] Ошибка веб поиска \n{e}.")
        raise


    
    logger.info(f"[web_search_node] Ending searching.")
    logger.debug(f"[web_search_node] Finded info: {result}"[:100])

    return {
        "web_search_result": result,
        "web_tries": tries,
    }


def augment_context_node(state: AgentState, model_embeddings) -> dict:
    """
    Добавляем веб-результат в накопленное поле org_web_evidence.
    Внимание! Тут старый результат поиска если он есть удаляется.
    Причина: старый результат нам не нужен, так как второй раз мы ищем если первый запрос модели был неккоректен.
    """
    logger = get_logger() # get logger
    
    logger.info(f"[augment_context_node] Starting finding chunks...")

    web_results = state.get("web_search_result") or []
    web_search_query = state.get("web_search_query", "")
    query = state.get("search_phrase", "")

    # fallback если нет запроса
    if query == "":
        if web_search_query != "":
            logger.warning(f"[augment_context_node] Query is clean. Using web search query instead {web_search_query}")
        else:
            logger.warning(f"[augment_context_node] Query and web search query is clean!! ")
            return {}

    
    logger.info(f"[augment_context_node] Phrase: {query}")
    logger.info(f"[augment_context_node] Starting finding chunks ...")
    # Поиск нужных чанков в документе
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
    
    
    logger.debug(f"[augment_context_node] Chunks finded: {top_chunks}")

    return {"org_web_evidence": top_chunks}


def classify_node(state: AgentState, llm) -> dict:
    """
    Классификационная нода
    """
    logger = get_logger() # get logger
    logger.info(f"[classify_node] Starting classification...")

    # Get state info
    org = dict(state["organization"])
    org["web_query"] = state.get("web_search_query", "")
    org["web_evidence"] = state.get("org_web_evidence", "")

    current_case = {
        "id": state["id"],
        "query": state["query"],
        "organization": org
    }
    
    # Construct a prompt
    msg = CLASSIFIER_PROMPT + "\n\nEXAMPLES:\n"+ CLASSIFIER_FEW_SHOT + "\n\nINPUT:\n" + json.dumps(current_case, ensure_ascii=False)

    # Запрос модели
    try:
        raw = safe_llm_invoke(llm, msg, retries=2)
    except Exception as e:
        logger.error(f"[classify_node] LLM Invoke error \n{e}")
        raise

    # Запрос чтения json
    try:
        obj = parse_json_or_repair(raw, llm=llm, retries=1)
    except Exception as e:
        logger.error(f"[classify_node] Json Parse error \n{e}")
        raise

    # Get relevance and reason
    rel = _clamp_relevance(obj.get("relevance"))
    reason = str(obj.get("reason", "")).strip()
    rel_true = state['relevance_true']


    logger.info(f"[classify_node] Model verdict: relevance {rel} (true - {rel_true}), reason {reason}")

    return {
        "relevance_model": rel,
        "reason": reason,
    }