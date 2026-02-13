from typing import List

import ollama # ollama должна импортиться до переменной окружения!
from langchain_community.tools import Tool, DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from org_relevance.common.types import WebSearchResult


def duckduckgo_web_search(query:str, max_results=5)-> List[WebSearchResult]:

    wrapper = DuckDuckGoSearchAPIWrapper(region="ru-ru", max_results=max_results)

    search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

    response = search.invoke(query)

    web_search_result: List[WebSearchResult] = []

    for dict_res in response:
        web_search_result.append({
            "content": dict_res['snippet'],
            "url": dict_res['link'],
            "title": dict_res['title']
        })

    return web_search_result


def ollama_web_search(query: str, max_results=5) -> List[WebSearchResult]:
    """
    Выполнение поиска с использованием ollama api
    Здесь надо учесть, что лимит может быть исчерпан!

    """
    response = ollama.web_search(query, max_results)['results']


    web_search_result: List[WebSearchResult] = []

    for dict_res in response:
        web_search_result.append({
            "content": dict_res['content'],
            "url": dict_res['url'],
            "title": dict_res['title']
        })

    return web_search_result