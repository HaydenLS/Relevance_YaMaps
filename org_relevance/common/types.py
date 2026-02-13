from typing import TypedDict, Literal, Optional, Dict, Any, List, Sequence, Tuple


Relevance = Literal[0.0, 0.1, 1.0]

class Organization(TypedDict, total=False):
    name: str
    category: str
    address: str
    description: str
    reviews: str

class WebSearchResult(TypedDict, total=False):
    content: str
    url: str
    title: str

class RetrievedChunk(TypedDict):
    text: str
    score: float
    url: str
    title: str

class AgentState(TypedDict, total=False):
    # input
    id: str
    query: str
    organization: Organization

    # control
    web_tries: int
    max_web_tries: int

    # router output
    can_decide_now: bool
    router_reason: str

    # web pipeline
    web_search_query: str
    web_search_result: str | WebSearchResult
    search_phrase: str
    org_web_evidence: str | RetrievedChunk


    # final output
    relevance: Relevance
    reason: str
