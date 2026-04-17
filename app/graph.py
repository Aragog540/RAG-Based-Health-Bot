"""
app/graph.py
────────────
LangGraph agentic RAG graph for the medical chatbot.

Graph flow:
  retrieve → grade_documents → generate → check_hallucination → respond

Each node is a pure function that receives and returns a typed State dict.
"""

from __future__ import annotations

from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from app.config import settings
from app.retriever import get_retriever


LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "hi": "Hindi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "pa": "Punjabi",
    "ur": "Urdu",
    "as": "Assamese",
    "or": "Odia",
    "ne": "Nepali",
    "bho": "Bhojpuri",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}


# ════════════════════════════════════════════════════════════════════════════
# State definition
# ════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    question: str
    language: str
    translated_question: str
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    grounded: bool
    messages: Annotated[List, operator.add]   # accumulated chat history


# ════════════════════════════════════════════════════════════════════════════
# LLM (free, local via Ollama)
# ════════════════════════════════════════════════════════════════════════════

def _get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


def _is_english(language: str) -> bool:
    normalized = (language or "en").strip().lower()
    return normalized == "en" or normalized.startswith("en-") or normalized == "english"


def _language_label(language: str) -> str:
    normalized = (language or "en").strip().lower()
    if normalized in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[normalized]
    if normalized in {"bhojpuri", "bhojpuri language"}:
        return "Bhojpuri"
    if normalized in {"bengali", "bangla"}:
        return "Bengali"
    if normalized in {"gujarati"}:
        return "Gujarati"
    if normalized in {"malayalam"}:
        return "Malayalam"
    return language.strip() or "English"


# ════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical document relevance grader. "
        "Given a user question and a retrieved document chunk, "
        "decide if the chunk is relevant.\n"
        "Reply with ONLY 'yes' or 'no'. No explanation.",
    ),
    (
        "human",
        "Question: {question}\n\nDocument chunk:\n{document}",
    ),
])

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Translate the user's medical question into plain English for retrieval. "
        "Preserve the original meaning, keep medical terms accurate, and output only the translation. "
        "Do not answer the question.",
    ),
    (
        "human",
        "Language: {language}\nQuestion: {question}",
    ),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert medical assistant. "
        "Answer the question using ONLY the provided medical book excerpts. "
        "Be precise, clear, and cite page numbers when available. "
        "If the answer is not in the excerpts, say so honestly. "
        "Do NOT fabricate information. "
        "Respond in {language}.\n\n"
        "Relevant excerpts:\n{context}",
    ),
    (
        "human",
        "Original question: {question}\n"
        "English retrieval query: {translated_question}\n"
        "Answer language: {language}",
    ),
])

HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fact-checker. "
        "Given an answer and the source documents it was based on, "
        "decide if the answer is grounded in (supported by) the documents. "
        "Reply with ONLY 'yes' (grounded) or 'no' (not grounded).",
    ),
    (
        "human",
        "Answer: {generation}\n\nSource documents:\n{documents}",
    ),
])


# ════════════════════════════════════════════════════════════════════════════
# Node functions
# ════════════════════════════════════════════════════════════════════════════

def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve top-k chunks from ChromaDB based on the question."""
    retriever = get_retriever()
    query = state.get("translated_question") or state["question"]
    docs = retriever.invoke(query)
    return {**state, "documents": docs}


def grade_documents_node(state: GraphState) -> GraphState:
    """Filter retrieved chunks — keep only relevant ones."""
    llm = _get_llm(temperature=0.0)
    chain = GRADE_PROMPT | llm
    query = state.get("translated_question") or state["question"]

    relevant = []
    for doc in state["documents"]:
        result = chain.invoke({
            "question": query,
            "document": doc.page_content,
        })
        if "yes" in result.content.lower():
            relevant.append(doc)

    return {**state, "relevant_documents": relevant}


def generate_node(state: GraphState) -> GraphState:
    """Generate an answer using the relevant document chunks."""
    llm = _get_llm(temperature=0.1)
    chain = GENERATE_PROMPT | llm

    docs = state.get("relevant_documents") or state["documents"]

    # Build context string with page numbers if available
    context_parts = []
    for i, doc in enumerate(docs, 1):
        page_info = ""
        if doc.metadata.get("page"):
            page_info = f" [Page {doc.metadata['page']}]"
        context_parts.append(f"[Excerpt {i}{page_info}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    result = chain.invoke({
        "question": state["question"],
        "translated_question": state.get("translated_question") or state["question"],
        "language": _language_label(state.get("language", "en")),
        "context": context,
    })

    return {
        **state,
        "generation": result.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=result.content),
        ],
    }


def check_hallucination_node(state: GraphState) -> GraphState:
    """Check whether the generated answer is grounded in the documents."""
    llm = _get_llm(temperature=0.0)
    chain = HALLUCINATION_PROMPT | llm

    docs_text = "\n\n".join(
        d.page_content for d in (state.get("relevant_documents") or state["documents"])
    )
    result = chain.invoke({
        "generation": state["generation"],
        "documents": docs_text,
    })
    grounded = "yes" in result.content.lower()
    return {**state, "grounded": grounded}


def translate_question_node(state: GraphState) -> GraphState:
    """Translate the user's question into English for retrieval when needed."""
    if _is_english(state.get("language", "en")):
        return {**state, "translated_question": state["question"]}

    llm = _get_llm(temperature=0.0)
    chain = TRANSLATE_PROMPT | llm
    result = chain.invoke({
        "language": _language_label(state.get("language", "en")),
        "question": state["question"],
    })
    translated_question = result.content.strip().strip('"')
    if not translated_question:
        translated_question = state["question"]
    return {**state, "translated_question": translated_question}


# ════════════════════════════════════════════════════════════════════════════
# Conditional edge
# ════════════════════════════════════════════════════════════════════════════

def decide_after_grading(state: GraphState) -> str:
    """Route to generate if we have relevant docs, else generate anyway."""
    # Even with no perfectly relevant docs, we attempt generation and
    # rely on hallucination check to flag quality.
    return "generate"


# ════════════════════════════════════════════════════════════════════════════
# Build graph
# ════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("translate_question", translate_question_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("check_hallucination", check_hallucination_node)

    # Set entry point
    graph.set_entry_point("translate_question")

    # Edges
    graph.add_edge("translate_question", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate"},
    )
    graph.add_edge("generate", "check_hallucination")
    graph.add_edge("check_hallucination", END)

    return graph.compile()


# Compiled graph (singleton)
rag_graph = build_graph()


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def run_rag(question: str, language: str = "en") -> dict:
    """
    Run the full RAG graph for a question.

    Returns:
        {
          "answer": str,
          "sources": [{"content": str, "page": int|None, "source": str|None}],
          "grounded": bool,
        }
    """
    initial_state: GraphState = {
        "question": question,
        "language": language,
        "translated_question": question,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "grounded": False,
        "messages": [],
    }

    final_state = rag_graph.invoke(initial_state)

    sources = []
    for doc in (final_state.get("relevant_documents") or final_state.get("documents", [])):
        sources.append({
            "content": doc.page_content[:400],   # truncate for response
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
        })

    return {
        "answer": final_state["generation"],
        "sources": sources,
        "grounded": final_state.get("grounded", True),
    }
