import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
from openai import OpenAI

from agno.agent import Agent
from agno.models.google import Gemini

from rag.pipeline import TelecomRAGPipeline
from anomaly.tools import fetch_anomaly_window

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# 1. Initialize RAG Pipeline
rag_pipeline = TelecomRAGPipeline()

def telecom_rag_retriever(
    query: str,
    num_documents: Optional[int] = 3,
    **kwargs
) -> Optional[List[Dict[str, str]]]:
    """Custom knowledge retriever that enforces RAG search for every query."""
    reranked_chunks = rag_pipeline.retrieve_context(query=query, top_k=num_documents or 3)

    docs = []
    for i, doc in enumerate(reranked_chunks):
        meta = doc.get("metadata", {})
        spec = meta.get("spec", "3GPP Spec")
        clause = meta.get("section_id", "Section")
        title = meta.get("title", "")
        content = doc.get("content", "").strip()
        docs.append({
            "title": f"[{spec} | Clause {clause}: {title}]",
            "content": content
        })
    return docs

# 2. Configure the Expert Model (Direct OpenAI client — no Agno overhead)
expert_client = OpenAI(
    api_key="not-needed",
    base_url=os.getenv("LOCAL_MODEL_URL", "https://transmit-albums-tries-serving.trycloudflare.com/v1"),
)
EXPERT_MODEL_ID = "/tmp/mistral-7b-instruct-v0.3.Q4_K_M.gguf"

# 3. Single-string expert tool — Gemini assembles the full prompt, expert just runs it
def call_expert(full_prompt: str) -> str:
    """
    Send the assembled prompt to the local Telecom Expert model for analysis.
    You MUST call this as the FINAL step after you have gathered ALL data using other tools.
    
    Args:
        full_prompt: The COMPLETE assembled prompt containing the user question,
                     the fetched anomaly report (if any), and the retrieved 3GPP spec chunks.
                     You must include ALL the actual data you fetched — do NOT use placeholders.
    """
    system_msg = (
        "You are an expert Telecom Technical Specialist.\n"
        "Your task is to generate a brief, highly-structured technical answer based STRICTLY on the provided 3GPP context.\n\n"
        "STRICT GUIDELINES:\n"
        "1. BREVITY & ESSENTIALS ONLY: Provide a direct answer containing ONLY the core technical facts, mechanisms, and key parameters. "
        "Avoid conversational filler, generic intros, or unnecessary background.\n"
        "2. STRUCTURE: Format the response using clean, compact bullet points with bold key terms.\n"
        "3. PRECISE CITATIONS: Include inline 3GPP citations for every technical point in the format [Spec: <spec_name> | Clause <section_id>].\n"
        "4. NO SPECULATION: Rely strictly on the provided context chunks."
    )
    completion = expert_client.chat.completions.create(
        model=EXPERT_MODEL_ID,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": full_prompt},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return completion.choices[0].message.content

# 4. Create the Router Agent (Gemini)
# Gemini has: fetch_anomaly_window, search_knowledge_base (RAG), call_expert
agent = Agent(
    name="Telecom Router",
    model=Gemini(id="gemini-3.5-flash-lite"),
    tools=[fetch_anomaly_window, call_expert],
    knowledge_retriever=telecom_rag_retriever,
    search_knowledge=True,
    instructions=[
        "You are a Telecom Routing Agent. Your ONLY responsibility is to gather raw data from tools and forward it to the Telecom Expert model via `call_expert`. You do NOT generate answers yourself.",
        "",
        "--- ROUTING PROTOCOL ---",
        "",
        "CASE 1 — User asks about a specific anomaly window (e.g. 'analyse WIN_033', 'what happened in WIN_1220'):",
        "  Step 1. Call `fetch_anomaly_window(window_id=<ID>)` to retrieve the full anomaly report text.",
        "  Step 2. Extract the DIAGNOSTIC QUESTION field from the report text.",
        "  Step 3. Call `search_knowledge_base(query=<diagnostic question>)` to retrieve relevant 3GPP specification chunks.",
        "  Step 4. Assemble a single string and call `call_expert(full_prompt=<string>)` using this exact format:",
        "      === ANOMALY REPORT ===",
        "      <paste the full anomaly report text verbatim>",
        "      === 3GPP SPECS ===",
        "      <paste every retrieved spec chunk verbatim, with its title>",
        "      === TASK ===",
        "      <paste the diagnostic question verbatim>",
        "",
        "CASE 2 — User asks a general 5G/telecom question (e.g. 'What is T310?', 'explain MCS'):",
        "  Step 1. Call `search_knowledge_base(query=<user question>)` to retrieve relevant 3GPP specification chunks.",
        "  Step 2. Assemble a single string and call `call_expert(full_prompt=<string>)` using this exact format:",
        "      === 3GPP SPECS ===",
        "      <paste every retrieved spec chunk verbatim, with its title>",
        "      === TASK ===",
        "      <paste the user's question verbatim>",
        "",
        "--- STRICT RULES ---",
        "- NEVER summarize, paraphrase, or truncate the data you paste into `call_expert`. Always use the exact raw text returned by tools.",
        "- NEVER call `call_expert` with placeholder values like 'anomaly report text' or '3GPP specs'.",
        "- Call `call_expert` EXACTLY ONCE per user request.",
        "- Return the EXACT output of `call_expert` to the user, word for word.",
    ],
    markdown=True
)

# 5. CLI
if __name__ == "__main__":
    print("Starting Telecom Hybrid Agent Chat Interface... Type 'exit' or 'quit' to close.\n")
    agent.cli_app(stream=True)
