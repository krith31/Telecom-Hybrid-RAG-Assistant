"""Evaluation expert adapter; preserves main.py prompt and generation settings."""
import os

def call_expert(full_prompt: str, client=None) -> str:
    """
    Send the assembled prompt to the local Telecom Expert model for analysis.
    You MUST call this as the FINAL step after you have gathered ALL data using other tools.
    
    Args:
        full_prompt: The COMPLETE assembled prompt containing the user question,
                     the fetched anomaly report (if any), and the retrieved 3GPP spec chunks.
                     You must include ALL the actual data you fetched — do NOT use placeholders.
    """
    if client is None:
        from openai import OpenAI
        url = os.getenv("LOCAL_MODEL_URL")
        if not url:
            raise RuntimeError("Set LOCAL_MODEL_URL to the running expert /v1 endpoint")
        client = OpenAI(api_key=os.getenv("LOCAL_MODEL_API_KEY", "not-needed"),
                        base_url=url, timeout=60.0, max_retries=2)
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
    completion = client.chat.completions.create(
        model=os.getenv("EXPERT_MODEL_ID", "/tmp/mistral-7b-instruct-v0.3.Q4_K_M.gguf"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": full_prompt},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Expert returned an empty answer")
    return answer
