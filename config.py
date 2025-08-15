import os, streamlit as st

def get_openai_key() -> str:
    # Accept any of these secret names; standardize on OPENAI_API_KEY in the UI
    key = (
        st.secrets.get("OPENAI")
    )
    if not key:
        raise RuntimeError("Missing OPENAI in st.secrets.")
    key = key.strip().strip("'").strip('"')   # remove accidental quotes/whitespace
    if not key.startswith("sk-"):
        raise RuntimeError("Malformed OpenAI key (doesn't start with 'sk-').")
    os.environ["OPENAI_API_KEY"] = key        # propagate for libs that read env
    return key

def mask(k: str) -> str:
    return f"{k[:7]}...{k[-4:]}" if k and len(k) > 12 else "(unset)"
