# app.py
import json
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- Paths ----------
MATRIX_PATH = Path("triz_matrix_manufacturing.json")
INDUSTRY_PATH = Path("By Industry.txt")

# ---------- Page & Theme ----------
st.set_page_config(
    page_title="TRIZ 40 GPT Problem Statement Generator",
    page_icon="🧠",
    layout="centered",
)

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)


def _css(dark: bool) -> str:
    if not dark:
        return """
        <style>
            .main {background-color: #f8f9fa;}
            h1, h2, h3, h4 {color: #003366;}
            .stButton>button {
                background-color: #003366; color: white; border-radius: 10px;
                height: 3em; width: 100%; font-weight: bold; border: none;
            }
            .stButton>button:hover {
                background-color: #0055a5; color: #ffffff; transform: scale(1.02);
            }
            .stButton>button:disabled {
                background-color: #adb5bd; color: #eee; cursor: not-allowed;
            }
        </style>
        """
    else:
        return """
        <style>
            .main {background-color: #0f172a;}
            h1, h2, h3, h4 {color: #c7d2fe;}
            .stButton>button {
                background-color: #1d4ed8; color: white; border-radius: 10px;
                height: 3em; width: 100%; font-weight: bold; border: none;
            }
            .stButton>button:hover {
                background-color: #2563eb; color: #ffffff; transform: scale(1.02);
            }
            .stButton>button:disabled {
                background-color: #475569; color: #9ca3af; cursor: not-allowed;
            }
        </style>
        """


st.markdown(_css(st.session_state.dark_mode), unsafe_allow_html=True)
st.title("🧠 TRIZ 40 GPT Problem Statement Generator")

# ---------- OpenAI Client ----------
@st.cache_resource
def get_openai_client():
    api_key = (
        st.secrets.get("openai", {}).get("api_key")
        if "openai" in st.secrets
        else os.getenv("OPENAI_API_KEY", "")
    )
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = get_openai_client()

with st.sidebar:
    st.markdown("### API Status")
    if client is None:
        st.error("❌ OpenAI: Not connected")
        st.caption("Add [openai] api_key in Secrets or OPENAI_API_KEY in env.")
    else:
        st.success("✅ OpenAI: Connected")

# ---------- Loaders ----------
@st.cache_data
def load_triz():
    data = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    params_by_id = {int(p["id"]): p["name"] for p in data["parameters"]}
    principles_by_id = {int(p["id"]): p["name"] for p in data["principles"]}
    matrix = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data["matrix"].items()}
    return params_by_id, principles_by_id, matrix


@st.cache_data
def load_industries():
    if not INDUSTRY_PATH.exists():
        return ["Engineering"]
    inds = [ln.strip() for ln in INDUSTRY_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seen, out = set(), []
    for i in inds:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


params_by_id, principles_by_id, matrix = load_triz()
industries = load_industries()

# ---------- UI ----------
def param_label(pid: int) -> str:
    return f"{params_by_id[pid]} (Parameter #{pid})"


param_ids_sorted = sorted(params_by_id.keys(), key=lambda x: params_by_id[x].lower())
label_to_id = {param_label(pid): pid for pid in param_ids_sorted}

c1, c2 = st.columns(2)
with c1:
    improving_choice = st.selectbox(
        "Improving Parameter",
        list(label_to_id.keys()),
        index=param_ids_sorted.index(9) if 9 in param_ids_sorted else 0,
    )
with c2:
    worsening_choice = st.selectbox(
        "Worsening Parameter",
        list(label_to_id.keys()),
        index=param_ids_sorted.index(30) if 30 in param_ids_sorted else 0,
    )

industry = st.selectbox(
    "By Industry",
    industries,
    index=(industries.index("Engineering") if "Engineering" in industries else 0),
)

st.markdown("---")

# ---------- Logic ----------
def lookup_principles(improving_id: int, worsening_id: int):
    row = matrix.get(improving_id, {})
    return row.get(worsening_id, [])


improving_id = label_to_id[improving_choice]
worsening_id = label_to_id[worsening_choice]
principle_ids = lookup_principles(improving_id, worsening_id)

st.subheader("📘 Contradiction Parameters Problem Statement:")
st.write(
    f"- **Improving**: `{params_by_id[improving_id]}` (#{improving_id})  \n"
    f"- **Worsening**: `{params_by_id[worsening_id]}` (#{worsening_id})  \n"
    f"- **Industry**: `{industry}`"
)

if not principle_ids:
    st.info("**No principles defined** for this parameter combination in the matrix.")
else:
    df = pd.DataFrame(
        [
            {
                "Principle ID": int(pid),
                "Principle Name": principles_by_id.get(int(pid), f"Unknown principle #{pid}"),
            }
            for pid in principle_ids
        ]
    ).drop_duplicates(subset=["Principle ID"])
    st.markdown("**Recommended TRIZ Principles:**")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Prompt ----------
st.markdown("### 🧾 ChatGPT Prompt (copy-paste)")
principle_part = (
    "No principles defined"
    if not principle_ids
    else "; ".join([f"#{int(pid)} — {principles_by_id[int(pid)]}" for pid in principle_ids])
)

prompt = f"""You are a TRIZ assistant. Use ONLY the TRIZ principles provided below (do not invent new ones).

Context:
- Industry: {industry}
- Improving parameter (ID #{improving_id}): {params_by_id[improving_id]}
- Worsening parameter (ID #{worsening_id}): {params_by_id[worsening_id]}
- Applicable TRIZ principles: {principle_part}

Tasks:
1) Write a precise **problem statement** that frames the technical contradiction (be specific; avoid solutions).
2) Explain why each listed principle could resolve this contradiction (short bullet points, 1–2 lines per principle).
3) Propose **3–5 idea starters** grounded ONLY in the listed principles (label each with its principle ID).
4) Note key assumptions and potential risks.

Rules:
- Always treat the numeric IDs as authoritative.
- If no principles are defined, say “No principles defined” and ask for adjacent-parameter exploration instead.
"""
st.code(prompt, language="markdown")

# ---------- ChatGPT ----------
def call_chatgpt(user_prompt: str) -> str:
    if client is None:
        return "⚠️ OpenAI client not initialized. Add your API key in Secrets or env."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a TRIZ expert assistant helping engineers resolve contradictions."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"


# ---------- Button Behavior ----------
btn_disabled = len(principle_ids) == 0
btn_label = "🚫 No TRIZ principles available" if btn_disabled else "🤖 Generate with ChatGPT"

if st.button(btn_label, disabled=btn_disabled):
    with st.spinner("Thinking with TRIZ..."):
        output = call_chatgpt(prompt)
    st.markdown("### 🧩 Generated Problem Statement")
    st.write(output)

# ---------- Footer ----------
st.caption("Data: TRIZ 39 parameters, 40 principles, and 39×39 matrix. IDs are authoritative; empty cells mean “No principles defined.”")



