import json
from pathlib import Path
import streamlit as st
import pandas as pd
from openai import OpenAI
import os

# ---------- File paths (repo root on Streamlit Cloud) ----------
MATRIX_PATH = Path("triz_matrix_manufacturing.json")
INDUSTRY_PATH = Path("By Industry.txt")

# ---------- Secrets / Client ----------
@st.cache_resource
def get_openai_client():
    """Robust client init that works on Streamlit Cloud.
    - Reads from Secrets: [openai] api_key
    - Fallback to env OPENAI_API_KEY
    - Never raises: returns None and surfaces a readable error in the UI
    """
    api_key = None
    # Secrets (preferred)
    try:
        if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
            api_key = str(st.secrets["openai"]["api_key"]).strip()
    except Exception:
        api_key = None
    # Env fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        st.warning("OpenAI API key not found. Add it in Streamlit Secrets under [openai] api_key or set OPENAI_API_KEY.")
        return None

    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {type(e).__name__}: {e}")
        return None

# ---------- Loaders ----------
@st.cache_data
def load_triz():
    data = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    params_by_id = {int(p["id"]): p["name"] for p in data["parameters"]}
    params_by_name = {v: k for k, v in params_by_id.items()}
    principles_by_id = {int(p["id"]): p["name"] for p in data["principles"]}
    matrix = {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in data["matrix"].items()}
    return params_by_id, params_by_name, principles_by_id, matrix

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

# ---------- UI ----------
st.set_page_config(page_title="TRIZ 40 GPT Problem Statement Generator", page_icon="🧠", layout="centered")

# Theme toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

# CSS builder
def _css(dark: bool) -> str:
    if not dark:
        return """
        <style>
            .main {background-color: #f8f9fa;}
            div.block-container{padding-top:1.5rem; padding-bottom:1.5rem;}
            h1, h2, h3, h4 {color: #003366;}
            .stButton>button {
                background-color: #003366; color: white; border-radius: 10px;
                height: 3em; width: 100%; font-weight: bold; border: none;
            }
            .stButton>button:hover { background-color: #0055a5; color: #ffffff; transform: scale(1.02); }
            .stSelectbox label {font-weight: 600; color: #333333;}
            .stDataFrame {border: 1px solid #ccc; border-radius: 10px;}
            pre code {background-color: #e9ecef; color: #212529; font-size: 0.9rem; border-radius: 8px;}
        </style>
        """
    else:
        return """
        <style>
            .main {background-color: #0f172a;}
            div.block-container{padding-top:1.5rem; padding-bottom:1.5rem; color:#e5e7eb;}
            h1, h2, h3, h4 {color: #c7d2fe;}
            .stButton>button {
                background-color: #1d4ed8; color: white; border-radius: 10px;
                height: 3em; width: 100%; font-weight: bold; border: none;
            }
            .stButton>button:hover { background-color: #2563eb; color: #ffffff; transform: scale(1.02); }
            .stSelectbox label {font-weight: 600; color: #e5e7eb;}
            .stDataFrame {border: 1px solid #334155; border-radius: 10px;}
            pre code {background-color: #111827; color: #e5e7eb; font-size: 0.9rem; border-radius: 8px;}
        </style>
        """

st.markdown(_css(st.session_state.dark_mode), unsafe_allow_html=True)

st.title("🧠 TRIZ 40 GPT Problem Statement Generator")
client = get_openai_client()
with st.sidebar:
    st.markdown("### API Status")
    if client is None:
        st.error("❌ OpenAI: Not connected")
        st.caption("Set [openai] api_key in Secrets or OPENAI_API_KEY env var.")
    else:
        st.success("✅ OpenAI: Connected")("🧠 TRIZ 40 GPT Problem Statement Generator")

st.write(
    "Please **SELECT** the Contradiction Parameters and the Industry to **GENERATE** the ChatGPT Problem Statement prompt."
)

params_by_id, params_by_name, principles_by_id, matrix = load_triz()
industries = load_industries()

def param_label(pid: int) -> str:
    return f"{params_by_id[pid]} (Parameter #{pid})"

param_ids_sorted = sorted(params_by_id.keys(), key=lambda x: params_by_id[x].lower())
improving_label_to_id = {param_label(pid): pid for pid in param_ids_sorted}
worsening_label_to_id = {param_label(pid): pid for pid in param_ids_sorted}

c1, c2 = st.columns(2)
with c1:
    improving_choice = st.selectbox(
        "Improving Parameter",
        list(improving_label_to_id.keys()),
        index=param_ids_sorted.index(9) if 9 in param_ids_sorted else 0,
    )
with c2:
    worsening_choice = st.selectbox(
        "Worsening Parameter",
        list(worsening_label_to_id.keys()),
        index=param_ids_sorted.index(30) if 30 in param_ids_sorted else 0,
    )

industry = st.selectbox("By Industry", industries, index=(industries.index("Engineering") if "Engineering" in industries else 0))

generate = st.button("Generate TRIZ GPT Problem Statement", type="primary")

st.markdown("---")

def lookup_principles(improving_id: int, worsening_id: int):
    row = matrix.get(improving_id, {})
    return row.get(worsening_id, [])

def ids_to_principle_rows(principle_ids):
    rows = []
    for pid in principle_ids:
        name = principles_by_id.get(int(pid), f"Unknown principle #{pid}")
        rows.append({"Principle ID": int(pid), "Principle Name": name})
    return rows

if generate:
    improving_id = improving_label_to_id[improving_choice]
    worsening_id = worsening_label_to_id[worsening_choice]

    st.subheader("📘 Contradiction Parameters Problem Statement:")
    st.write(
        f"- **Improving**: `{params_by_id[improving_id]}` (#{improving_id})  \n"
        f"- **Worsening**: `{params_by_id[worsening_id]}` (#{worsening_id})  \n"
        f"- **Industry**: `{industry}`"
    )

    principle_ids = lookup_principles(improving_id, worsening_id)
    if not principle_ids:
        st.info("**No principles defined** for this parameter combination in the matrix.")
    else:
        df = pd.DataFrame(ids_to_principle_rows(principle_ids)).drop_duplicates(subset=["Principle ID"])
        st.markdown("**Recommended TRIZ Principles:**")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### 🧾 ChatGPT Prompt (copy-paste)")
if not principle_ids:
    principle_part = "No principles defined"
else:
    principle_part = "; ".join([f"#{int(pid)} — {principles_by_id[int(pid)]}" for pid in principle_ids])

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

# --- One-click generation via ChatGPT ---
if st.button("🤖 Generate with ChatGPT"):
    if client is None:
        st.error("OpenAI client not initialized. Add your API key in Secrets.")
    else:
        with st.spinner("Thinking with TRIZ..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a TRIZ expert assistant helping engineers resolve contradictions."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=800,
                )
                content = resp.choices[0].message.content
                st.markdown("### 🧩 Generated Problem Statement")
                st.write(content)
            except Exception as e:
                st.error(f"Error generating response: {e}")

st.caption("Data: TRIZ 39 parameters, 40 principles, and 39×39 matrix. IDs are authoritative; empty cells mean 'No principles defined.'")
