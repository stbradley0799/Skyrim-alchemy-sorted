import itertools
import heapq
import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
TOP_K = 10

INGREDIENTS_URL = (
    "https://raw.githubusercontent.com/zsiciarz/skyrim-alchemy-toolbox/master/data/ingredients.json"
)

EFFECTS_URL = (
    "https://raw.githubusercontent.com/bmatsuo/Skyrim-Alchemy/master/data/effects.csv"
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_ingredients():
    """
    Returns:
      { "Giant's Toe": ["Damage Stamina", "Fortify Health", ...], ... }
    """
    r = requests.get(INGREDIENTS_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    out = {}
    # Expected structure: {"ingredients":[{"name":..., "effects":[...4...]}, ...]}
    for item in data.get("ingredients", []):
        name = str(item.get("name", "")).strip()
        effects = item.get("effects", [])
        if name and isinstance(effects, list) and len(effects) == 4:
            out[name] = [str(e).strip() for e in effects]
    return out


def _pick_effect_and_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Tries to infer:
      - effect name column (text)
      - value column (numeric)
    Works even if columns are named unexpectedly.
    """
    cols = list(df.columns)

    # Prefer common name columns
    preferred_name_cols = ["effect", "name", "Effect", "Name", "effect_name", "EffectName"]
    name_col = None
    for c in preferred_name_cols:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        # fallback: first object/string-like column
        for c in cols:
            if df[c].dtype == object:
                name_col = c
                break
    if name_col is None:
        # last resort: first column
        name_col = cols[0]

    # Prefer common value columns
    preferred_value_cols = ["value", "Value", "gold", "Gold", "cost", "Cost", "base_value", "BaseValue", "basecost", "Base cost"]
    value_col = None
    for c in preferred_value_cols:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        # fallback: first numeric column
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                value_col = c
                break
    if value_col is None:
        # fallback: try converting columns to numeric and pick the best
        best = None
        best_nonnull = -1
        for c in cols:
            if c == name_col:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            nn = int(s.notna().sum())
            if nn > best_nonnull:
                best_nonnull = nn
                best = c
        value_col = best if best is not None else cols[-1]

    return name_col, value_col


@st.cache_data(show_spinner=False)
def load_effect_values():
    """
    Returns:
      { "Fortify Health": 350, ... }
    Base single-effect potion values for ranking.
    Loader is robust to varying CSV column names.
    """
    # Let pandas sniff separators if needed
    df = pd.read_csv(EFFECTS_URL, sep=None, engine="python")

    if df.empty or len(df.columns) < 2:
        raise RuntimeError("Effects CSV loaded but appears empty or malformed.")

    name_col, value_col = _pick_effect_and_value_columns(df)

    # Normalize
    names = df[name_col].astype(str).str.strip()
    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    eff_vals = {}
    for n, v in zip(names, vals):
        if n:
            eff_vals[n] = int(v) if float(v).is_integer() else float(v)

    return eff_vals


# ============================================================
# CORE LOGIC
# ============================================================
def shared_effects(ingredients, ing_db):
    sets = [set(ing_db[i]) for i in ingredients]
    out = set()
    for a, b in itertools.combinations(sets, 2):
        out |= (a & b)
    return out


def top_k_potions(selected, ing_db, eff_vals, include_2, include_3, query):
    heap = []
    tie = 0
    q = query.lower().strip()

    sizes = []
    if include_2:
        sizes.append(2)
    if include_3:
        sizes.append(3)

    for k in sizes:
        for combo in itertools.combinations(selected, k):
            effs = shared_effects(combo, ing_db)
            if not effs:
                continue

            # Value = sum of each shared effect's base value
            value = 0
            known = []
            for e in effs:
                v = eff_vals.get(e, 0)
                if v:
                    value += float(v)
                    known.append(e)

            if value <= 0 or not known:
                continue

            known_sorted = sorted(set(known))
            potion = "Potion of " + " + ".join(known_sorted)

            row = {
                "Value": int(value) if float(value).is_integer() else value,
                "Potion": potion,
                "Ingredients": ", ".join(combo),
                "Effects": ", ".join(known_sorted),
            }

            if q:
                blob = (row["Potion"] + " " + row["Ingredients"] + " " + row["Effects"]).lower()
                if q not in blob:
                    continue

            tie += 1
            item = (float(value), tie, row)

            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif float(value) > heap[0][0]:
                heapq.heapreplace(heap, item)

    if not heap:
        return pd.DataFrame(columns=["Value", "Potion", "Ingredients", "Effects"])

    best = sorted(heap, key=lambda x: x[0], reverse=True)
    return pd.DataFrame([x[2] for x in best])


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Skyrim Potion Value Sorter", layout="wide")
st.title("Skyrim Potion Value Sorter")

with st.spinner("Loading data..."):
    ing_db = load_ingredients()
    eff_vals = load_effect_values()

all_ingredients = sorted(ing_db.keys())

selected = st.multiselect(
    "Available ingredients (unselect what you don't have)",
    options=all_ingredients,
    default=all_ingredients,  # default to ALL selected like you want
)

with st.sidebar:
    st.header("Recipe types")
    include_2 = st.checkbox("2-ingredient recipes", value=True)
    include_3 = st.checkbox("3-ingredient recipes", value=True)

query = st.text_input("Search (ingredient or effect)", "")

with st.spinner("Calculating top 10..."):
    df = top_k_potions(
        selected=selected,
        ing_db=ing_db,
        eff_vals=eff_vals,
        include_2=include_2,
        include_3=include_3,
        query=query,
    )

st.write(f"Showing top {len(df)} craftable potions (top 10 max)")
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV (top 10)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="skyrim_potions_top10.csv",
    mime="text/csv",
            )
