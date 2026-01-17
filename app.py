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
    for item in data["ingredients"]:
        name = item["name"]
        effects = item["effects"]
        if len(effects) == 4:
            out[name] = effects
    return out


@st.cache_data(show_spinner=False)
def load_effect_values():
    """
    Returns:
      { "Fortify Health": 350, ... }
    Base single-effect potion values.
    """
    df = pd.read_csv(EFFECTS_URL)
    df["effect"] = df["effect"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    return dict(zip(df["effect"], df["value"]))


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

            value = sum(eff_vals.get(e, 0) for e in effs)
            if value <= 0:
                continue

            potion = "Potion of " + " + ".join(sorted(effs))
            row = {
                "Value": value,
                "Potion": potion,
                "Ingredients": ", ".join(combo),
                "Effects": ", ".join(sorted(effs)),
            }

            if q:
                blob = (row["Potion"] + row["Ingredients"] + row["Effects"]).lower()
                if q not in blob:
                    continue

            tie += 1
            item = (value, tie, row)

            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif value > heap[0][0]:
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
    default=all_ingredients,
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

st.write(f"Showing top {len(df)} craftable potions")
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV (top 10)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="skyrim_potions_top10.csv",
    mime="text/csv",
    )
