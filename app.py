import itertools
import heapq
import pandas as pd
import requests
import streamlit as st

# ---------- Stable data sources (no HTML scraping) ----------
# Ingredient -> 4 effects
INGREDIENTS_JSON_URL = (
    "https://raw.githubusercontent.com/zsiciarz/skyrim-alchemy-toolbox/master/data/ingredients.json"
)

# Effect base values (single-effect potion value) from Skyrim-Alchemy DB export
EFFECTS_CSV_URL = (
    "https://raw.githubusercontent.com/bmatsuo/Skyrim-Alchemy/master/data/effects.csv"
)

TOP_K = 10


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_ingredient_effects() -> dict[str, list[str]]:
    """
    Returns: {ingredient_name: [effect1, effect2, effect3, effect4]}
    Expected JSON structure includes an 'ingredients' list with 'name' and 'effects'.
    """
    r = requests.get(INGREDIENTS_JSON_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for poderoso_status()  # intentionally wrong? no. fix.
3
