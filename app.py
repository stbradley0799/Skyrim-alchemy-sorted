import itertools
import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ---- Data sources ----
UESP_INGREDIENTS_URL = "https://en.uesp.net/wiki/Skyrim:Ingredients"
UESP_EFFECTS_URL = "https://en.uesp.net/wiki/Skyrim:Alchemy_Effects"

# ---- Value model (practical approximation for ranking) ----
DURATION_SCALE = 0.0794328  # used in many Skyrim alchemy value approximations


@dataclass(frozen=True)
class EffectInfo:
    name: str
    base_cost: float
    base_mag: float
    base_dur: float


def _get_html(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def _find_ingredients_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Find the UESP ingredients table by header names.
    """
    required = {"ingredient", "effect 1", "effect 2", "effect 3", "effect 4"}
    for table in soup.find_all("table"):
        tr = table.find("tr")
        if not tr:
            continue
        headers = [h.get_text(" ", strip=True).lower() for h in tr.find_all(["th", "td"])]
        if required.issubset(set(headers)):
            return table
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_ingredients() -> Dict[str, List[str]]:
    """
    Returns:
      { "Giant's Toe": ["Damage Stamina", "Fortify Health", "Fortify Carry Weight", "Damage Stamina Regen"], ... }
    """
    html = _get_html(UESP_INGREDIENTS_URL)
    soup = BeautifulSoup(html, "html.parser")

    table = _find_ingredients_table(soup)
    if table is None:
        raise RuntimeError("Could not find the Ingredients table on UESP. Layout may have changed.")

    ing: Dict[str, List[str]] = {}
    rows = table.find_all("tr")[1:]  # skip header
    for tr in rows:
        cols = tr.find_all(["td", "th"])
        if len(cols) < 5:
            continue

        name = cols[0].get_text(" ", strip=True)
        e1 = cols[1].get_text(" ", strip=True)
        e2 = cols[2].get_text(" ", strip=True)
        e3 = cols[3].get_text(" ", strip=True)
        e4 = cols[4].get_text(" ", strip=True)

        if not name or name.lower() == "ingredient":
            continue

        effects = [e1, e2, e3, e4]
        effects = [e for e in effects if e]
        if len(effects) == 4:
            ing[name] = effects

    if not ing:
        raise RuntimeError("Parsed 0 ingredients. UESP layout may have changed.")
    return ing


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def load_effects() -> Dict[str, EffectInfo]:
    """
    Parse effect base stats from UESP Skyrim:Alchemy_Effects.
    Returns:
      { "Fortify Health": EffectInfo(...), ... }
    """
    html = _get_html(UESP_EFFECTS_URL)
    soup = BeautifulSoup(html, "html.parser")

    # Find a table that has the columns we need.
    target = None
    for table in soup.find_all("table"):
        tr = table.find("tr")
        if not tr:
            continue
        headers = [h.get_text(" ", strip=True).lower() for h in tr.find_all(["th", "td"])]
        has_name = ("name" in headers) or ("effect" in headers)
        has_cost = ("base cost" in headers) or ("cost" in headers)
        has_mag = ("base magnitude" in headers) or ("magnitude" in headers)
        has_dur = ("base duration" in headers) or ("duration" in headers)
        if has_name and has_cost and has_mag and has_dur:
            target = table
            break

    if target is None:
        raise RuntimeError("Could not find the Alchemy Effects table on UESP. Layout may have changed.")

    header_cells = [h.get_text(" ", strip=True).lower() for h in target.find("tr").find_all(["th", "td"])]

    def idx_of(*names: str) -> Optional[int]:
        for n in names:
            if n in header_cells:
                return header_cells.index(n)
        return None

    name_i = idx_of("name", "effect")
    cost_i = idx_of("base cost", "cost")
    mag_i = idx_of("base magnitude", "magnitude")
    dur_i = idx_of("base duration", "duration")

    if None in (name_i, cost_i, mag_i, dur_i):
        raise RuntimeError("Effects table found, but could not locate required columns.")

    def to_float(txt: str) -> float:
        txt = txt.replace(",", "").strip()
        out = ""
        for ch in txt:
            if ch.isdigit() or ch in ".-":
                out += ch
            else:
                if out:
                    break
        return float(out) if out else 0.0

    effects: Dict[str, EffectInfo] = {}
    for tr in target.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) <= max(name_i, cost_i, mag_i, dur_i):
            continue

        name = cells[name_i].get_text(" ", strip=True)
        if not name:
            continue

        base_cost = to_float(cells[cost_i].get_text(" ", strip=True))
        base_mag = to_float(cells[mag_i].get_text(" ", strip=True))
        base_dur = to_float(cells[dur_i].get_text(" ", strip=True))

        effects[name] = EffectInfo(name=name, base_cost=base_cost, base_mag=base_mag, base_dur=base_dur)

    if not effects:
        raise RuntimeError("Parsed 0 effects. UESP layout may have changed.")
    return effects


def shared_effects(ingredients: Tuple[str, ...], ing_db: Dict[str, List[str]]) -> Set[str]:
    """
    Skyrim rule: potion effects are effects shared by at least 2 ingredients in the mix.
    """
    sets = [set(ing_db[i]) for i in ingredients]
    out: Set[str] = set()
    for a, b in itertools.combinations(sets, 2):
        out |= (a & b)
    return out


def effect_gold_value(e: EffectInfo, mult: float) -> int:
    """
    Practical value approximation for sorting/ranking.
    mult=1.0 ~ "base-ish"; increasing mult simulates stronger alchemy.
    """
    mag = e.base_mag * mult
    dur = e.base_dur * mult

    val = e.base_cost
    if e.base_mag > 0:
        val *= (mag ** 1.1)
    if e.base_dur > 0:
        val *= (DURATION_SCALE * (dur ** 1.1))
    return int(math.floor(val))


def top_k_potions(
    selected: List[str],
    ing_db: Dict[str, List[str]],
    eff_db: Dict[str, EffectInfo],
    include_2: bool,
    include_3: bool,
    mult: float,
    query: str,
    k: int = 10,
) -> pd.DataFrame:
    """
    Efficiently computes ONLY the top K rows using a heap.
    If query is provided, it returns the top K among matching rows.
    """
    sizes = []
    if include_2:
        sizes.append(2)
    if include_3:
        sizes.append(3)

    q = (query or "").strip().lower()
    heap: List[Tuple[int, int, dict]] = []  # (value, tie_id, row)
    tie = 0

    selected_sorted = sorted(selected)

    for size in sizes:
        for combo in itertools.combinations(selected_sorted, size):
            effs = shared_effects(combo, ing_db)
            if not effs:
                continue

            known = []
            value = 0
            for eff_name in effs:
                info = eff_db.get(eff_name)
                if info:
                    value += effect_gold_value(info, mult)
                    known.append(eff_name)

            if value <= 0 or not known:
                continue

            known_sorted = sorted(set(known))
            potion_name = "Potion of " + " + ".join(known_sorted)

            row = {
                "Value": value,
                "Potion": potion_name,
                "Ingredients": ", ".join(combo),
                "Effects": ", ".join(known_sorted),
            }

            # If searching, filter rows before considering them for top K
            if q:
                searchable = f"{row['Potion']} {row['Ingredients']} {row['Effects']}".lower()
                if q not in searchable:
                    continue

            tie += 1
            item = (value, tie, row)

            if len(heap) < k:
                heapq.heappush(heap, item)
            else:
                # Keep only the best K (heap[0] is smallest value)
                if value > heap[0][0]:
                    heapq.heapreplace(heap, item)

    if not heap:
        return pd.DataFrame(columns=["Value", "Potion", "Ingredients", "Effects"])

    # Sort descending by value
    best = sorted(heap, key=lambda x: x[0], reverse=True)
    df = pd.DataFrame([item[2] for item in best])
    df = df.sort_values("Value", ascending=False).reset_index(drop=True)
    return df


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Skyrim Potion Value Sorter", layout="wide")
st.title("Skyrim Potion Value Sorter (filter by ingredients)")

with st.spinner("Loading ingredient & effect data… (first load can take ~10–30s)"):
    ing_db = load_ingredients()
    eff_db = load_effects()

all_ings = sorted(ing_db.keys())

# ✅ default selection = ALL ingredients (your requested behavior)
selected = st.multiselect(
    "Available ingredients (unselect what you don't have)",
    options=all_ings,
    default=all_ings,
)

with st.sidebar:
    st.header("Mix sizes")
    include_2 = st.checkbox("2-ingredient recipes", value=True)
    include_3 = st.checkbox("3-ingredient recipes", value=True)

    st.header("Value model")
    st.caption("1.0 = base-ish ranking. Increase to approximate stronger alchemy.")
    mult = st.slider("Alchemy multiplier", 0.5, 10.0, 1.0, 0.1)

query = st.text_input("Search (ingredient or effect)", "")

with st.spinner("Finding top 10…"):
    df = top_k_potions(
        selected=selected,
        ing_db=ing_db,
        eff_db=eff_db,
        include_2=include_2,
        include_3=include_3,
        mult=mult,
        query=query,
        k=10,
    )

st.write(f"Top results: {len(df):,} (showing top 10)")
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV (top 10)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="skyrim_potions_top10.csv",
    mime="text/csv",
)
