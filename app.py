import itertools
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Data sources (stable, public)
UESP_EFFECTS_URL = "https://en.uesp.net/wiki/Skyrim:Alchemy_Effects"
ING_FILE_URL = "https://raw.githubusercontent.com/cbednarski/skyrim-alchemy/master/data/uesp-ingredients-list.txt"

# Skyrim-ish value formula (approx, widely used)
DURATION_SCALE = 0.0794328

@dataclass(frozen=True)
class EffectInfo:
    name: str
    base_cost: float
    base_mag: float
    base_dur: float

@st.cache_data(show_spinner=False, ttl=60*60*24)
def load_ingredients() -> Dict[str, Dict]:
    """
    Loads ingredient -> 4 effects from a maintained text list based on UESP data.
    Format per line (name, formid, 4 effects, weight, value) separated by tabs/spaces.
    """
    txt = requests.get(ING_FILE_URL, timeout=30).text
    ings: Dict[str, Dict] = {}

    # Lines look like:
    # Abecean Longfin <formid> Weakness to Frost Fortify Sneak Weakness to Poison Fortify Restoration 0.5 15
    # We'll parse conservatively: name is first token group until formid (hex-ish), then 4 effect names, then weight/value.
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue

        # Find formid token (usually 8 hex chars or like 00106e1b)
        form_idx = None
        for i, tok in enumerate(parts):
            if re.fullmatch(r"[0-9a-fA-F]{8}", tok):
                form_idx = i
                break
        if form_idx is None or form_idx == 0:
            continue

        name = " ".join(parts[:form_idx])
        rest = parts[form_idx+1:]

        # At end: weight and value (two numbers). We'll peel those off.
        if len(rest) < 6:
            continue
        # last two should be numbers; if not, skip
        try:
            float(rest[-2]); float(rest[-1])
        except ValueError:
            continue

        effect_tokens = rest[:-2]
        # We need to split effect_tokens into 4 effect names.
        # UESP effect names are from a finite list; easiest: we’ll match using the effects table after we load it.
        ings[name] = {"name": name, "effect_tokens": effect_tokens}

    return ings

@st.cache_data(show_spinner=False, ttl=60*60*24)
def load_effects() -> Dict[str, EffectInfo]:
    """
    Parses UESP Alchemy Effects table to get Base cost / Base magnitude / Base duration.
    """
    html = requests.get(UESP_EFFECTS_URL, timeout=30).text
    soup = BeautifulSoup(html, "lxml")

    # UESP has tables; we look for rows containing effect name and numeric base fields.
    # This parser is intentionally tolerant.
    effects: Dict[str, EffectInfo] = {}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for tr in rows:
            tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
            if len(tds) < 4:
                continue

            # Heuristic: a row with 1st cell = name, and somewhere three numeric cells for base cost/mag/dur
            name = tds[0]
            nums = []
            for cell in tds[1:]:
                # pull first float-looking token
                m = re.search(r"(-?\d+(\.\d+)?)", cell)
                if m:
                    nums.append(float(m.group(1)))
            if len(nums) >= 3:
                base_cost, base_mag, base_dur = nums[0], nums[1], nums[2]
                # Filter out obvious non-effect headers
                if name and not name.lower().startswith("name"):
                    effects[name] = EffectInfo(name=name, base_cost=base_cost, base_mag=base_mag, base_dur=base_dur)

    if not effects:
        raise RuntimeError("Could not parse effects from UESP (layout may have changed).")

    return effects

def net_multiplier(skill: int, alchemist_bonus: float, specialty_bonus: float, fortify_bonus: float) -> float:
    # Common approximation of Skyrim's alchemy scaling
    skill_term = (math.floor(skill / 5) / 10.0) + 4.0
    return skill_term * (1.0 + alchemist_bonus) * (1.0 + specialty_bonus) * (1.0 + fortify_bonus)

def effect_gold_value(e: EffectInfo, mult: float) -> int:
    mag = e.base_mag * mult
    dur = e.base_dur * mult

    val = e.base_cost
    if e.base_mag > 0:
        val *= (mag ** 1.1)
    if e.base_dur > 0:
        val *= DURATION_SCALE * (dur ** 1.1)
    return int(math.floor(val))

def shared_effects(effects_by_ing: List[Set[str]]) -> Set[str]:
    shared: Set[str] = set()
    for a, b in itertools.combinations(effects_by_ing, 2):
        shared |= (a & b)
    return shared

def split_into_4_effects(effect_tokens: List[str], effect_names: Set[str]) -> List[str]:
    """
    Greedy longest-match segmentation using known effect names.
    """
    tokens = effect_tokens
    out = []
    i = 0
    while i < len(tokens) and len(out) < 4:
        best = None
        best_j = None
        # try longest spans first
        for j in range(min(len(tokens), i+6), i, -1):  # effect names usually not longer than ~6 words
            cand = " ".join(tokens[i:j])
            if cand in effect_names:
                best = cand
                best_j = j
                break
        if best is None:
            # fail-safe: stop
            return []
        out.append(best)
        i = best_j
    return out if len(out) == 4 else []

def build_df(available: List[str], allow2: bool, allow3: bool, skill: int, alch: float, spec: float, fort: float) -> pd.DataFrame:
    ings_raw = load_ingredients()
    effects_db = load_effects()
    effect_names = set(effects_db.keys())

    # finalize ingredient effects using the known effect list
    ings: Dict[str, Dict] = {}
    for name in available:
        item = ings_raw.get(name)
        if not item:
            continue
        effs = split_into_4_effects(item["effect_tokens"], effect_names)
        if effs:
            ings[name] = {"name": name, "effects": effs}

    mult = net_multiplier(skill, alch, spec, fort)
    sizes = ([2] if allow2 else []) + ([3] if allow3 else [])

    rows = []
    ing_list = list(ings.values())
    for k in sizes:
        for combo in itertools.combinations(ing_list, k):
            eff_sets = [set(i["effects"]) for i in combo]
            shared = shared_effects(eff_sets)
            if not shared:
                continue
            value = sum(effect_gold_value(effects_db[e], mult) for e in shared if e in effects_db)
            if value <= 0:
                continue
            rows.append({
                "Value": value,
                "Ingredients": ", ".join(i["name"] for i in combo),
                "Effects": ", ".join(sorted(shared))
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("Value", ascending=False).reset_index(drop=True)
    return df

# ---------------- UI ----------------
st.set_page_config(page_title="Skyrim Potion Value Sorter", layout="wide")
st.title("Skyrim Potion Value Sorter (filter by ingredients)")

with st.sidebar:
    st.header("Filters")
    allow2 = st.checkbox("2-ingredient mixes", value=True)
    allow3 = st.checkbox("3-ingredient mixes", value=True)

    st.header("Value assumptions (optional)")
    skill = st.slider("Alchemy skill", 15, 100, 15, 1)
    alch = st.slider("Alchemist perk bonus", 0.0, 1.0, 0.0, 0.05)
    spec = st.slider("Benefactor/Poisoner/Physician bonus", 0.0, 0.25, 0.0, 0.05)
    fort = st.slider("Fortify Alchemy gear total", 0.0, 3.0, 0.0, 0.05)

st.subheader("Select what ingredients you have")
with st.spinner("Loading ingredient list…"):
    ings_raw = load_ingredients()
all_ings = sorted(ings_raw.keys())

available = st.multiselect(
    "Available ingredients (unselect what you don't have)",
    options=all_ings,
    default=all_ings
)

query = st.text_input("Search results (ingredient or effect)", "")

with st.spinner("Generating ranked potions…"):
    df = build_df(available, allow2, allow3, skill, alch, spec, fort)

if query.strip() and not df.empty:
    q = query.strip().lower()
    df = df[df["Ingredients"].str.lower().str.contains(q) | df["Effects"].str.lower().str.contains(q)]

st.write(f"Results: {len(df):,}")
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="skyrim_potions_sorted.csv",
    mime="text/csv",
              )
