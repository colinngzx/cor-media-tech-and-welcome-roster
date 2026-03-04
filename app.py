# app.py
import io
import re
import random
import requests
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Config
# =========================
WELCOME_GID = "2080125013"
MEDIA_GID = "0"

# Google Sheet "export CSV" URL pattern
GSHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

MEDIA_LEAD_PRIORITY = ["Gavin", "Ben", "Mich Lo"]  # prioritized among eligible leads if present


WELCOME_ROLES_NON_HC = ["Welcome Team Lead", "Member 1", "Member 2", "Member 3"]
WELCOME_ROLES_HC = ["Welcome Team Lead", "Member 1", "Member 2", "Member 3", "Member 4"]

MEDIA_ROLES = ["Sound Crew", "Projectionist", "Stream Director", "Cam 1", "Cam 2", "Media Team Lead"]
MEDIA_CORE_ROLES = ["Sound Crew", "Projectionist", "Stream Director", "Cam 1"]  # roles we auto-assign
MEDIA_MANUAL_PLACEHOLDER = {"Cam 2"}  # always blank


# =========================
# Utilities
# =========================
def norm_name(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def is_yes(v) -> bool:
    return str(v).strip().lower() == "yes"


def fetch_sheet_gid(sheet_id: str, gid: str) -> pd.DataFrame:
    url = GSHEET_CSV_URL.format(sheet_id=sheet_id, gid=gid)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def all_sundays(year: int, month: int) -> List[pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd(1)).normalize()
    dates = pd.date_range(start, end, freq="D")
    sundays = [d for d in dates if d.dayofweek == 6]  # Mon=0 ... Sun=6
    return sundays


def month_options() -> List[Tuple[int, int, str]]:
    # returns (year, month, label)
    now = pd.Timestamp.today().normalize()
    opts = []
    for i in range(0, 24):
        d = (now + pd.DateOffset(months=i)).replace(day=1)
        opts.append((int(d.year), int(d.month), d.strftime("%b %Y")))
    return opts


def reset_for_ministry(ministry: str):
    # Keep selected ministry only; nuke everything else to enforce isolation
    for k in list(st.session_state.keys()):
        if k not in {"ministry"}:
            del st.session_state[k]
    st.session_state["ministry"] = ministry


def seeded_shuffle(items: List[str], seed: int):
    rng = random.Random(seed)
    rng.shuffle(items)


# =========================
# Models
# =========================
@dataclass
class ServiceDate:
    date: pd.Timestamp
    hc: bool = False
    combined: bool = False
    notes: str = ""


# =========================
# Scoring / constraints helpers
# =========================
def pairing_penalty(candidate: str, already_assigned: List[str], pair_counts: Dict[Tuple[str, str], int]) -> int:
    # Penalize if candidate has served with already_assigned before.
    # Uses symmetric pair keys.
    pen = 0
    for other in already_assigned:
        a, b = sorted([candidate, other])
        pen += pair_counts.get((a, b), 0)
    return pen


def week_rest_blocked(person: str, date: pd.Timestamp, last_assigned_week: Dict[str, pd.Timestamp]) -> bool:
    if person not in last_assigned_week:
        return False
    prev = last_assigned_week[person]
    # "consecutive weeks" => previous service date is 7 days before current (since 1 service/week Sunday)
    return (date - prev).days == 7


def choose_best_candidate(
    candidates: List[str],
    date: pd.Timestamp,
    already_assigned: List[str],
    shift_counts: Dict[str, int],
    pair_counts: Dict[Tuple[str, str], int],
    last_assigned_week: Dict[str, pd.Timestamp],
    rng_seed: int,
    allow_consecutive_if_exhausted: bool = True,
) -> Optional[str]:
    if not candidates:
        return None

    # Filter out consecutive-week candidates if possible
    non_consec = [c for c in candidates if not week_rest_blocked(c, date, last_assigned_week)]
    pool = non_consec if non_consec else (candidates if allow_consecutive_if_exhausted else [])

    if not pool:
        return None

    # Weighted rotation primary: lowest shift counts.
    min_shifts = min(shift_counts.get(c, 0) for c in pool)
    pool = [c for c in pool if shift_counts.get(c, 0) == min_shifts]

    # Social mixing secondary: minimize pairing penalty with already assigned
    penalties = {c: pairing_penalty(c, already_assigned, pair_counts) for c in pool}
    min_pen = min(penalties.values()) if penalties else 0
    pool = [c for c in pool if penalties[c] == min_pen]

    # No alphabetical bias: shuffle ties deterministically-ish
    pool = pool[:]
    seeded_shuffle(pool, rng_seed)

    return pool[0] if pool else None


def update_pair_counts(crew: List[str], pair_counts: Dict[Tuple[str, str], int]):
    crew = [c for c in crew if c]
    for i in range(len(crew)):
        for j in range(i + 1, len(crew)):
            a, b = sorted([crew[i], crew[j]])
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1


# =========================
# Welcome engine
# =========================
def build_welcome_people(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns:
    # A Name, B Team Lead, D Gender, F Couple (numeric id), G senior citizen
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Name"] = df["Name"].map(norm_name)

    # Tolerant column access by exact names you specified
    lead_col = "Team Lead"
    gender_col = "Gender"
    couple_col = "Couple"  # may not be header; user said column F. We'll fallback if not.
    senior_col = "senior citizen"

    # If headers differ, try to find likely matches
    def find_col(target: str) -> Optional[str]:
        for c in df.columns:
            if str(c).strip().lower() == target.strip().lower():
                return c
        return None

    lead_col = find_col("Team Lead") or lead_col
    gender_col = find_col("Gender") or gender_col
    senior_col = find_col("senior citizen") or senior_col

    # Couple column: you said column F contains 1,2,3... Could be header like "Couple"
    # We'll try exact match, else attempt to pick 6th column if present.
    couple_col_found = find_col("Couple")
    if couple_col_found:
        couple_col = couple_col_found
    else:
        # F is index 5
        if len(df.columns) >= 6:
            couple_col = df.columns[5]

    df["is_lead"] = df.get(lead_col, "").apply(lambda x: str(x).strip() != "" and str(x).strip() != "0")
    df["gender"] = df.get(gender_col, "").astype(str).str.strip().str.lower()
    df["is_senior"] = df.get(senior_col, "").apply(lambda x: str(x).strip().lower() in {"yes", "true", "1", "y"})
    df["couple_id"] = pd.to_numeric(df.get(couple_col, np.nan), errors="coerce")

    df = df[df["Name"] != ""].drop_duplicates(subset=["Name"]).reset_index(drop=True)
    return df[["Name", "is_lead", "gender", "is_senior", "couple_id"]]


def welcome_couple_map(people: pd.DataFrame) -> Dict[str, str]:
    # Map each person in a couple to their partner name (only for pairs)
    cmap = {}
    by_id = people.dropna(subset=["couple_id"]).groupby("couple_id")["Name"].apply(list).to_dict()
    for cid, names in by_id.items():
        if len(names) == 2:
            a, b = names
            cmap[a] = b
            cmap[b] = a
        # If bad data (len != 2), ignore couple logic for that id
    return cmap


def generate_welcome_roster(
    services: List[ServiceDate],
    people: pd.DataFrame,
    availability: Dict[str, Set[pd.Timestamp]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Output grid: rows=roles, cols=dates (formatted dd-MMM)
    """
    couple_partner = welcome_couple_map(people)
    names = people["Name"].tolist()
    male_names = set(people.loc[people["gender"].str.startswith("m"), "Name"].tolist())
    seniors = set(people.loc[people["is_senior"] == True, "Name"].tolist())
    lead_pool = set(people.loc[people["is_lead"] == True, "Name"].tolist())

    shift_counts: Dict[str, int] = {n: 0 for n in names}
    last_assigned_week: Dict[str, pd.Timestamp] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}

    date_labels = [s.date.strftime("%d-%b") for s in services]
    # Build max rows (HC has 5 total; non-HC 4). We'll build 5 and blank when not needed.
    all_rows = ["Welcome Team Lead", "Member 1", "Member 2", "Member 3", "Member 4"]
    grid = pd.DataFrame(index=all_rows, columns=date_labels, data="")

    for idx, s in enumerate(services):
        d = s.date
        needed_roles = WELCOME_ROLES_HC if s.hc else WELCOME_ROLES_NON_HC
        assigned: Dict[str, str] = {r: "" for r in all_rows}

        def is_available(person: str) -> bool:
            return d in availability.get(person, set())

        # For couples rule: only schedule if both available; when choose one, auto add partner into next member slot.
        def couple_ok(person: str) -> bool:
            partner = couple_partner.get(person)
            if not partner:
                return True
            return is_available(person) and is_available(partner)

        # --- Pick Welcome Team Lead ---
        lead_candidates = [n for n in lead_pool if is_available(n) and couple_ok(n)]
        lead = choose_best_candidate(
            lead_candidates,
            d,
            already_assigned=[],
            shift_counts=shift_counts,
            pair_counts=pair_counts,
            last_assigned_week=last_assigned_week,
            rng_seed=1000 + idx,
            allow_consecutive_if_exhausted=True,
        )
        if lead:
            assigned["Welcome Team Lead"] = lead
            # note: lead DOES count as a shift in Welcome (you didn't exclude it), so count it
            shift_counts[lead] += 1
            last_assigned_week[lead] = d

        # Track crew list as we go for social mixing penalties
        crew_list: List[str] = [assigned["Welcome Team Lead"]] if assigned["Welcome Team Lead"] else []

        # We'll fill members sequentially; enforce Member 1 male-only; couple magnet into next available member slot.
        member_slots = [r for r in needed_roles if r.startswith("Member")]
        # Ensure senior requirement: at least one member is senior (fallback allowed if none available)
        senior_required = True

        def next_empty_slot() -> Optional[str]:
            for slot in member_slots:
                if assigned[slot] == "":
                    return slot
            return None

        # helper to place person into a specific slot
        def place(slot: str, person: str):
            assigned[slot] = person
            crew_list.append(person)
            shift_counts[person] += 1
            last_assigned_week[person] = d

        # First fill Member 1 (male-only)
        slot1 = "Member 1" if "Member 1" in member_slots else None
        if slot1:
            candidates = []
            for n in names:
                if n in crew_list:
                    continue
                if n not in male_names:
                    continue
                if not is_available(n):
                    continue
                if not couple_ok(n):
                    continue
                candidates.append(n)

            pick = choose_best_candidate(
                candidates,
                d,
                already_assigned=crew_list,
                shift_counts=shift_counts,
                pair_counts=pair_counts,
                last_assigned_week=last_assigned_week,
                rng_seed=2000 + idx,
                allow_consecutive_if_exhausted=True,
            )
            if pick:
                place(slot1, pick)
                # auto-add partner
                partner = couple_partner.get(pick)
                if partner:
                    slotn = next_empty_slot()
                    if slotn:
                        place(slotn, partner)

        # Fill remaining member slots
        while True:
            slot = next_empty_slot()
            if not slot:
                break

            # if senior required, try to satisfy it with remaining slots
            need_senior_now = senior_required and not any((assigned[s] in seniors) for s in member_slots if assigned[s])
            if need_senior_now:
                # candidates seniors first
                cand_pool = []
                for n in names:
                    if n in crew_list:
                        continue
                    if not is_available(n):
                        continue
                    if not couple_ok(n):
                        continue
                    if n not in seniors:
                        continue
                    cand_pool.append(n)

                pick = choose_best_candidate(
                    cand_pool,
                    d,
                    already_assigned=crew_list,
                    shift_counts=shift_counts,
                    pair_counts=pair_counts,
                    last_assigned_week=last_assigned_week,
                    rng_seed=3000 + idx + len(crew_list),
                    allow_consecutive_if_exhausted=True,
                )

                if pick is None:
                    # fallback allowed: fill non-senior
                    cand_pool = []
                    for n in names:
                        if n in crew_list:
                            continue
                        if not is_available(n):
                            continue
                        if not couple_ok(n):
                            continue
                        cand_pool.append(n)

                    pick = choose_best_candidate(
                        cand_pool,
                        d,
                        already_assigned=crew_list,
                        shift_counts=shift_counts,
                        pair_counts=pair_counts,
                        last_assigned_week=last_assigned_week,
                        rng_seed=3100 + idx + len(crew_list),
                        allow_consecutive_if_exhausted=True,
                    )

            else:
                cand_pool = []
                for n in names:
                    if n in crew_list:
                        continue
                    if not is_available(n):
                        continue
                    if not couple_ok(n):
                        continue
                    cand_pool.append(n)

                pick = choose_best_candidate(
                    cand_pool,
                    d,
                    already_assigned=crew_list,
                    shift_counts=shift_counts,
                    pair_counts=pair_counts,
                    last_assigned_week=last_assigned_week,
                    rng_seed=4000 + idx + len(crew_list),
                    allow_consecutive_if_exhausted=True,
                )

            if pick is None:
                break

            place(slot, pick)
            partner = couple_partner.get(pick)
            if partner:
                slotn = next_empty_slot()
                if slotn:
                    place(slotn, partner)

        # Apply to grid
        col = d.strftime("%d-%b")
        for r in all_rows:
            # If role not needed (non-HC Member 4), keep blank
            if r not in needed_roles:
                grid.loc[r, col] = ""
            else:
                grid.loc[r, col] = assigned.get(r, "")

        # update pairing stats using whole crew (lead + members)
        update_pair_counts([assigned[r] for r in needed_roles], pair_counts)

    return grid, shift_counts


# =========================
# Media engine
# =========================
def build_media_people(df: pd.DataFrame) -> pd.DataFrame:
    # Expected: Name (A), Team lead (B), Stream Director (C), Camera (D), Projection (E), Sound (F)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Name"] = df["Name"].map(norm_name)
    df = df[df["Name"] != ""].drop_duplicates(subset=["Name"]).reset_index(drop=True)

    def col_exact(name: str) -> str:
        for c in df.columns:
            if c.strip().lower() == name.strip().lower():
                return c
        return name

    c_lead = col_exact("Team lead")
    c_sd = col_exact("Stream Director")
    c_cam = col_exact("Camera")
    c_proj = col_exact("Projection")
    c_sound = col_exact("Sound")

    out = pd.DataFrame({
        "Name": df["Name"],
        "lead_eligible": df.get(c_lead, "").apply(is_yes),
        "q_stream": df.get(c_sd, "").apply(is_yes),
        "q_camera": df.get(c_cam, "").apply(is_yes),
        "q_projection": df.get(c_proj, "").apply(is_yes),
        "q_sound": df.get(c_sound, "").apply(is_yes),
    })
    return out


def generate_media_roster(
    services: List[ServiceDate],
    people: pd.DataFrame,
    availability: Dict[str, Set[pd.Timestamp]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Output grid: rows=roles, cols=dates (formatted dd-MMM)
    Team Lead is extra slot and does NOT count in shift load.
    """
    names = people["Name"].tolist()
    eligible_leads = set(people.loc[people["lead_eligible"] == True, "Name"].tolist())

    # Qualification maps
    q = {
        "Sound Crew": set(people.loc[people["q_sound"] == True, "Name"].tolist()),
        "Projectionist": set(people.loc[people["q_projection"] == True, "Name"].tolist()),
        "Stream Director": set(people.loc[people["q_stream"] == True, "Name"].tolist()),
        "Cam 1": set(people.loc[people["q_camera"] == True, "Name"].tolist()),
    }

    shift_counts: Dict[str, int] = {n: 0 for n in names}  # counts only core roles, not Team Lead
    last_assigned_week: Dict[str, pd.Timestamp] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}

    # Role-stagnation prevention: track per-person role counts
    role_counts: Dict[Tuple[str, str], int] = {}  # (person, role)->count

    date_labels = [s.date.strftime("%d-%b") for s in services]
    grid = pd.DataFrame(index=MEDIA_ROLES, columns=date_labels, data="")

    for idx, s in enumerate(services):
        d = s.date
        col = d.strftime("%d-%b")
        assigned = {r: "" for r in MEDIA_ROLES}

        def is_available(person: str) -> bool:
            return d in availability.get(person, set())

        crew_list: List[str] = []

        # Fill core roles
        for r_i, role in enumerate(MEDIA_CORE_ROLES):
            # candidate must be available, qualified, not already assigned in this service
            candidates = []
            for n in (q.get(role, set()) & set(names)):
                if n in crew_list:
                    continue
                if not is_available(n):
                    continue
                candidates.append(n)

            if not candidates:
                assigned[role] = ""
                continue

            # First apply weighted rotation + mixing + weekly rest by choose_best_candidate,
            # but add cross-training preference: among the tied best candidates, prefer those
            # with the lowest count in THIS role.
            pick = choose_best_candidate(
                candidates,
                d,
                already_assigned=crew_list,
                shift_counts=shift_counts,
                pair_counts=pair_counts,
                last_assigned_week=last_assigned_week,
                rng_seed=5000 + idx * 10 + r_i,
                allow_consecutive_if_exhausted=True,
            )

            if pick:
                # Cross-training: if there are others equally good by shift/mix,
                # we can re-tie-break by role_counts.
                # We'll reconstruct a tie pool with same shift & penalty & rest eligibility.
                min_shifts = min(shift_counts.get(c, 0) for c in candidates)
                pool = [c for c in candidates if shift_counts.get(c, 0) == min_shifts]

                # rest filter if possible
                non_consec = [c for c in pool if not week_rest_blocked(c, d, last_assigned_week)]
                pool = non_consec if non_consec else pool

                penalties = {c: pairing_penalty(c, crew_list, pair_counts) for c in pool}
                min_pen = min(penalties.values()) if penalties else 0
                pool = [c for c in pool if penalties[c] == min_pen]

                # Cross-training tie-break: lowest role count
                min_role = min(role_counts.get((c, role), 0) for c in pool) if pool else 0
                pool = [c for c in pool if role_counts.get((c, role), 0) == min_role]

                # shuffle to avoid alphabetical bias
                seeded_shuffle(pool, 6000 + idx * 10 + r_i)

                pick = pool[0] if pool else pick

                assigned[role] = pick
                crew_list.append(pick)
                shift_counts[pick] += 1
                last_assigned_week[pick] = d
                role_counts[(pick, role)] = role_counts.get((pick, role), 0) + 1

        # Cam 2 placeholder always blank
        assigned["Cam 2"] = ""

        # Team lead extra slot: choose from eligible lead pool that is AVAILABLE.
        # Prefer: (a) among eligible leads, those in MEDIA_LEAD_PRIORITY in order,
        # but still apply weekly rest & mixing to avoid repeating if possible.
        lead_candidates = [n for n in eligible_leads if is_available(n)]
        # If no candidates, leave blank
        lead_pick = None
        if lead_candidates:
            # make a priority-first filtered list, but don't force it if they'd violate rest when others exist
            priority_present = [n for n in MEDIA_LEAD_PRIORITY if n in lead_candidates]

            # Build a scored selection:
            # primary: priority bucket (0 if in priority list else 1)
            # secondary: pairing penalty with crew_list
            # tertiary: weekly rest (prefer non-consecutive if possible)
            # tie: shuffle
            def lead_sort_key(n: str):
                pri = 0 if n in priority_present else 1
                pen = pairing_penalty(n, crew_list, pair_counts)
                rest = 1 if week_rest_blocked(n, d, last_assigned_week) else 0
                return (pri, rest, pen)

            # Split to avoid consecutive if possible
            non_consec = [n for n in lead_candidates if not week_rest_blocked(n, d, last_assigned_week)]
            pool = non_consec if non_consec else lead_candidates

            # compute best by key
            best_key = min(lead_sort_key(n) for n in pool)
            pool2 = [n for n in pool if lead_sort_key(n) == best_key]
            seeded_shuffle(pool2, 7000 + idx)
            lead_pick = pool2[0] if pool2 else None

        assigned["Media Team Lead"] = lead_pick or ""

        # Note: team lead DOES NOT count as shift load; but does participate in pairing history
        full_crew_for_pairs = [assigned[r] for r in MEDIA_CORE_ROLES if assigned[r]]
        if assigned["Media Team Lead"]:
            full_crew_for_pairs.append(assigned["Media Team Lead"])

        update_pair_counts(full_crew_for_pairs, pair_counts)

        for r in MEDIA_ROLES:
            grid.loc[r, col] = assigned.get(r, "")

    return grid, shift_counts


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Ministry Scheduler", layout="wide")

st.title("Ministry Scheduler (Media Tech + Welcome)")
st.caption("Edit in app → Export to CSV. Availability is entered each run (not stored).")

with st.sidebar:
    st.header("Stage 1 — Ministry")
    ministry = st.selectbox("Choose ministry", ["Media Tech", "Welcome Ministry"], key="ministry_select")

    # Enforce isolation reset on change
    if "ministry" not in st.session_state:
        st.session_state["ministry"] = ministry
    if ministry != st.session_state["ministry"]:
        reset_for_ministry(ministry)

    sheet_id = st.text_input("Google Sheet ID", key="sheet_id", help="The long ID in the Google Sheets URL")

    st.divider()
    st.header("Stage 2 — Months")
    opts = month_options()
    labels = [o[2] for o in opts]
    default_idxs = list(range(0, 3))
    selected = st.multiselect("Select months to schedule", labels, default=[labels[i] for i in default_idxs])

    st.divider()
    st.header("Actions")
    if st.button("Reload members from sheet"):
        st.session_state["members_loaded"] = False


def ensure_members_loaded():
    if not st.session_state.get("sheet_id"):
        st.info("Enter your Google Sheet ID in the sidebar to load members.")
        return None

    if st.session_state.get("members_loaded", False) and "people_df" in st.session_state:
        return st.session_state["people_df"]

    try:
        if st.session_state["ministry"] == "Welcome Ministry":
            raw = fetch_sheet_gid(st.session_state["sheet_id"], WELCOME_GID)
            people = build_welcome_people(raw)
        else:
            raw = fetch_sheet_gid(st.session_state["sheet_id"], MEDIA_GID)
            people = build_media_people(raw)

        st.session_state["people_df"] = people
        st.session_state["members_loaded"] = True
        return people
    except Exception as e:
        st.error(f"Failed to load sheet data: {e}")
        return None


people_df = ensure_members_loaded()
if people_df is None:
    st.stop()

names = people_df["Name"].tolist()

# =========================
# Build service dates
# =========================
# Convert selected months into service list
services: List[ServiceDate] = []
selected_set = set(selected)
for (y, m, label) in opts:
    if label in selected_set:
        for d in all_sundays(y, m):
            services.append(ServiceDate(date=d, hc=False, combined=False, notes=""))

services = sorted(services, key=lambda s: s.date)

if not services:
    st.warning("Select at least one month to create Sundays.")
    st.stop()

# Persist service metadata across reruns
if "service_meta" not in st.session_state:
    st.session_state["service_meta"] = {s.date.strftime("%Y-%m-%d"): {"HC": False, "Combined": False, "Notes": ""} for s in services}

# Merge new services if months changed
for s in services:
    key = s.date.strftime("%Y-%m-%d")
    if key not in st.session_state["service_meta"]:
        st.session_state["service_meta"][key] = {"HC": False, "Combined": False, "Notes": ""}

# Drop services not in current selection
current_keys = {s.date.strftime("%Y-%m-%d") for s in services}
for k in list(st.session_state["service_meta"].keys()):
    if k not in current_keys:
        del st.session_state["service_meta"][k]

st.subheader("Stage 2 — Service dates (edit flags/notes)")
meta_rows = []
for s in services:
    k = s.date.strftime("%Y-%m-%d")
    meta = st.session_state["service_meta"][k]
    meta_rows.append({
        "Date": s.date.strftime("%Y-%m-%d"),
        "HC": bool(meta.get("HC", False)),
        "Combined": bool(meta.get("Combined", False)),
        "Notes": str(meta.get("Notes", "")),
    })
meta_df = pd.DataFrame(meta_rows)

edited_meta = st.data_editor(
    meta_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "HC": st.column_config.CheckboxColumn("HC"),
        "Combined": st.column_config.CheckboxColumn("Combined"),
        "Notes": st.column_config.TextColumn("Notes"),
    },
    key="meta_editor",
)

# write back edited meta to session state
for _, row in edited_meta.iterrows():
    st.session_state["service_meta"][row["Date"]] = {"HC": bool(row["HC"]), "Combined": bool(row["Combined"]), "Notes": str(row["Notes"])}

# rebuild services with latest meta
services2: List[ServiceDate] = []
for s in services:
    k = s.date.strftime("%Y-%m-%d")
    meta = st.session_state["service_meta"][k]
    services2.append(ServiceDate(date=s.date, hc=bool(meta["HC"]), combined=bool(meta["Combined"]), notes=str(meta["Notes"])))
services = services2

# =========================
# Stage 3 availability
# =========================
st.subheader("Stage 3 — Availability (enter each run)")
st.caption("Mark who is AVAILABLE for each date. Unavailable people are strictly excluded for that date.")

date_labels = [s.date.strftime("%d-%b") for s in services]
date_map = {s.date.strftime("%d-%b"): s.date for s in services}

if "availability_matrix" not in st.session_state:
    # default: everyone available
    st.session_state["availability_matrix"] = pd.DataFrame(
        index=names,
        columns=date_labels,
        data=True
    )

# If members list changed, align matrix
avail = st.session_state["availability_matrix"].copy()
avail = avail.reindex(index=names, fill_value=True)
avail = avail.reindex(columns=date_labels, fill_value=True)
st.session_state["availability_matrix"] = avail

edited_avail = st.data_editor(
    st.session_state["availability_matrix"],
    use_container_width=True,
    column_config={c: st.column_config.CheckboxColumn(c) for c in date_labels},
    key="avail_editor",
)

st.session_state["availability_matrix"] = edited_avail

# build availability dict: person -> set of dates they are available
availability: Dict[str, Set[pd.Timestamp]] = {}
for person in edited_avail.index:
    availability[person] = {date_map[c] for c in edited_avail.columns if bool(edited_avail.loc[person, c])}


# =========================
# Stage 4 generate roster
# =========================
st.subheader("Stage 4 — Generate + Edit roster")
colA, colB = st.columns([1, 2])
with colA:
    generate = st.button("Generate roster", type="primary")
with colB:
    st.caption("After generation, you can edit any cell directly before exporting to CSV.")

if generate:
    if st.session_state["ministry"] == "Welcome Ministry":
        roster, loads = generate_welcome_roster(services, people_df, availability)
        st.session_state["roster"] = roster
        st.session_state["loads"] = loads
    else:
        roster, loads = generate_media_roster(services, people_df, availability)
        st.session_state["roster"] = roster
        st.session_state["loads"] = loads

if "roster" not in st.session_state:
    st.info("Click **Generate roster** to create the schedule.")
    st.stop()

# Show notes/flags separately (since output grid is roles x dates)
with st.expander("View service flags/notes", expanded=False):
    flag_df = pd.DataFrame([{
        "Date": s.date.strftime("%d-%b"),
        "HC": s.hc,
        "Combined": s.combined,
        "Notes": s.notes
    } for s in services])
    st.dataframe(flag_df, use_container_width=True, hide_index=True)

# Editable roster grid (roles rows, date columns)
st.markdown("#### Roster (editable)")
roster_df = st.session_state["roster"].copy()

edited_roster = st.data_editor(
    roster_df,
    use_container_width=True,
    key="roster_editor",
)

st.session_state["roster"] = edited_roster

# Load stats
st.markdown("#### Load stats (current session)")
loads = st.session_state.get("loads", {})
load_df = pd.DataFrame({"Name": list(loads.keys()), "Shifts": list(loads.values())}).sort_values(["Shifts", "Name"])
st.dataframe(load_df, use_container_width=True, hide_index=True)

# Export CSV
st.markdown("#### Export")
# export with first column "Details" as in screenshot
export_df = edited_roster.copy()
export_df.insert(0, "Details", export_df.index)
export_df_reset = export_df.reset_index(drop=True)

csv_bytes = export_df_reset.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name=f"{st.session_state['ministry'].replace(' ', '_').lower()}_roster.csv",
    mime="text/csv",
)

st.caption("CSV export matches the grid: first column is role (Details), each subsequent column is a service date.")
