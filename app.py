import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
from calendar import monthrange
import random
import re

st.set_page_config(page_title="COR Media Tech and Welcome Roster", layout="wide")

# =========================
# Google Sheet configuration
# =========================
SHEET_ID = "1jh6ScfqpHe7rRN1s-9NYPsm7hwqWWLjdLKTYThRRGUo"

MINISTRY_GIDS = {
    "Media Tech": "0",
    "Welcome": "2080125013",
}

def gsheet_csv_url(sheet_id: str, gid: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


# =========================
# Utilities
# =========================
def clean_col(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_yes_no(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return "Y"
    if s in {"n", "no", "false", "0"}:
        return "N"
    return ""

def parse_roles_cell(x) -> list[str]:
    """Roles can be comma-separated; empty means general."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def sunday_dates_in_month(year: int, month: int) -> list[date]:
    first = date(year, month, 1)
    last = date(year, month, monthrange(year, month)[1])
    # find first Sunday
    d = first
    while d.weekday() != 6:  # Monday=0 ... Sunday=6
        d += timedelta(days=1)
    sundays = []
    while d <= last:
        sundays.append(d)
        d += timedelta(days=7)
    return sundays

def month_key(d: date) -> str:
    return d.strftime("%Y-%m")

def pretty_service_label(d: date) -> str:
    return d.strftime("%a %d %b %Y")

def safe_person_name(x) -> str:
    return clean_col(x)

def load_team(ministry: str) -> pd.DataFrame:
    gid = MINISTRY_GIDS[ministry]
    url = gsheet_csv_url(SHEET_ID, gid)
    df = pd.read_csv(url)
    df.columns = [clean_col(c) for c in df.columns]

    # Flexible column mapping
    # Required: a name column (Name / Volunteer / Person etc.)
    name_col = None
    for c in df.columns:
        if c.lower() in {"name", "person", "volunteer", "volunteers"}:
            name_col = c
            break
    if name_col is None:
        # take first column as name as fallback
        name_col = df.columns[0]

    # Optional columns
    active_col = None
    for c in df.columns:
        if c.lower() in {"active", "is_active", "status"}:
            active_col = c
            break

    role_col = None
    for c in df.columns:
        if c.lower() in {"role", "roles", "position", "positions"}:
            role_col = c
            break

    max_col = None
    for c in df.columns:
        if c.lower() in {"max_per_month", "max", "limit"}:
            max_col = c
            break

    out = pd.DataFrame()
    out["Name"] = df[name_col].apply(safe_person_name)
    out = out[out["Name"].astype(bool)]

    if active_col:
        # interpret blank as active
        act = df[active_col].apply(normalize_yes_no)
        out["Active"] = act.replace("", "Y")
    else:
        out["Active"] = "Y"

    out = out[out["Active"] == "Y"].copy()

    if role_col:
        out["Roles"] = df[role_col].apply(parse_roles_cell)
    else:
        out["Roles"] = [[] for _ in range(len(out))]

    if max_col:
        def to_int(v):
            try:
                if pd.isna(v) or str(v).strip() == "":
                    return None
                return int(float(v))
            except Exception:
                return None
        out["MaxPerMonth"] = df[max_col].apply(to_int)
    else:
        out["MaxPerMonth"] = None

    out.reset_index(drop=True, inplace=True)
    return out


# =========================
# Roster rules per ministry
# =========================
MINISTRY_RULES = {
    "Media Tech": {
        "slots": ["Sound", "Slides", "Livestream"],
        "default_people_per_slot": {"Sound": 1, "Slides": 1, "Livestream": 1},
    },
    "Welcome": {
        "slots": ["Welcome"],
        "default_people_per_slot": {"Welcome": 2},
    },
}

def required_people_for_service(ministry: str, combined: bool, hc: bool) -> dict[str, int]:
    """
    You can tweak this logic. For now:
    - Combined service: same staffing by default
    - HC: same staffing by default (you can increase if you want)
    """
    base = MINISTRY_RULES[ministry]["default_people_per_slot"].copy()
    # Example adjustments you can enable:
    # if combined and ministry == "Welcome":
    #     base["Welcome"] = 3
    # if hc and ministry == "Media Tech":
    #     base["Sound"] = 2
    return base


# =========================
# Scheduling / assignment
# =========================
def generate_roster(
    ministry: str,
    team: pd.DataFrame,
    service_dates: list[date],
    service_meta: pd.DataFrame,
    availability: pd.DataFrame,
    seed: int = 7,
) -> pd.DataFrame:
    """
    availability index: Name; columns: service_key (YYYY-MM-DD)
      Values: Y/N/blank (blank treated as Y)
    service_meta index: service_key; columns: HC(bool), Combined(bool), Notes(str)
    """
    rng = random.Random(seed)

    slots = MINISTRY_RULES[ministry]["slots"]
    names = team["Name"].tolist()

    # Count assignments overall + per month
    overall_count = {n: 0 for n in names}
    per_month_count = {(n, mk): 0 for n in names for mk in set(month_key(d) for d in service_dates)}

    # person role capability: if roles list empty => can do any slot
    roles_map = {row["Name"]: set(row["Roles"]) for _, row in team.iterrows()}
    max_map = {row["Name"]: row["MaxPerMonth"] for _, row in team.iterrows()}

    rows = []
    for d in service_dates:
        skey = d.isoformat()
        mk = month_key(d)
        combined = bool(service_meta.loc[skey, "Combined"])
        hc = bool(service_meta.loc[skey, "HC"])

        needed = required_people_for_service(ministry, combined=combined, hc=hc)

        # Build candidate pool per slot
        service_assignments = {slot: [] for slot in slots}

        for slot in slots:
            k = needed.get(slot, 0)
            if k <= 0:
                continue

            # eligible: available + role match + not exceeding monthly max
            eligible = []
            for n in names:
                # Availability: blank = Y (easy mode)
                val = availability.loc[n, skey] if (n in availability.index and skey in availability.columns) else ""
                val = normalize_yes_no(val)
                if val == "N":
                    continue

                rset = roles_map.get(n, set())
                if len(rset) > 0 and slot not in rset:
                    continue

                mmax = max_map.get(n)
                if mmax is not None and per_month_count[(n, mk)] >= mmax:
                    continue

                eligible.append(n)

            # Sort by fairness: fewest assignments first, then fewest in month
            rng.shuffle(eligible)
            eligible.sort(key=lambda n: (overall_count[n], per_month_count[(n, mk)]))

            chosen = []
            for n in eligible:
                if len(chosen) >= k:
                    break
                # avoid double-booking same person in two slots same service
                already = any(n in service_assignments[s] for s in slots)
                if already:
                    continue
                chosen.append(n)

            service_assignments[slot] = chosen

        # Write rows (one row per slot)
        for slot in slots:
            assigned = service_assignments.get(slot, [])
            # Update counts
            for n in assigned:
                overall_count[n] += 1
                per_month_count[(n, mk)] += 1

            rows.append({
                "Date": d,
                "Service": pretty_service_label(d),
                "ServiceKey": skey,
                "Month": mk,
                "Slot": slot,
                "Assigned": ", ".join(assigned),
                "HC": "Y" if hc else "",
                "Combined": "Y" if combined else "",
                "Notes": service_meta.loc[skey, "Notes"] if "Notes" in service_meta.columns else "",
            })

    out = pd.DataFrame(rows)
    return out


# =========================
# UI
# =========================
st.title("COR Media Tech and Welcome Roster")

with st.sidebar:
    st.header("Setup")
    ministry = st.selectbox("Ministry", ["Media Tech", "Welcome"])

    st.caption("This app auto-loads from the configured Google Sheet tabs.")
    st.write("Spreadsheet ID:")
    st.code(SHEET_ID)
    st.write("Tab gid used:")
    st.code(f"{ministry}: {MINISTRY_GIDS[ministry]}")

    today = date.today()
    default_start = date(today.year, today.month, 1)
    default_end = date(today.year, today.month, monthrange(today.year, today.month)[1])

    start_month = st.date_input("Start month", default_start)
    end_month = st.date_input("End month", default_end)

    seed = st.number_input("Random seed (for tie-breaks)", min_value=1, max_value=999999, value=7, step=1)

    st.divider()
    st.header("Generate")
    generate_btn = st.button("Generate roster", type="primary")


# Validate month range
if start_month > end_month:
    st.error("Start month must be before End month.")
    st.stop()

# Convert to month spans (use first day of start_month and last day of end_month)
start = date(start_month.year, start_month.month, 1)
end = date(end_month.year, end_month.month, monthrange(end_month.year, end_month.month)[1])

# Build list of Sundays within range
service_dates = []
cur = start
while cur <= end:
    for d in sunday_dates_in_month(cur.year, cur.month):
        if start <= d <= end:
            service_dates.append(d)
    # advance to next month
    if cur.month == 12:
        cur = date(cur.year + 1, 1, 1)
    else:
        cur = date(cur.year, cur.month + 1, 1)

service_dates = sorted(list(dict.fromkeys(service_dates)))  # unique preserve order

# Load team
try:
    team = load_team(ministry)
except Exception as e:
    st.error(
        "Could not load the Google Sheet as CSV.\n\n"
        "Checklist:\n"
        "1) Google Sheet is shared as 'Anyone with the link' (Viewer)\n"
        "2) The gid is correct\n\n"
        f"Error: {e}"
    )
    st.stop()

if team.empty:
    st.warning("No active people found in the sheet/tab.")
    st.stop()

# Service metadata table
service_keys = [d.isoformat() for d in service_dates]
meta_default = pd.DataFrame(
    {
        "HC": [False] * len(service_keys),
        "Combined": [False] * len(service_keys),
        "Notes": [""] * len(service_keys),
    },
    index=service_keys,
)
meta_default.insert(0, "Service", [pretty_service_label(d) for d in service_dates])

st.subheader("1) Service details")
st.write("Mark which Sundays are Holy Communion (HC), Combined service, and add any notes.")
service_meta_edit = st.data_editor(
    meta_default,
    use_container_width=True,
    key=f"meta_{ministry}_{start}_{end}",
)

# Normalize edited meta
service_meta = service_meta_edit.copy()
service_meta = service_meta.set_index(service_meta.index)
service_meta["HC"] = service_meta["HC"].astype(bool)
service_meta["Combined"] = service_meta["Combined"].astype(bool)
service_meta["Notes"] = service_meta["Notes"].fillna("").astype(str)

st.subheader("2) Availability")
st.write("Set availability per person per Sunday. Leave blank = available (easy mode). Use N for not available.")

avail_default = pd.DataFrame(index=team["Name"].tolist(), columns=service_keys)
avail_default = avail_default.fillna("")
avail_default.insert(0, "Roles", team.set_index("Name")["Roles"].apply(lambda r: ", ".join(r) if r else "Any").reindex(avail_default.index).values)
avail_default.insert(1, "MaxPerMonth", team.set_index("Name")["MaxPerMonth"].reindex(avail_default.index).values)

availability_edit = st.data_editor(
    avail_default,
    use_container_width=True,
    key=f"avail_{ministry}_{start}_{end}",
)

# Strip helper cols
availability = availability_edit.copy()
availability = availability.drop(columns=["Roles", "MaxPerMonth"], errors="ignore")
availability.index = availability_edit.index

# Generate roster
if generate_btn:
    roster = generate_roster(
        ministry=ministry,
        team=team,
        service_dates=service_dates,
        service_meta=service_meta[["HC", "Combined", "Notes"]].copy(),
        availability=availability,
        seed=int(seed),
    )
    st.session_state["roster"] = roster

st.subheader("3) Roster")
if "roster" not in st.session_state:
    st.info("Click **Generate roster** in the sidebar.")
    st.stop()

roster = st.session_state["roster"].copy()

st.write("You can edit the Assigned column before exporting.")
roster_edit = st.data_editor(
    roster[["Service", "Slot", "Assigned", "HC", "Combined", "Notes"]],
    use_container_width=True,
    key=f"roster_{ministry}_{start}_{end}",
)

# Download
csv_bytes = roster_edit.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download roster CSV",
    data=csv_bytes,
    file_name=f"{ministry.replace(' ', '_').lower()}_roster_{start}_{end}.csv",
    mime="text/csv",
)
