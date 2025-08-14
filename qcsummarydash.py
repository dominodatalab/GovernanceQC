# qcsumarydash.py  (Streamlit)
# Governance Explorer App

# === [ANCHOR:imports] ========================================================
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from urllib.parse import urlparse, urljoin

import pandas as pd
import requests
import streamlit as st
# === [END:imports] ===========================================================


# === [ANCHOR:config] =========================================================
def _resolve_api_host() -> str:
    """
    Resolve the Domino API base URL without hardcoding.
    Precedence:
      1) Public host from site config
      2) API_HOST (explicit override for local dev / CI)
      3) DOMINO_API_HOST (standard Domino env var)
      4) DOMINO_RUN_HOST_PATH (Apps-only fallback: extract scheme://host)
    """
    api_proxy = os.getenv("DOMINO_API_PROXY")
    if api_proxy:
        config = requests.get(f"{api_proxy}/cliSiteConfig")
        return config.json()["host"]
    
    v = os.getenv("API_HOST")
    if v:
        return v.rstrip("/")

    v = os.getenv("DOMINO_API_HOST")
    if v:
        return v.rstrip("/")

    rhp = os.getenv("DOMINO_RUN_HOST_PATH")
    if rhp:
        p = urlparse(rhp)
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"

    return ""  # unresolved

def _build_url(path: str) -> str:
    base = API_HOST.rstrip('/') + '/'
    return base + path.lstrip('/')

API_HOST = _resolve_api_host()
API_KEY = os.getenv("DOMINO_USER_API_KEY")

st.set_page_config(page_title="Governance Explorer App", layout="wide")

# Optional global failsafe to silence *any* accidental debug UI
# Set GE_SUPPRESS_DEBUG=0 if you ever want to use st.write/json/experimental_show.
if os.getenv("GE_SUPPRESS_DEBUG", "1") == "1":
    st.write = lambda *a, **k: None
    st.json = lambda *a, **k: None
    if hasattr(st, "experimental_show"):
        st.experimental_show = lambda *a, **k: None
    if hasattr(st, "echo"):
        @contextmanager
        def _null_cm():
            yield
        # Make st.echo a no-op context manager
        st.echo = _null_cm

# Validate auth & host early with clear guidance
if not API_KEY:
    st.error("No Domino user API key found. Please enable authentication in the App Launcher.")
    st.stop()

if not API_HOST:
    st.error(
        "Could not resolve Domino API host. Set DOMINO_API_HOST on the cluster, "
        "or provide API_HOST for local development. (Apps also supply DOMINO_RUN_HOST_PATH.)"
    )
    st.stop()

HEADERS = {"X-Domino-Api-Key": API_KEY, "Accept": "application/json"}
# === [END:config] ============================================================


# === [ANCHOR:helpers] ========================================================
def _get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def parse_dt(x: Any) -> Optional[datetime]:
    if not x:
        return None
    if isinstance(x, (int, float)):
        try:
            return datetime.fromtimestamp(float(x)/1000.0, tz=timezone.utc)
        except Exception:
            return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        return None

def fmt_user(u: Optional[Dict[str, Any]]) -> str:
    if not u:
        return "Unassigned"
    return u.get("userName") or u.get("name") or "Unassigned"

def latest_time(candidates: List[Any]) -> Optional[datetime]:
    best = None
    for c in candidates:
        dt = parse_dt(c)
        if dt and (best is None or dt > best):
            best = dt
    return best

def safe_branch_from_attachments(atts: List[Dict[str, Any]]) -> str:
    if not atts:
        return ""
    def key(a):
        ts = parse_dt(a.get("createdAt"))
        return (0, ts) if ts else (1, datetime.min.replace(tzinfo=timezone.utc))
    atts_sorted = sorted(atts, key=key, reverse=True)
    for a in atts_sorted:
        ident = a.get("identifier", {})
        br = ident.get("branch")
        if br:
            return br
    return ""

# Sensible human ordering for common stage sets
CANONICAL_ORDERS = [
    ["Developer Checklist", "Independent QC Programming", "Lead Reviewer Signoff"],
    ["Self QC", "Double Programming", "Code Review"],
    ["Independent Programming", "Comparison & Reconciliation", "Approval"],
    [
        "Developer Preparation & Documentation",
        "Independent Double Programming",
        "Independent Statistical Review",
        "Regulatory QA Signoff",
    ],
]

def order_stages_like_humans(stage_names: List[str]) -> List[str]:
    if not stage_names:
        return []
    names = list(dict.fromkeys(stage_names))
    for canon in CANONICAL_ORDERS:
        if all(n in names for n in canon):
            ordered = [n for n in canon if n in names]
            extra = [n for n in names if n not in ordered]
            return ordered + extra
    return names

def stage_assignee_for_name(stages: List[Dict[str, Any]], stage_name: str) -> str:
    best_ts = None
    best_user = None
    for s in stages:
        s_name = (s.get("stage") or {}).get("name")
        if s_name != stage_name:
            continue
        ts = parse_dt(s.get("assignedAt"))
        if ts and (best_ts is None or ts > best_ts):
            best_ts = ts
            best_user = s.get("assignee")
    return fmt_user(best_user)

def compute_last_updated(bundle: Dict[str, Any]) -> Optional[datetime]:
    times = [bundle.get("createdAt")]
    for a in bundle.get("attachments", []):
        times.append(a.get("createdAt"))
    for s in bundle.get("stages", []):
        times.append(s.get("assignedAt"))
    for p in bundle.get("policies", []):
        times.append(p.get("createdAt"))
        times.append(p.get("deactivatedAt"))
    return latest_time(times)

def current_stage_assignee(bundle: Dict[str, Any]) -> str:
    sa = bundle.get("stageAssignee")
    if sa and sa.get("name"):
        return sa["name"]
    curr_stage = bundle.get("stage")
    best_ts = None
    best_user = None
    for s in bundle.get("stages", []):
        s_name = (s.get("stage") or {}).get("name")
        if s_name != curr_stage:
            continue
        ts = parse_dt(s.get("assignedAt"))
        if ts and (best_ts is None or ts > best_ts):
            best_ts = ts
            best_user = s.get("assignee")
    return fmt_user(best_user)

def days_in_current_stage(bundle: Dict[str, Any]) -> Optional[float]:
    curr_stage = bundle.get("stage")
    ts = None
    for s in bundle.get("stages", []):
        s_name = (s.get("stage") or {}).get("name")
        if s_name == curr_stage:
            cand = parse_dt(s.get("assignedAt"))
            if cand and (ts is None or cand > ts):
                ts = cand
    if ts is None:
        ts = parse_dt(bundle.get("createdAt"))
    if ts is None:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
# === [END:helpers] ===========================================================

# === [ANCHOR:helpers_charts] ================================================
import altair as alt

def filter_archived(bundles: List[Dict[str, Any]], hide_archived: bool) -> List[Dict[str, Any]]:
    """Return bundles filtered by 'Archived' state when requested."""
    if not hide_archived:
        return bundles
    return [b for b in bundles if (b.get("state", "") or "").lower() != "archived"]


def build_stage_lineage_from_events(
    events: List[Dict[str, Any]],
    current_stage: str
) -> List[Dict[str, Any]]:
    """
    From audit events, build a simple lineage:
      [{'stage': str, 'assignee': str, 'start': dt, 'end': dt or None, 'is_current': bool}]
    We treat fieldChanges on 'stage' as transitions and capture the assignee closest
    to that time (if present) for display.
    """
    # sort ASC to build intervals
    evs = sorted(events, key=lambda e: parse_dt(e.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
    out: List[Dict[str, Any]] = []

    def _assignee_from_event(e):
        for t in e.get("targets", []):
            for fc in t.get("fieldChanges", []) or []:
                if fc.get("fieldName") == "assignee":
                    added = fc.get("added") or []
                    if added and added[0].get("name"):
                        return added[0]["name"]
        actor = (e.get("actor") or {}).get("name")
        return actor or "Unassigned"

    last_stage = None
    last_start = None
    last_assignee = "Unassigned"

    for e in evs:
        ts = parse_dt(e.get("timestamp"))
        if not ts:
            continue

        # capture assignee updates if they happen
        for t in e.get("targets", []):
            for fc in t.get("fieldChanges", []) or []:
                if fc.get("fieldName") == "assignee":
                    last_assignee = _assignee_from_event(e)

        # stage transitions
        for t in e.get("targets", []):
            for fc in t.get("fieldChanges", []) or []:
                if fc.get("fieldName") == "stage":
                    # close previous
                    if last_stage is not None:
                        out.append({
                            "stage": last_stage,
                            "assignee": last_assignee,
                            "start": last_start,
                            "end": ts,
                            "is_current": False,
                        })
                    # open new
                    last_stage = fc.get("after") or ""
                    last_start = ts

    # close/open current at the end
    if last_stage is not None:
        out.append({
            "stage": last_stage,
            "assignee": last_assignee,
            "start": last_start,
            "end": None,
            "is_current": (last_stage == current_stage)
        })

    # If we never saw a stage change, create a single segment that starts at min event time
    if not out and evs:
        first_ts = parse_dt(evs[0].get("timestamp"))
        out.append({
            "stage": current_stage or "Unknown",
            "assignee": last_assignee,
            "start": first_ts,
            "end": None,
            "is_current": True
        })

    return out


def _lineage_frame(lineage: List[Dict[str, Any]]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = []
    for seg in lineage:
        rows.append({
            "Stage": seg["stage"] or "",
            "Assignee": seg["assignee"] or "Unassigned",
            "Start": seg["start"],
            "End": seg["end"] or now,
            "Status": "Current" if seg.get("is_current") else "Normal",
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Start")
    return df


def render_lineage_swimlane(lineage: List[Dict[str, Any]]):
    """
    Altair swimlane with *cleaner x-axis*: one tick per day, labels '%b %d' (e.g., Aug 07).
    """
    df = _lineage_frame(lineage)
    if df.empty:
        st.info("No stage lineage to display.")
        return

    color = alt.Scale(
        domain=["Current", "Normal"],
        range=["#22c55e", "#93c5fd"],
    )

    chart = (
        alt.Chart(df)
        .mark_bar(size=22, cornerRadius=6)
        .encode(
            x=alt.X(
                "Start:T",
                title=None,
                axis=alt.Axis(
                    format="%b %d",
                    tickCount="day",   # <-- daily ticks; cleaner than '12 PM' etc.
                    labelAngle=0,
                    grid=False
                ),
            ),
            x2="End:T",
            y=alt.Y("Stage:N", title="Stage"),
            color=alt.Color("Status:N", scale=color, legend=alt.Legend(title=None, orient="right")),
            tooltip=[
                alt.Tooltip("Stage:N"),
                alt.Tooltip("Assignee:N"),
                alt.Tooltip("Start:T", format="%Y-%m-%d %H:%M UTC"),
                alt.Tooltip("End:T", format="%Y-%m-%d %H:%M UTC"),
            ],
        )
        .properties(height=160, width="container")
        .interactive()
    )

    # Add assignee labels in the middle of each bar
    text = (
        alt.Chart(df)
        .mark_text(dy=-10, size=10)
        .encode(
            x=alt.X("Start:T"),
            x2="End:T",
            y=alt.Y("Stage:N"),
            text="Assignee:N",
        )
    )

    st.altair_chart(chart + text, use_container_width=True)


def render_lineage_vertical(lineage: List[Dict[str, Any]]):
    """
    Minimal HTML vertical timeline.
    The previous issue you saw (raw HTML showing) happens when st.code/st.text is used.
    We use st.markdown(..., unsafe_allow_html=True).
    """
    df = _lineage_frame(lineage)
    if df.empty:
        st.info("No stage lineage to display.")
        return

    # inject CSS once
    if "ge_vertical_css" not in st.session_state:
        st.markdown("""
        <style>
          .ge-vline {border-left: 2px solid #e5e7eb; margin-left: 16px; padding-left: 16px;}
          .ge-card {margin: 10px 0; padding: 8px 12px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px;}
          .ge-stage {font-weight: 600;}
          .ge-assignee {color: #64748b; font-size: 12px;}
          .ge-ts {color: #475569; font-size: 12px;}
        </style>
        """, unsafe_allow_html=True)
        st.session_state["ge_vertical_css"] = True

    html = ['<div class="ge-vline">']
    for _, r in df.iterrows():
        html.append(
            f"""
            <div class="ge-card">
              <div class="ge-stage">{r['Stage']}</div>
              <div class="ge-assignee">{r['Assignee']}</div>
              <div class="ge-ts">
                {r['Start'].strftime('%b %d, %H:%M UTC')} → {r['End'].strftime('%b %d, %H:%M UTC')}
              </div>
            </div>
            """
        )
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)
# === [END:helpers_charts] ====================================================


# === [ANCHOR:fetchers] =======================================================
@st.cache_data(ttl=60)
def fetch_bundles(limit: int = 1000) -> List[Dict[str, Any]]:
    url = _build_url("/api/governance/v1/bundles")
    data = _get(url, params={"limit": limit})
    if not data:
        return []
    return data.get("data") or data.get("bundles") or []

@st.cache_data(ttl=60)
def fetch_audit_events(params: Dict[str, Any]) -> Dict[str, Any]:
    url = _build_url("/api/audittrail/v1/auditevents")
    return _get(url, params=params) or {"events": [], "estimatedMatches": 0}
# === [END:fetchers] ==========================================================


# === [ANCHOR:ui_top] =========================================================
st.title("Governance Explorer App")
tab1, tab2, tab3 = st.tabs(["All Bundles", "Bundle History", "Metrics"])
# === [END:ui_top] ============================================================


# === [ANCHOR:tab1_all_bundles] ===============================================
with tab1:
    st.caption("Current state of all governance bundles.")

    # Unique key to avoid duplicate-id error across tabs
    hide_archived_tab1 = st.checkbox("Hide archived bundles", value=True, key="hide_archived_tab1")

    all_bundles = fetch_bundles(limit=1000)
    bundles = filter_archived(all_bundles, hide_archived_tab1)

    bundles_sorted = sorted(bundles, key=lambda b: (b.get("name") or "").lower())

    rows = []
    for b in bundles_sorted:
        name = b.get("name", "")
        state = b.get("state", "")
        curr_stage = b.get("stage", "")
        curr_assignee = current_stage_assignee(b)
        last_updated = compute_last_updated(b)
        project = b.get("projectName", "")
        policy = b.get("policyName", "")
        created = parse_dt(b.get("createdAt"))
        owner = b.get("projectOwner") or (b.get("createdBy") or {}).get("userName") or ""
        stage_names_present = [
            (s.get("stage") or {}).get("name")
            for s in b.get("stages", [])
            if (s.get("stage") or {}).get("name")
        ]
        ordered_names = order_stages_like_humans(stage_names_present)[:4]
        s1, s2, s3, s4 = (ordered_names + ["", "", "", ""])[:4]

        r = {
            "Bundle Name": name,
            "State": state,
            "Current Stage": curr_stage,
            "Current Stage Assignee": curr_assignee,
            "Last Updated": last_updated.isoformat().replace("+00:00", "Z") if last_updated else "",
            "Project Name": project,
            "Policy Name": policy,
            "Date bundle created": created.isoformat().replace("+00:00", "Z") if created else "",
            "Owner of bundle": owner,
            "Stage 1 Name": s1,
            "Stage 1 Assignee": stage_assignee_for_name(b.get("stages", []), s1) if s1 else "Unassigned",
            "Stage 2 Name": s2,
            "Stage 2 Assignee": stage_assignee_for_name(b.get("stages", []), s2) if s2 else "Unassigned",
            "Stage 3 Name": s3,
            "Stage 3 Assignee": stage_assignee_for_name(b.get("stages", []), s3) if s3 else "Unassigned",
            "Stage 4 Name": s4,
            "Stage 4 Assignee": stage_assignee_for_name(b.get("stages", []), s4) if s4 else "Unassigned",
            "Repo Branch": safe_branch_from_attachments(b.get("attachments", [])),
            "Bundle ID": b.get("id", ""),
            "_days_in_stage": days_in_current_stage(b),
        }
        rows.append(r)

    if not rows:
        st.info("No bundles returned.")
    else:
        df_all = pd.DataFrame(rows)
        for col in ["Last Updated", "Date bundle created"]:
            if col in df_all.columns:
                df_all[col] = df_all[col].fillna("")
        st.dataframe(df_all.drop(columns=["_days_in_stage"]), use_container_width=True)
# === [END:tab1_all_bundles] ==================================================



# === [ANCHOR:suppressor] =====================================================
@contextmanager
def _suppress_debug_output():
    """Silence accidental debug UI (write/json/experimental_show/echo) within a block."""
    _w = getattr(st, "write", None)
    _j = getattr(st, "json", None)
    _show = getattr(st, "experimental_show", None)
    _echo = getattr(st, "echo", None)

    try:
        if _w:    st.write = lambda *a, **k: None
        if _j:    st.json = lambda *a, **k: None
        if _show: st.experimental_show = lambda *a, **k: None
        if _echo:
            @contextmanager
            def _null_cm2():
                yield
            st.echo = _null_cm2
        yield
    finally:
        if _w:    st.write = _w
        if _j:    st.json = _j
        if _show: st.experimental_show = _show
        if _echo: st.echo = _echo
# === [END:suppressor] ========================================================

# === [ANCHOR:tab2_history] ===================================================
import altair as alt

def _fmt_ts_any(x):
    # Reuse your parse_dt if available; otherwise fall back
    try:
        return parse_dt(x)
    except Exception:
        from datetime import datetime, timezone
        try:
            return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        except Exception:
            return None

def _build_stage_segments_from_events(evts, bundle_fallback_stage, bundle_created_at, current_assignee_guess="Unassigned"):
    """
    Build stage segments from chronological events.
    Uses:
      - Change Governance Bundle Stage  -> start/end segments
      - Update Governance Bundle Stage Assignee -> updates assignee for the current open segment
    """
    # Keep only events that matter for lineage/assignee
    lineage_names = {"Change Governance Bundle Stage", "Update Governance Bundle Stage Assignee"}
    seq = []
    for e in evts:
        name = (e.get("action") or {}).get("eventName")
        if name not in lineage_names:
            continue
        ts = _fmt_ts_any(e.get("timestamp"))
        if not ts:
            continue
        if name == "Change Governance Bundle Stage":
            # after = new stage
            new_stage = ""
            for t in e.get("targets", []):
                for fc in t.get("fieldChanges", []) or []:
                    if fc.get("fieldName") == "stage":
                        new_stage = str(fc.get("after") or "")
                        break
            seq.append((ts, "stage", new_stage))
        elif name == "Update Governance Bundle Stage Assignee":
            # assignee added
            a = "Unassigned"
            for t in e.get("targets", []):
                for fc in t.get("fieldChanges", []) or []:
                    if fc.get("fieldName") == "assignee":
                        added = fc.get("added") or []
                        a = added[0].get("name") if added and added[0].get("name") else "Unassigned"
                        break
            seq.append((ts, "assignee", a))

    if not seq and not bundle_fallback_stage:
        return []  # nothing we can do

    seq.sort(key=lambda x: x[0])  # chronological

    # Start values
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # Initial stage and start time
    stage = bundle_fallback_stage
    start_time = _fmt_ts_any(bundle_created_at) or (seq[0][0] if seq else now)
    assignee = current_assignee_guess

    segments = []
    for ts, kind, val in seq:
        if kind == "stage":
            # close previous segment
            if stage:
                segments.append({
                    "stage": stage,
                    "start": start_time,
                    "end": ts,
                    "assignee": assignee or "Unassigned",
                })
            stage = val
            start_time = ts
        elif kind == "assignee":
            assignee = val or "Unassigned"

    # close final open segment to "now"
    if stage and start_time:
        segments.append({
            "stage": stage,
            "start": start_time,
            "end": now,
            "assignee": assignee or "Unassigned",
        })

    return segments

def _render_swimlane(segments):
    if not segments:
        st.info("No stage history found for this bundle.")
        return

    df_seg = pd.DataFrame(segments)
    # mark current (last segment) in green
    df_seg["is_current"] = False
    if not df_seg.empty:
        df_seg.loc[df_seg.index.max(), "is_current"] = True

    # human labels
    df_seg["label"] = df_seg["assignee"].fillna("Unassigned")

    # Altair chart
    base = alt.Chart(df_seg)

    # Color: current segment green, others blue
    color = alt.condition(
        alt.datum.is_current,
        alt.value("#19a974"),   # green
        alt.value("#74a9ff")    # blue
    )

    bars = base.mark_bar(size=18).encode(
        y=alt.Y("stage:N", title="Stage", sort=list(df_seg["stage"].unique())[::-1]),
        x=alt.X("start:T", title=None),
        x2="end:T",
        color=color,
        tooltip=[
            alt.Tooltip("stage:N"),
            alt.Tooltip("assignee:N", title="Assignee"),
            alt.Tooltip("start:T", title="Start"),
            alt.Tooltip("end:T", title="End"),
        ],
    )

    labels = base.mark_text(
        align="center",
        baseline="middle",
        dy=-14,
        fontSize=11
    ).encode(
        x=alt.X("start:T"),
        y=alt.Y("stage:N"),
        text=alt.Text("label:N")
    )

    chart = (bars + labels).resolve_scale(color="independent").properties(height=max(150, 40*df_seg["stage"].nunique()))

    # Clean x-axis: day ticks, no 12 PM spam
    chart = chart.configure_axisX(
        format="%b %d",
        labelAngle=0,
        tickCount="day",
        grid=True
    )
    st.altair_chart(chart, use_container_width=True)

with tab2:
    st.caption("Audit history for a selected bundle (governance-related events only).")

    # Load bundles if not present
    try:
        bundles
    except NameError:
        bundles = fetch_bundles(limit=1000)

    # Hide archived bundles (default ON)
    hide_archived_t2 = st.checkbox(
        "Hide archived bundles",
        value=True,
        key="tab2_hide_archived_bundles"
    )

    bundle_pool = [b for b in bundles if (not hide_archived_t2 or (b.get("state") or "").lower() != "archived")]

    # Bundle selector
    name_list = sorted(list({b.get("name","") for b in bundle_pool if b.get("name")}), key=str.lower)
    selected_name = st.selectbox(
        "Select bundle",
        options=name_list,
        index=0 if name_list else None,
        placeholder="Choose a bundle…",
        key="tab2_bundle_select"
    )

    # Filters
    col_ev, col_proj = st.columns([3, 3])
    with col_ev:
        event_catalog = [
            "Create Governance Bundle",
            "Change Governance Bundle Stage",
            "Change Governance Bundle State",
            "Create Governance Bundle Stage Approval Request",
            "Accept Governance Bundle Stage Approval Request",
            "Update Governance Bundle Stage Assignee",
            "Add Policy to Governance Bundle",
            "Deactivate Policy in Governance Bundle",
            "Add Attachment to Bundle",              # keep in the list for user filtering
            "Remove Attachment from Bundle",
            "Submit Results in a Bundle",
            "Copy Governance Bundle results from another Bundle",
        ]
        ev_choices = st.multiselect(
            "Event filter (optional)",
            options=event_catalog,
            default=[],   # empty = show all
            placeholder="Choose events to filter (leave empty to show all)",
            key="tab2_event_multiselect"
        )

    with col_proj:
        projects = sorted(list({b.get("projectName","") for b in bundle_pool if b.get("projectName")}))
        proj_filter = st.multiselect("Project filter (optional)", projects, default=[], key="tab2_project_multiselect")

    start_iso = st.text_input("Start (UTC)", placeholder="YYYY/MM/DD", key="tab2_start")
    end_iso   = st.text_input("End (UTC)",   placeholder="YYYY/MM/DD", key="tab2_end")

    if not selected_name:
        st.info("Pick a bundle to see its audit trail.")
    else:
        # Resolve bundle and id
        candidates = [b for b in bundles if b.get("name") == selected_name]
        candidates.sort(
            key=lambda x: parse_dt(x.get("createdAt")) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        bundle = candidates[0] if candidates else None
        bundle_id = bundle.get("id") if bundle else None
        bundle_created_at = bundle.get("createdAt") if bundle else None
        current_stage_guess = bundle.get("stage") if bundle else ""
        current_assignee_guess = current_stage_assignee(bundle) if bundle else "Unassigned"

        if not bundle_id:
            st.warning("Couldn't find that bundle.")
        else:
            # Fetch audit events
            params = {
                "targetType": "governanceBundle",
                "targetId": bundle_id,
                "limit": 500,
                "sort": "-timestamp",
            }
            if start_iso:
                try:
                    _ = parse_dt(start_iso)
                    params["since"] = start_iso if "T" in start_iso else start_iso + "T00:00:00Z"
                except Exception:
                    pass
            if end_iso:
                try:
                    _ = parse_dt(end_iso)
                    params["until"] = end_iso if "T" in end_iso else end_iso + "T23:59:59Z"
                except Exception:
                    pass

            raw = fetch_audit_events(params)
            events = raw.get("events", [])

            # Synthesize "Add Attachment to Bundle" from bundle attachments
            attachments = (bundle or {}).get("attachments", []) or []
            synth = []
            for a in attachments:
                ts = a.get("createdAt")
                actor = (a.get("createdBy") or {}).get("userName") or (a.get("creator") or "")
                synth.append({
                    "timestamp": ts,
                    "action": {"eventName": "Add Attachment to Bundle"},
                    "actor": {"name": actor},
                    "in": {"name": bundle.get("projectName","")},
                    # keep extras in a custom payload so they can be shown later if desired
                    "_attachment": {
                        "type": a.get("type") or "",
                        "identifier": a.get("identifier") or {},
                    }
                })

            # Merge and sort
            all_events = (events or []) + synth
            # Apply filters (note: attachment items also have action.eventName)
            def _keep(e):
                ok = True
                if ev_choices:
                    ok = ok and ((e.get("action") or {}).get("eventName") in ev_choices)
                if proj_filter:
                    ok = ok and (((e.get("in") or {}).get("name")) in proj_filter)
                return ok
            all_events = [e for e in all_events if _keep(e)]
            # descending -> reverse to ascending for lineage calc
            all_events_sorted_asc = sorted(all_events, key=lambda e: _fmt_ts_any(e.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))

            # --- Stage lineage (swimlane only) ---
            st.subheader("Stage lineage")
            segments = _build_stage_segments_from_events(
                all_events_sorted_asc,
                bundle_fallback_stage=current_stage_guess,
                bundle_created_at=bundle_created_at,
                current_assignee_guess=current_assignee_guess
            )
            _render_swimlane(segments)

            # --- Event details table ---
            st.subheader("Event details")
            rows_h = []
            for e in sorted(all_events, key=lambda x: _fmt_ts_any(x.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)):
                ts = _fmt_ts_any(e.get("timestamp"))
                actor = (e.get("actor") or {}).get("name") or ""
                proj = (e.get("in") or {}).get("name") or ""
                bundle_nm = selected_name
                action_name = (e.get("action") or {}).get("eventName") or ""

                stage_name, before, after, field = "", "", "", ""
                if action_name in ("Change Governance Bundle Stage", "Change Governance Bundle State", "Update Governance Bundle Stage Assignee"):
                    for a in e.get("affecting", []) or []:
                        if a.get("entityType") == "governancePolicyStage" and a.get("name"):
                            stage_name = a["name"]
                            break
                    for t in e.get("targets", []):
                        for fc in t.get("fieldChanges", []) or []:
                            fname = fc.get("fieldName")
                            if fname in ("stage","state"):
                                before, after, field = str(fc.get("before") or ""), str(fc.get("after") or ""), fname
                            elif fname == "assignee":
                                added = fc.get("added") or []
                                removed = fc.get("removed") or []
                                before = removed[0].get("name") if removed and removed[0].get("name") else "Unassigned"
                                after  = added[0].get("name") if added and added[0].get("name")  else "Unassigned"
                                field = "assignee"
                elif action_name == "Add Attachment to Bundle":
                    field = "attachment"
                    att = e.get("_attachment") or {}
                    after = (att.get("type") or "")  # short descriptor
                    # stage_name can stay blank for attachment events

                rows_h.append({
                    "Time (UTC)": ts.strftime("%Y-%m-%d %H:%M:%S (UTC)") if ts else "",
                    "Action": action_name,
                    "Stage": stage_name,
                    "User": actor,
                    "Project": proj,
                    "Bundle": bundle_nm,
                    "Before": before,
                    "After": after,
                    "Change": field,
                })

            if rows_h:
                dfh = pd.DataFrame(rows_h)
                st.dataframe(dfh, use_container_width=True, height=380)
            else:
                st.info("No audit events found with the current filters.")
# === [END:tab2_history] ======================================================

# === [ANCHOR:tab3_metrics] ===================================================
with tab3:
    st.caption("Which bundles have been sitting in the same stage the longest?")

    # Unique key avoids duplicate-id error across tabs
    hide_archived_tab3 = st.checkbox("Hide archived bundles", value=True, key="hide_archived_tab3")

    try:
        all_bundles
    except NameError:
        all_bundles = fetch_bundles(limit=1000)
    bundles = filter_archived(all_bundles, hide_archived_tab3)

    if not bundles:
        st.info("No bundles available.")
    else:
        rows = []
        for b in bundles:
            rows.append({
                "Bundle Name": b.get("name",""),
                "Project Name": b.get("projectName",""),
                "Policy Name": b.get("policyName",""),
                "Current Stage": b.get("stage",""),
                "Current Stage Assignee": current_stage_assignee(b),
                "Days in Current Stage": days_in_current_stage(b),
            })
        dfm = pd.DataFrame(rows)
        dfm["Days in Current Stage"] = pd.to_numeric(dfm["Days in Current Stage"], errors="coerce").fillna(-1)
        dfm_sorted = dfm.sort_values(by="Days in Current Stage", ascending=False, na_position="last")

        st.dataframe(dfm_sorted, use_container_width=True)

        top = dfm_sorted[dfm_sorted["Days in Current Stage"] >= 0].head(15)
        if not top.empty:
            st.bar_chart(
                top.set_index("Bundle Name")["Days in Current Stage"],
                use_container_width=True,
            )
        else:
            st.info("No valid stage-duration data yet (need current-stage timestamps).")
# === [END:tab3_metrics] ======================================================