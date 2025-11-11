import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Market Cap Builder (Wide Sheets)", layout="wide")

# ======== DEFAULTS (edit if your tab names change) ========
DEFAULT_PRICE_SHEET_ID = "1tJ_Enw4NWzynxaADv2wEr0B2bS4H-FVJXkFXbCuG0qc"   # Price book
DEFAULT_PRICE_TAB      = "MASTER"

DEFAULT_SHARES_SHEET_ID = "1iXaWvwpfBXKqJXrEipKzXD0r2q7VhUwyU4b5n7QlURs"  # Shares book
DEFAULT_SHARES_TAB      = "Current_Shares"


# ===================== HELPERS =====================
def parse_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide format:
      - Row 0: date headers (B..)
      - Col A: tickers (A2..)
      - Grid: numeric values
    Returns: DataFrame index=ticker, columns=datetime (unique), values=float.
    Handles duplicate dates by collapsing duplicates, keeping the RIGHTMOST non-null.
    """
    if df.empty:
        return df

    # 1) Set headers from first row; ensure first col is 'Ticker'
    headers = df.iloc[0, :].tolist()
    headers[0] = headers[0] if pd.notna(headers[0]) and str(headers[0]).strip() else "Ticker"
    df.columns = headers

    # 2) Drop header row; normalize ticker col
    df = df.iloc[1:, :].copy()
    df.rename(columns={df.columns[0]: "Ticker"}, inplace=True)
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    # 3) Parse date headers (allow multiple formats)
    date_cols_raw = [c for c in df.columns if c != "Ticker"]
    parsed_dates = []
    for c in date_cols_raw:
        dt = pd.to_datetime(c, errors="coerce", dayfirst=False)
        if pd.isna(dt):
            dt = pd.to_datetime(c, errors="coerce", dayfirst=True)
        parsed_dates.append(dt)

    # Keep only columns that parsed to a real datetime
    keep_mask = [pd.notna(x) for x in parsed_dates]
    kept_raw = [c for c, k in zip(date_cols_raw, keep_mask) if k]
    kept_parsed = [d for d in parsed_dates if pd.notna(d)]

    # 4) Subset and rename to parsed datetime column names
    sub = df[["Ticker"] + kept_raw].copy()
    rename_map = {old: new for old, new in zip(kept_raw, kept_parsed)}
    sub.rename(columns=rename_map, inplace=True)

    # 5) Index by ticker
    sub.set_index("Ticker", inplace=True)

    # 6) Handle duplicate datetime columns robustly
    value_cols = list(sub.columns)
    if len(value_cols) != len(set(value_cols)):
        # Collapse duplicates: RIGHTMOST non-null wins
        combined = {}
        # preserve chronological order for final sort; grouping by exact datetime value
        unique_dates = sorted(set(value_cols))
        # We need original column order to ensure "rightmost wins"
        original_order = list(sub.columns)
        for d in unique_dates:
            dupes = [col for col in original_order if col == d]
            # Start with all-NaN
            cur = pd.Series(np.nan, index=sub.index, dtype="float64")
            # Iterate LEFT->RIGHT but assign using series.combine_first(cur) so later (right) overrides earlier
            for col in dupes:
                series = pd.to_numeric(sub[col], errors="coerce")
                cur = series.combine_first(cur)
            combined[d] = cur
        sub = pd.DataFrame(combined, index=sub.index)
    else:
        # Columns are unique; simple numeric coercion
        for c in value_cols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # 7) Sort columns chronologically
    sub = sub.reindex(sorted(sub.columns), axis=1)

    return sub


@st.cache_data(show_spinner=False)
def read_sheet_via_gspread(sa_json: dict, spreadsheet_id: str, worksheet_name: str) -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(sa_json, scopes=scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    return pd.DataFrame(values)


@st.cache_data(show_spinner=False)
def read_sheet_via_csv(spreadsheet_id: str, worksheet_name: str) -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq?tqx=out:csv&sheet={worksheet_name}"
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.warning(f"CSV fetch failed for {worksheet_name}: {e}")
        return pd.DataFrame()


def build_adjusted_shares(shares_wide: pd.DataFrame, price_dates: pd.Index, fixed_cutoff: pd.Timestamp | None) -> pd.DataFrame:
    """
    Rule:
      - For dates BEFORE earliest shares date (or fixed_cutoff if set) → use shares at that date
      - For dates ON/AFTER → carry-forward latest known shares ≤ date
    Returns shares aligned to price_dates (same order).
    """
    if shares_wide.empty:
        return shares_wide

    shares_wide = shares_wide.copy().reindex(sorted(shares_wide.columns), axis=1)
    earliest_col = shares_wide.columns.min()
    first_shares_date = pd.to_datetime(fixed_cutoff) if fixed_cutoff is not None else earliest_col

    out_cols = pd.Index(sorted(set(price_dates)))
    shares_aligned = shares_wide.reindex(columns=out_cols)
    shares_ff = shares_aligned.ffill(axis=1)

    # For columns earlier than first_shares_date → use the snapshot at first_shares_date
    early_cols = out_cols[out_cols < first_shares_date]
    if len(early_cols) > 0 and first_shares_date in shares_ff.columns:
        base_vals = shares_ff[first_shares_date]
        # broadcast base_vals to early columns
        shares_ff.loc[:, early_cols] = shares_ff.loc[:, early_cols].apply(lambda col: base_vals, axis=0)

    return shares_ff.reindex(columns=price_dates)


def compute_market_cap(price_wide: pd.DataFrame, shares_wide: pd.DataFrame, fixed_cutoff: pd.Timestamp | None) -> pd.DataFrame:
    common = price_wide.index.intersection(shares_wide.index)
    if len(common) == 0:
        common = price_wide.index
    p = price_wide.reindex(index=common)
    adj_sh = build_adjusted_shares(shares_wide.reindex(index=common), p.columns, fixed_cutoff)
    return p * adj_sh


def _format_date_idx(idx: pd.Index) -> pd.Index:
    return pd.Index([d.strftime("%Y-%m-%d") for d in idx])


def choose_auto_window(n_points: int, span_days: int) -> int:
    """
    Adaptive smoothing window (in days) for long ranges.
    We NEVER drop points — the raw series is fully plotted,
    and smoothing is an overlay with min_periods=1.
    """
    if span_days <= 120:
        return 1       # no smoothing
    if span_days <= 365:
        return 7       # ~1 week
    if span_days <= 3 * 365:
        return 14      # ~2 weeks
    if span_days <= 7 * 365:
        return 30      # ~1 month
    return 60          # ~2 months for very long spans


def build_series_for_chart(df_wide: pd.DataFrame, ticker: str, d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    """Return long DataFrame: date, value (filtered by date range)."""
    if ticker not in df_wide.index:
        return pd.DataFrame(columns=["date", "value"])
    s = df_wide.loc[ticker]
    s = s[(s.index >= d1) & (s.index <= d2)]
    out = pd.DataFrame({"date": s.index, "value": s.values}).dropna()
    return out


# ===================== UI =====================
st.title("📈 Market Cap Builder (Wide Sheets, Carry-Forward Shares)")
st.caption("Before earliest shares date → use that snapshot; otherwise use last known shares ≤ date. Charts include auto-smoothing overlays while keeping all points visible.")

with st.sidebar:
    st.header("Data Sources")

    # Auto-detect if we have service account in secrets
    secrets_has_sa = "gcp_service_account" in st.secrets
    mode_default = 0 if secrets_has_sa else 1

    access_mode = st.radio(
        "How to read Google Sheets?",
        ["Service Account (st.secrets)", "CSV (public/published)", "Upload Service Account JSON"],
        index=mode_default,
        help="Use secrets for private sheets; CSV works if tabs are public. Upload is a local fallback."
    )

    price_sheet_id = st.text_input("Price Sheet ID", value=DEFAULT_PRICE_SHEET_ID)
    price_tab = st.text_input("Price Tab Name", value=DEFAULT_PRICE_TAB)

    shares_sheet_id = st.text_input("Shares Sheet ID", value=DEFAULT_SHARES_SHEET_ID)
    shares_tab = st.text_input("Shares Tab Name", value=DEFAULT_SHARES_TAB)

    # Credentials selection
    sa_json = None
    if access_mode == "Upload Service Account JSON":
        uploaded = st.file_uploader("Upload service-account JSON", type=["json"], accept_multiple_files=False)
        if uploaded is not None:
            sa_json = json.loads(uploaded.read().decode("utf-8"))
    elif access_mode == "Service Account (st.secrets)" and secrets_has_sa:
        sa_json = dict(st.secrets["gcp_service_account"])

    st.divider()
    st.subheader("Computation Options")
    use_fixed_cutoff = st.checkbox("Force fixed cutoff date (override earliest shares column)?", value=False)
    fixed_cutoff_date = None
    if use_fixed_cutoff:
        fixed_cutoff_date = st.date_input("Fixed cutoff date", value=datetime(2025, 11, 10))
        fixed_cutoff_date = pd.to_datetime(fixed_cutoff_date)

    st.divider()
    st.subheader("Display & Debug")
    show_raw = st.checkbox("Show raw tables (Price & Shares)", value=False)
    show_debug = st.checkbox("Show duplicate-date diagnostics", value=False)
    default_tickers = st.text_input("Preselect tickers (comma-separated)", value="")
    st.button("🔄 Clear cache & reload", on_click=lambda: st.cache_data.clear())


# ===================== LOAD =====================
with st.spinner("Loading sheets..."):
    if access_mode in ("Service Account (st.secrets)", "Upload Service Account JSON"):
        if sa_json is None:
            st.info("Provide service-account credentials (secrets or upload), or switch to CSV mode if your sheets are public.")
            st.stop()
        raw_price = read_sheet_via_gspread(sa_json, price_sheet_id, price_tab)
        raw_shares = read_sheet_via_gspread(sa_json, shares_sheet_id, shares_tab)
    else:
        raw_price = read_sheet_via_csv(price_sheet_id, price_tab)
        raw_shares = read_sheet_via_csv(shares_sheet_id, shares_tab)

if raw_price.empty or raw_shares.empty:
    st.error("One or both sheets returned no data. Check IDs/tab names and sharing settings.")
    st.stop()

price_wide = parse_wide(raw_price)
shares_wide = parse_wide(raw_shares)

if price_wide.empty:
    st.error("Price sheet parsed empty. Verify header row and first column layout.")
    st.stop()
if shares_wide.empty:
    st.error("Shares sheet parsed empty. Verify header row and first column layout.")
    st.stop()

# Optional debug
if show_debug:
    # Reconstruct from raw to detect duplicates BEFORE collapsing
    def count_dupes(df_raw):
        headers = df_raw.iloc[0, :].tolist()
        headers[0] = headers[0] if pd.notna(headers[0]) and str(headers[0]).strip() else "Ticker"
        date_headers = [h for h in headers[1:] if h]
        # Try parse to uniform string format for easier human eyeballing
        parsed = []
        for h in date_headers:
            d = pd.to_datetime(h, errors="coerce", dayfirst=False)
            if pd.isna(d):
                d = pd.to_datetime(h, errors="coerce", dayfirst=True)
            parsed.append(d.strftime("%Y-%m-%d") if pd.notna(d) else str(h))
        dupes = pd.Series(parsed).duplicated().sum()
        return dupes

    st.info(f"Duplicate parsed dates (price raw headers): ~{count_dupes(raw_price)}")
    st.info(f"Duplicate parsed dates (shares raw headers): ~{count_dupes(raw_shares)}")

# ===================== COMPUTE =====================
marketcap_wide = compute_market_cap(price_wide, shares_wide, fixed_cutoff_date)

# ===================== CONTROLS (table views) =====================
all_tickers = marketcap_wide.index.tolist()
preselect = [t.strip() for t in default_tickers.split(",") if t.strip()] if default_tickers else []
preselect = [t for t in preselect if t in all_tickers]

sel_tickers = st.multiselect(
    "Choose tickers (table views)",
    options=all_tickers,
    default=preselect if preselect else all_tickers[: min(10, len(all_tickers))],
)

all_dates = marketcap_wide.columns
date_min, date_max = all_dates.min(), all_dates.max()
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date (table)", value=date_min.date(), min_value=date_min.date(), max_value=date_max.date())
with col2:
    end_date = st.date_input("End date (table)", value=date_max.date(), min_value=date_min.date(), max_value=date_max.date())

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
if start_dt > end_dt:
    st.warning("Start date is after end date; swapping.")
    start_dt, end_dt = end_dt, start_dt

p_view = price_wide.reindex(index=sel_tickers, columns=all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)])
s_view = shares_wide.reindex(index=sel_tickers, columns=all_dates[(all_dates >= start_dt) & (all_dates <= end_dt)])
m_view = marketcap_wide.reindex(index=sel_tickers, columns=p_view.columns)

# ===================== DISPLAY =====================
if show_raw:
    st.subheader("Price (raw, wide)")
    st.dataframe(p_view.rename(columns=dict(zip(p_view.columns, _format_date_idx(p_view.columns)))))

    st.subheader("Shares (raw, wide)")
    st.dataframe(s_view.rename(columns=dict(zip(s_view.columns, _format_date_idx(s_view.columns)))))

st.subheader("Market Cap (Computed, wide)")
st.dataframe(m_view.rename(columns=dict(zip(m_view.columns, _format_date_idx(m_view.columns)))))

# ===================== GRAPH BUILDER =====================
st.markdown("## 📉 Graph Builder (Ticker + 2 Dates, with Auto/Manual Smoothing)")

# Inputs specific to the chart
c1, c2, c3 = st.columns([2,2,2])
with c1:
    chart_ticker = st.selectbox("Ticker (chart)", options=all_tickers, index=0 if all_tickers else None)
with c2:
    chart_metric = st.selectbox("Metric", options=["MarketCap", "Price", "Shares"], index=0)
with c3:
    smoothing_mode = st.selectbox(
        "Smoothing",
        options=["Auto", "Off", "Manual"],
        index=0,
        help="Auto chooses a rolling window by date span; overlay keeps all points."
    )

dcol1, dcol2 = st.columns(2)
with dcol1:
    d1 = st.date_input("Chart start date", value=date_min.date(), min_value=date_min.date(), max_value=date_max.date())
with dcol2:
    d2 = st.date_input("Chart end date", value=date_max.date(), min_value=date_min.date(), max_value=date_max.date())

d1 = pd.to_datetime(d1)
d2 = pd.to_datetime(d2)
if d1 > d2:
    d1, d2 = d2, d1

if chart_ticker:
    # Pick the source data
    src_df = {"MarketCap": marketcap_wide, "Price": price_wide, "Shares": shares_wide}[chart_metric]
    long = build_series_for_chart(src_df, chart_ticker, d1, d2)

    if long.empty:
        st.info("No data to chart for the selected range.")
    else:
        n_points = len(long)
        span_days = (long["date"].max() - long["date"].min()).days if n_points > 1 else 0

        # Determine the window
        if smoothing_mode == "Off":
            win = 1
        elif smoothing_mode == "Manual":
            win = st.slider("Rolling window (days)", min_value=1, max_value=120, value=choose_auto_window(n_points, span_days))
        else:
            win = choose_auto_window(n_points, span_days)

        # Build smoothed series (overlay). min_periods=1 keeps every timestamp.
        smoothed = long.copy()
        smoothed["smooth"] = smoothed["value"].rolling(window=win, min_periods=1).mean()

        # Altair chart: raw line + points + smoothed line (overlay)
        base = alt.Chart(long).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title=chart_metric),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",")]
        )
        raw_line = base.mark_line(opacity=0.4).interactive()
        raw_pts  = base.mark_circle(size=12, opacity=0.25)

        smooth_line = alt.Chart(smoothed).encode(
            x="date:T",
            y=alt.Y("smooth:Q", title=chart_metric),
            tooltip=[alt.Tooltip("date:T"), alt.Tooltip("smooth:Q", format=",")]
        ).mark_line(strokeWidth=2)

        st.altair_chart((raw_line + raw_pts + smooth_line).properties(height=420), use_container_width=True)
        st.caption(f"Points: {n_points}  •  Span: {span_days} days  •  Rolling window used: {win} day(s)")

# ===================== DOWNLOAD =====================
csv_buf = io.StringIO()
m_view.to_csv(csv_buf)
st.download_button(
    "Download Market Cap (CSV for table selection)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="marketcap_wide.csv",
    mime="text/csv",
)

with st.expander("How the computation works"):
    st.write("""
- Both sheets are parsed as **wide** tables (tickers in column A, dates on row 1).
- We collapse **duplicate date headers** safely (rightmost non-null wins).
- Dates are aligned across both sheets.
- Cutoff date = earliest shares column by default (or a fixed date if you toggle it).
- For price dates **before** that cutoff → use **shares at the cutoff date**.
- For price dates **on/after** → use **last known shares ≤ date** (row-wise forward-fill).
- Market Cap = Price × Adjusted Shares (element-wise).
- The chart shows **all points** and overlays an **adaptive rolling mean** (no points dropped).
""")
