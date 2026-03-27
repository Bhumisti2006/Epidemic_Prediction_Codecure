# ============================================================
#  EpiCast — Epidemic Intelligence Platform
#  Hackathon Project: Epidemic Spread Prediction (Epidemiology + AI)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="EpiCast — Epidemic Intelligence Platform",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME & CUSTOM CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: #070b14;
    color: #c9d1e8;
}

/* ── Main background ── */
.stApp { background: #070b14; }
section[data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1a2035; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 22px; }
.kpi-card {
    background: linear-gradient(135deg, #0f1628 0%, #121a2e 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 22px;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.red::before   { background: linear-gradient(90deg, #e83a5e, #ff6b8a); }
.kpi-card.blue::before  { background: linear-gradient(90deg, #3a7bd5, #6fb3f7); }
.kpi-card.green::before { background: linear-gradient(90deg, #00c9a7, #4dffd4); }
.kpi-card.amber::before { background: linear-gradient(90deg, #f4a522, #ffd166); }

.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; letter-spacing: 0.12em;
             text-transform: uppercase; color: #4a5980; margin-bottom: 8px; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600;
             line-height: 1; }
.kpi-card.red   .kpi-value { color: #e83a5e; }
.kpi-card.blue  .kpi-value { color: #6fb3f7; }
.kpi-card.green .kpi-value { color: #00c9a7; }
.kpi-card.amber .kpi-value { color: #f4a522; }
.kpi-sub { font-size: 11px; color: #3a4a6b; margin-top: 6px; }

/* ── Section headers ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.15em;
    text-transform: uppercase; color: #3a5080;
    border-left: 2px solid #1e3060;
    padding-left: 10px; margin: 20px 0 14px;
}

/* ── Alert boxes ── */
.alert-high {
    background: rgba(232,58,94,0.08); border: 1px solid rgba(232,58,94,0.3);
    border-left: 3px solid #e83a5e; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #f4a0b0; margin: 10px 0;
}
.alert-medium {
    background: rgba(244,165,34,0.08); border: 1px solid rgba(244,165,34,0.3);
    border-left: 3px solid #f4a522; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #f4d090; margin: 10px 0;
}
.alert-low {
    background: rgba(0,201,167,0.08); border: 1px solid rgba(0,201,167,0.3);
    border-left: 3px solid #00c9a7; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #90f4e0; margin: 10px 0;
}

/* ── Sidebar ── */
.sidebar-stat {
    background: #0f1628; border: 1px solid #1a2a44;
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
}
.sidebar-stat .label { font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.1em; color: #3a5080; }
.sidebar-stat .val { font-family: 'IBM Plex Mono', monospace; font-size: 20px;
    font-weight: 600; color: #e83a5e; }

/* ── Selectbox / widgets ── */
.stSelectbox > div > div { background: #0f1628 !important; border-color: #1e2d4a !important; }
.stSlider .stSliderTrack { background: #1e2d4a !important; }
div[data-testid="stMetric"] { background: #0f1628; border: 1px solid #1e2d4a; border-radius: 10px; padding: 12px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: #0d1220; border: 1px solid #1a2540;
    border-radius: 8px; color: #4a6090;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    letter-spacing: 0.08em; padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f2040, #1a3060) !important;
    border-color: #2a4a8a !important; color: #7ab0f0 !important;
}

/* ── Hotspot table ── */
.hotspot-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 12px; border-radius: 8px;
    background: #0d1220; border: 1px solid #1a2540;
    margin-bottom: 6px;
}
.hs-rank { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #3a5080; width: 28px; }
.hs-name { font-size: 13px; font-weight: 600; flex: 1; }
.hs-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    padding: 3px 10px; border-radius: 20px; font-weight: 600;
}
.badge-critical { background: rgba(232,58,94,.15); color: #f07090; border: 1px solid rgba(232,58,94,.3); }
.badge-high     { background: rgba(244,165,34,.15); color: #f4c060; border: 1px solid rgba(244,165,34,.3); }
.badge-medium   { background: rgba(100,149,237,.15); color: #80a8f0; border: 1px solid rgba(100,149,237,.3); }
.badge-low      { background: rgba(0,201,167,.15); color: #60e8cc; border: 1px solid rgba(0,201,167,.3); }
</style>
""", unsafe_allow_html=True)


# ─── PLOTLY DARK TEMPLATE ────────────────────────────────────
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="#070b14",
    plot_bgcolor="#0a0f1e",
    font=dict(family="IBM Plex Mono, monospace", color="#6a7fa0", size=11),
    xaxis=dict(gridcolor="#0f1a2e", linecolor="#1a2540", zerolinecolor="#1a2540", showgrid=True),
    yaxis=dict(gridcolor="#0f1a2e", linecolor="#1a2540", zerolinecolor="#1a2540", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a2540", borderwidth=1),
    margin=dict(t=40, b=40, l=50, r=20),
)

def apply_theme(fig, **overrides):
    merged = {**LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**merged)
    return fig

def styled_fig():
    fig = go.Figure()
    apply_theme(fig)
    return fig

# ─── SAFE VLINE HELPER ──────────────────────────────────────
def safe_vline(fig, ts, label, color="#2a4060"):
    """
    Add a vertical line + annotation that works on Python 3.14 + new pandas/plotly.
    Converts any Timestamp / datetime to an ISO string before passing to Plotly,
    avoiding the 'int + Timestamp' arithmetic bug inside shapeannotation._mean().
    """
    x_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=x_str, x1=x_str,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=color, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=x_str, y=1,
        xref="x", yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=9, color=color),
        xanchor="left",
        bgcolor="rgba(0,0,0,0)",
    )


# ─── HELPERS ────────────────────────────────────────────────
def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(int(n))

def smooth(arr, sigma=3):
    clipped = np.clip(arr, 0, np.percentile(arr[arr > 0], 99) if np.any(arr > 0) else 1)
    return gaussian_filter1d(clipped.astype(float), sigma=sigma)

def risk_color(level):
    return {"Critical": "#e83a5e", "High": "#f4a522", "Medium": "#6495ed", "Low": "#00c9a7"}.get(level, "#6a7fa0")

def risk_badge_class(level):
    return {"Critical": "badge-critical", "High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-low")

def outbreak_risk(n):
    if n > 50000: return "Critical"
    if n > 10000: return "High"
    if n > 1000:  return "Medium"
    return "Low"


# ─── DATA LOADING ────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    CONFIRMED = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    DEATHS    = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

    conf   = pd.read_csv(CONFIRMED)
    deaths = pd.read_csv(DEATHS)

    def melt_and_group(df):
        melted = df.melt(
            id_vars=["Province/State", "Country/Region", "Lat", "Long"],
            var_name="Date", value_name="Cases"
        )
        melted["Date"] = pd.to_datetime(melted["Date"], infer_datetime_format=True)
        return melted.groupby(["Country/Region", "Date"])["Cases"].sum().reset_index()

    conf_grouped   = melt_and_group(conf)
    deaths_grouped = melt_and_group(deaths)

    population = {
        "US": 331e6, "India": 1380e6, "Brazil": 212e6, "France": 67e6,
        "Germany": 83e6, "United Kingdom": 67e6, "Italy": 60e6, "Russia": 146e6,
        "Turkey": 84e6, "Spain": 47e6, "Vietnam": 97e6, "Argentina": 45e6,
        "China": 1400e6, "Japan": 125e6, "Korea, South": 52e6, "Mexico": 128e6,
        "Indonesia": 274e6, "South Africa": 60e6, "Pakistan": 220e6, "Iran": 85e6,
        "Australia": 25e6, "Canada": 38e6, "Netherlands": 17e6, "Belgium": 11e6,
        "Sweden": 10e6, "Poland": 38e6, "Colombia": 51e6, "Philippines": 110e6,
        "Malaysia": 33e6, "Thailand": 70e6, "Chile": 19e6, "Austria": 9e6,
        "Greece": 11e6, "Portugal": 10e6, "Ukraine": 44e6, "Israel": 9e6,
        "Switzerland": 9e6, "Denmark": 6e6, "Singapore": 6e6, "New Zealand": 5e6,
    }

    merged = conf_grouped.merge(
        deaths_grouped.rename(columns={"Cases": "Deaths"}),
        on=["Country/Region", "Date"], how="left"
    )
    merged["Deaths"] = merged["Deaths"].fillna(0)

    latest = merged[merged["Date"] == merged["Date"].max()].copy()
    latest["Population"] = latest["Country/Region"].map(population).fillna(50e6)
    latest["Cases_per_Million"] = (latest["Cases"] / latest["Population"] * 1e6).round(0)

    return merged, latest, population

with st.spinner("Loading epidemic data..."):
    merged, latest, population = load_data()


# ─── FEATURE ENGINEERING ─────────────────────────────────────
def build_features(country_df: pd.DataFrame) -> pd.DataFrame:
    df = country_df.sort_values("Date").copy()
    df["Cases"] = np.maximum(df["Cases"], 0)

    df["Daily_Cases"] = df["Cases"].diff().fillna(0).clip(lower=0)
    df["Daily_Smooth"] = smooth(df["Daily_Cases"].values, sigma=4)

    df["Avg_7"]  = df["Daily_Smooth"].rolling(7,  min_periods=1).mean()
    df["Avg_14"] = df["Daily_Smooth"].rolling(14, min_periods=1).mean()
    df["Avg_28"] = df["Daily_Smooth"].rolling(28, min_periods=1).mean()

    df["Growth_Rate"] = df["Avg_7"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df["Acceleration"] = df["Avg_7"].diff().diff().fillna(0)

    df["Doubling_Days"] = np.where(
        df["Growth_Rate"] > 0.001,
        np.log(2) / np.log1p(df["Growth_Rate"]),
        999.0
    )

    rolling_sum = df["Daily_Smooth"].rolling(7, min_periods=1).sum()
    df["Rt"] = (rolling_sum / rolling_sum.shift(5).replace(0, np.nan)).fillna(1.0).clip(0, 10)
    df["Rt"] = smooth(df["Rt"].values, sigma=5)

    df["Day"] = np.arange(len(df))
    return df


# ─── ML MODEL ────────────────────────────────────────────────
def forecast_country(df: pd.DataFrame, days: int = 14):
    df = build_features(df)
    df_model = df[df["Daily_Smooth"] > 0].tail(120)

    if len(df_model) < 30:
        return None, None, df

    features = ["Day", "Avg_7", "Avg_14", "Avg_28", "Growth_Rate", "Acceleration", "Rt"]
    X = df_model[features]
    y = df_model["Daily_Smooth"]

    rf  = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    gb  = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42)

    rf.fit(X, y)
    gb.fit(X, y)

    n_boot = 40
    boot_preds = np.zeros((n_boot, days))
    for b in range(n_boot):
        idx = np.random.choice(len(X), len(X), replace=True)
        rf_b = RandomForestRegressor(n_estimators=60, max_depth=7, random_state=b, n_jobs=-1)
        rf_b.fit(X.iloc[idx], y.iloc[idx])

        last_day    = df_model["Day"].iloc[-1]
        last_avg7   = df_model["Avg_7"].iloc[-1]
        last_avg14  = df_model["Avg_14"].iloc[-1]
        last_avg28  = df_model["Avg_28"].iloc[-1]
        last_growth = df_model["Growth_Rate"].iloc[-1]
        last_accel  = df_model["Acceleration"].iloc[-1]
        last_rt     = df_model["Rt"].iloc[-1]

        preds_b = []
        for i in range(1, days + 1):
            p = rf_b.predict(pd.DataFrame(
                [[last_day+i, last_avg7, last_avg14, last_avg28, last_growth, last_accel, last_rt]],
                columns=features))[0]
            p = max(0, p)
            last_avg7  = (last_avg7  * 6 + p) / 7
            last_avg14 = (last_avg14 * 13 + p) / 14
            last_avg28 = (last_avg28 * 27 + p) / 28
            last_growth = (p - last_avg7) / (last_avg7 + 1e-5)
            preds_b.append(p)
        boot_preds[b] = preds_b

    last_day    = df_model["Day"].iloc[-1]
    last_avg7   = df_model["Avg_7"].iloc[-1]
    last_avg14  = df_model["Avg_14"].iloc[-1]
    last_avg28  = df_model["Avg_28"].iloc[-1]
    last_growth = df_model["Growth_Rate"].iloc[-1]
    last_accel  = df_model["Acceleration"].iloc[-1]
    last_rt     = df_model["Rt"].iloc[-1]

    preds = []
    for i in range(1, days + 1):
        row = pd.DataFrame(
            [[last_day+i, last_avg7, last_avg14, last_avg28, last_growth, last_accel, last_rt]],
            columns=features)
        p = (0.55 * rf.predict(row)[0] + 0.45 * gb.predict(row)[0])
        p = max(0, p)
        last_avg7  = (last_avg7  * 6 + p) / 7
        last_avg14 = (last_avg14 * 13 + p) / 14
        last_avg28 = (last_avg28 * 27 + p) / 28
        last_growth = (p - last_avg7) / (last_avg7 + 1e-5)
        preds.append(p)

    preds = np.array(preds)
    lower = np.percentile(boot_preds, 10, axis=0)
    upper = np.percentile(boot_preds, 90, axis=0)

    future_dates = pd.date_range(df["Date"].iloc[-1], periods=days + 1)[1:]
    forecast_df = pd.DataFrame({
        "Date": future_dates, "Predicted": preds,
        "Lower": lower, "Upper": upper
    })

    return forecast_df, rf, df


# ─── SEIR MODEL ──────────────────────────────────────────────
def run_seir(N, I0, R0_val, days=120, sigma=1/5.2, gamma=1/18):
    beta = R0_val * gamma
    S, E, I, R = [N - I0 - 1], [1], [I0], [0]
    for _ in range(days):
        s, e, i, r = S[-1], E[-1], I[-1], R[-1]
        dS = -beta * s * i / N
        dE =  beta * s * i / N - sigma * e
        dI =  sigma * e - gamma * i
        dR =  gamma * i
        S.append(max(0, s + dS));  E.append(max(0, e + dE))
        I.append(max(0, i + dI));  R.append(max(0, r + dR))
    return np.array(S), np.array(E), np.array(I), np.array(R)


# ─── HOTSPOT DETECTION ───────────────────────────────────────
@st.cache_data(ttl=3600)
def detect_hotspots(top_n=20):
    countries = latest.nlargest(top_n, "Cases")["Country/Region"].tolist()
    rows = []
    for c in countries:
        cdf = merged[merged["Country/Region"] == c].sort_values("Date")
        if len(cdf) < 30: continue
        cdf = build_features(cdf)
        recent = cdf.tail(14)
        avg_accel  = recent["Acceleration"].mean()
        avg_growth = recent["Growth_Rate"].mean()
        latest_rt  = cdf["Rt"].iloc[-1]
        daily_peak = recent["Daily_Smooth"].max()
        rows.append({
            "Country": c,
            "Rt": round(latest_rt, 2),
            "Growth_Rate_14d": round(avg_growth * 100, 2),
            "Acceleration": round(avg_accel, 1),
            "Peak_Daily": int(daily_peak),
            "Risk": outbreak_risk(daily_peak),
        })
    return pd.DataFrame(rows).sort_values("Rt", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'IBM Plex Mono\', monospace; font-size:13px; font-weight:600; '
                'color:#4a7ab0; letter-spacing:0.1em;">EPICAST</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px; color:#2a3a5a; margin-top:-10px; margin-bottom:20px;">'
                'Epidemic Intelligence Platform</p>', unsafe_allow_html=True)

    years = sorted(merged["Date"].dt.year.unique())
    selected_year = st.selectbox("Year", years, index=len(years) - 1)

    all_countries = sorted(merged["Country/Region"].unique())
    default_idx = all_countries.index("India") if "India" in all_countries else 0
    selected_country = st.selectbox("Country / Region", all_countries, index=default_idx)

    forecast_days = st.slider("Forecast Horizon (days)", 7, 30, 14)

    st.divider()

    total_conf   = int(latest["Cases"].sum())
    total_deaths = int(latest["Deaths"].sum())
    cfr = round(total_deaths / total_conf * 100, 2) if total_conf else 0

    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="label">Total Confirmed</div>
        <div class="val">{fmt(total_conf)}</div>
    </div>
    <div class="sidebar-stat" style="border-color:#1a2a44;">
        <div class="label">Total Deaths</div>
        <div class="val" style="color:#9090a0;">{fmt(total_deaths)}</div>
    </div>
    <div class="sidebar-stat" style="border-color:#1a2a44;">
        <div class="label">Case Fatality Rate</div>
        <div class="val" style="color:#f4a522;">{cfr}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">Top Spread</p>', unsafe_allow_html=True)
    top10 = latest.nlargest(10, "Cases")[["Country/Region", "Cases"]]
    for _, row in top10.iterrows():
        pct = row["Cases"] / total_conf * 100
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'font-size:11px;border-bottom:1px solid #0f1628;">'
            f'<span style="color:#6a8aaa;">{row["Country/Region"][:18]}</span>'
            f'<span style="font-family:\'IBM Plex Mono\';color:#3a6090;">{pct:.1f}%</span></div>',
            unsafe_allow_html=True
        )


# ────────────────────────────────────────────────────────────────────
#  MAIN HEADER
# ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-family:\'IBM Plex Mono\',monospace; font-size:22px; font-weight:600; '
    'color:#4a7ab0; letter-spacing:0.06em; margin-bottom:4px;">EPICAST</h1>'
    '<p style="font-size:13px; color:#2a4060; margin-bottom:20px;">'
    f'Epidemic Intelligence Platform — {selected_year} Data — {selected_country}</p>',
    unsafe_allow_html=True
)

year_df    = merged[merged["Date"].dt.year == selected_year]
country_df = year_df[year_df["Country/Region"] == selected_country].copy()

with st.spinner(f"Running ML forecast for {selected_country}..."):
    forecast_df, model, enriched_df = forecast_country(country_df, days=forecast_days)

latest_cases = int(enriched_df["Cases"].iloc[-1]) if len(enriched_df) else 0
latest_daily = int(enriched_df["Daily_Smooth"].iloc[-1]) if len(enriched_df) else 0
latest_rt    = round(enriched_df["Rt"].iloc[-1], 2) if len(enriched_df) else 0
pred_d7      = int(forecast_df["Predicted"].iloc[forecast_days - 1]) if forecast_df is not None else 0
risk_now     = outbreak_risk(latest_daily)

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card red">
    <div class="kpi-label">Total Confirmed</div>
    <div class="kpi-value">{fmt(latest_cases)}</div>
    <div class="kpi-sub">{selected_country} cumulative</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-label">Current Daily (Smoothed)</div>
    <div class="kpi-value">{fmt(latest_daily)}</div>
    <div class="kpi-sub">Gaussian-smoothed trend</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Effective Rt</div>
    <div class="kpi-value">{latest_rt}</div>
    <div class="kpi-sub">Rt &gt; 1 means spreading</div>
  </div>
  <div class="kpi-card amber">
    <div class="kpi-label">Day-{forecast_days} Forecast</div>
    <div class="kpi-value">{fmt(pred_d7)}</div>
    <div class="kpi-sub">Risk: <b style="color:{risk_color(risk_now)}">{risk_now}</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

if latest_rt > 1.5:
    st.markdown(f'<div class="alert-high">Critical: Rt = {latest_rt} — Exponential growth detected. '
                f'Immediate intervention recommended.</div>', unsafe_allow_html=True)
elif latest_rt > 1.0:
    st.markdown(f'<div class="alert-medium">Warning: Rt = {latest_rt} — Disease is actively spreading. '
                f'Monitor for escalation.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="alert-low">Stable: Rt = {latest_rt} — Outbreak is contained or declining.</div>',
                unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
#  TABS
# ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Global Outbreak Map",
    "Trend + Forecast",
    "Transmission Modeling",
    "Hotspot Detection",
])


# ══════════════════════════════════════════
#  TAB 1 — GLOBAL MAP
# ══════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-title">Global Outbreak Distribution</p>', unsafe_allow_html=True)

    map_data = latest.copy()
    raw = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/"
        "csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    )
    geo = raw.groupby("Country/Region")[["Lat", "Long"]].mean().reset_index()
    map_data = map_data.merge(geo, on="Country/Region", how="left").dropna(subset=["Lat", "Long"])

    fig_map = go.Figure(go.Scattergeo(
        lat=map_data["Lat"],
        lon=map_data["Long"],
        text=map_data["Country/Region"],
        customdata=np.stack([map_data["Cases"], map_data["Deaths"]], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Confirmed: %{customdata[0]:,.0f}<br>"
            "Deaths: %{customdata[1]:,.0f}<extra></extra>"
        ),
        mode="markers",
        marker=dict(
            size=np.log1p(map_data["Cases"]) * 1.8,
            color=map_data["Cases"],
            colorscale=[[0, "#0a0f1e"], [0.3, "#1a3060"], [0.6, "#c03060"], [1.0, "#ff2050"]],
            colorbar=dict(
                title=dict(text="Cases", font=dict(size=10, color="#4a6090")),
                tickfont=dict(size=9, color="#4a6090"),
            ),
            opacity=0.85,
            line=dict(width=0.4, color="rgba(255,255,255,0.15)"),
        ),
    ))
    fig_map.update_geos(
        bgcolor="#070b14",
        landcolor="#0d1628",
        oceancolor="#050a12",
        showocean=True,
        showland=True,
        showcountries=True,
        countrycolor="#1a2540",
        coastlinecolor="#1a2540",
        lakecolor="#050a12",
        framecolor="#1a2540",
        projection_type="natural earth",
    )
    apply_theme(fig_map,
        height=460,
        margin=dict(t=10, b=10, l=0, r=0),
        geo=dict(showframe=False),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<p class="section-title">Cases per Million Population — Top 20</p>', unsafe_allow_html=True)
    top20 = latest.nlargest(20, "Cases_per_Million").sort_values("Cases_per_Million")
    fig_bar = go.Figure(go.Bar(
        x=top20["Cases_per_Million"], y=top20["Country/Region"],
        orientation="h",
        marker=dict(
            color=top20["Cases_per_Million"],
            colorscale=[[0, "#1a2a4a"], [0.5, "#3060a0"], [1, "#e83a5e"]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>Cases/M: %{x:,.0f}<extra></extra>",
    ))
    apply_theme(fig_bar, height=380, showlegend=False,
                yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 2 — TREND + FORECAST
# ══════════════════════════════════════════
with tab2:
    if len(enriched_df) < 10:
        st.warning("Insufficient data for this country/year combination.")
    else:
        recent = enriched_df.tail(90)

        st.markdown('<p class="section-title">Smoothed Daily Cases + ML Forecast</p>', unsafe_allow_html=True)

        fig2 = styled_fig()

        # Reported daily bars
        fig2.add_trace(go.Bar(
            x=recent["Date"], y=recent["Daily_Cases"],
            name="Reported Daily",
            marker=dict(color="rgba(58, 96, 160, 0.25)", line=dict(width=0)),
        ))

        # Smooth trend
        fig2.add_trace(go.Scatter(
            x=recent["Date"], y=recent["Daily_Smooth"],
            name="7-Day Smooth Trend",
            mode="lines",
            line=dict(color="#6495ed", width=2.5),
        ))

        # 7-day avg
        fig2.add_trace(go.Scatter(
            x=recent["Date"], y=recent["Avg_7"],
            name="7-Day Avg",
            mode="lines",
            line=dict(color="#f4a522", width=1.5, dash="dot"),
        ))

        if forecast_df is not None:
            # Confidence interval band
            fig2.add_trace(go.Scatter(
                x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
                y=pd.concat([forecast_df["Upper"], forecast_df["Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,201,167,0.08)",
                line=dict(width=0),
                name="80% CI",
            ))

            # Forecast line
            fig2.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["Predicted"],
                name=f"{forecast_days}-Day Forecast",
                mode="lines+markers",
                line=dict(color="#00c9a7", width=2.5, dash="dash"),
            ))

            # ✅ FIX: use safe_vline() — converts Timestamp → ISO string internally
            safe_vline(fig2, enriched_df["Date"].iloc[-1], "Latest Data", color="#2a4060")

        fig2.update_layout(
            height=380,
            barmode="overlay",
            legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
            xaxis_title=None,
            yaxis_title="Daily Cases",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 3 — TRANSMISSION MODELING (SEIR)
# ══════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">SEIR Compartmental Transmission Model</p>', unsafe_allow_html=True)

    pop_est  = population.get(selected_country, 50_000_000)
    latest_i = max(1, int(enriched_df["Daily_Smooth"].iloc[-1]) * 5)

    c1, c2, c3 = st.columns(3)
    with c1: R0_input  = st.slider("Basic Reproduction Number (R0)", 0.5, 5.0, float(max(1.1, latest_rt)), 0.1)
    with c2: sim_days  = st.slider("Simulation Duration (days)", 30, 360, 180)
    with c3: pop_scale = st.number_input("Population", value=int(pop_est), step=1_000_000, format="%d")

    S, E, I, R = run_seir(N=pop_scale, I0=latest_i, R0_val=R0_input, days=sim_days)

    seir_dates = pd.date_range(enriched_df["Date"].iloc[-1], periods=sim_days + 1)
    peak_day   = int(np.argmax(I))
    peak_inf   = int(I[peak_day])

    fig_seir = styled_fig()
    fig_seir.add_trace(go.Scatter(x=seir_dates, y=S/pop_scale*100, mode="lines",
        name="Susceptible", line=dict(color="#6495ed", width=2, shape="spline")))
    fig_seir.add_trace(go.Scatter(x=seir_dates, y=E/pop_scale*100, mode="lines",
        name="Exposed",     line=dict(color="#f4a522", width=2, shape="spline")))
    fig_seir.add_trace(go.Scatter(x=seir_dates, y=I/pop_scale*100, mode="lines",
        name="Infectious",  line=dict(color="#e83a5e", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(232,58,94,0.06)"))
    fig_seir.add_trace(go.Scatter(x=seir_dates, y=R/pop_scale*100, mode="lines",
        name="Recovered",   line=dict(color="#00c9a7", width=2, shape="spline")))

    # ✅ FIX: convert seir_dates[peak_day] Timestamp → ISO string via safe_vline
    safe_vline(fig_seir, seir_dates[peak_day],
               f"Infectious Peak: Day {peak_day}", color="#e83a5e")

    fig_seir.update_layout(height=380, xaxis_title=None, yaxis_title="% of Population",
                           yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_seir, use_container_width=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Peak Infectious",         fmt(peak_inf),  f"Day {peak_day}")
    sc2.metric("Peak Date",               str(seir_dates[peak_day].date()))
    sc3.metric("Total Affected",          f"{(1 - S[-1]/pop_scale)*100:.1f}%")
    sc4.metric("Herd Immunity Threshold", f"{(1 - 1/R0_input)*100:.1f}%")

    st.markdown('<p class="section-title">Intervention Scenario Comparison</p>', unsafe_allow_html=True)

    scenarios = {
        "No Intervention (R0)":             R0_input,
        "Moderate Measures (R0 - 30%)":     R0_input * 0.7,
        "Strong Measures (R0 - 60%)":       R0_input * 0.4,
        "Elimination Goal (R0 = 0.8)":      0.8,
    }
    colors_sc = ["#e83a5e", "#f4a522", "#6495ed", "#00c9a7"]

    fig_sc = styled_fig()
    for (label, r0), col in zip(scenarios.items(), colors_sc):
        _, _, I_sc, _ = run_seir(pop_scale, latest_i, r0, sim_days)
        fig_sc.add_trace(go.Scatter(
            x=seir_dates, y=I_sc / pop_scale * 100,
            mode="lines", name=label,
            line=dict(color=col, width=2, shape="spline"),
        ))

    fig_sc.update_layout(height=300, xaxis_title=None, yaxis_title="% Infectious",
                         yaxis=dict(ticksuffix="%"))
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════
#  TAB 4 — HOTSPOT DETECTION
# ══════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Real-time Outbreak Hotspot Detection</p>', unsafe_allow_html=True)

    with st.spinner("Scanning for hotspots..."):
        hotspot_df = detect_hotspots(top_n=25)

    col_hs1, col_hs2 = st.columns([3, 2])

    with col_hs1:
        st.markdown('<p style="font-size:11px;color:#2a4060;margin-bottom:10px;">Ranked by Effective Rt</p>',
                    unsafe_allow_html=True)
        for i, row in hotspot_df.head(12).iterrows():
            badge = risk_badge_class(row["Risk"])
            st.markdown(f"""
            <div class="hotspot-row">
                <span class="hs-rank">{i+1}</span>
                <span class="hs-name">{row['Country']}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                             color:#4a7ab0;width:60px;">Rt {row['Rt']:.2f}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                             color:#3a5070;width:70px;">{row['Growth_Rate_14d']:+.1f}%/d</span>
                <span class="hs-badge {badge}">{row['Risk']}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_hs2:
        fig_bubble = go.Figure(go.Scatter(
            x=hotspot_df["Growth_Rate_14d"],
            y=hotspot_df["Rt"],
            mode="markers+text",
            text=hotspot_df["Country"],
            textposition="top center",
            textfont=dict(size=8, color="#4a6090"),
            marker=dict(
                size=np.log1p(hotspot_df["Peak_Daily"]) * 3.5,
                color=hotspot_df["Rt"],
                colorscale=[[0, "#1a3060"], [0.5, "#f4a522"], [1, "#e83a5e"]],
                opacity=0.85,
                line=dict(width=0.5, color="rgba(255,255,255,0.1)"),
            ),
            hovertemplate="<b>%{text}</b><br>Rt: %{y:.2f}<br>14d Growth: %{x:.1f}%<extra></extra>",
        ))
        fig_bubble.add_hline(y=1.0, line=dict(color="#e83a5e", width=1, dash="dot"))
        fig_bubble.add_vline(x=0,   line=dict(color="#2a4060", width=1, dash="dot"))
        apply_theme(fig_bubble, height=440,
            xaxis_title="14-Day Growth Rate (%)", yaxis_title="Effective Rt",
            showlegend=False,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown('<p class="section-title">Outbreak Risk Matrix</p>', unsafe_allow_html=True)
    risk_map = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    hotspot_df["Risk_Num"] = hotspot_df["Risk"].map(risk_map)

    hs15 = hotspot_df.head(15).copy()

    def rt_to_color(rt):
        if rt >= 2.0:   return "#e83a5e"
        elif rt >= 1.5: return "#f4a522"
        elif rt >= 1.0: return "#6495ed"
        else:           return "#1a2a4a"

    bar_colors = [rt_to_color(r) for r in hs15["Rt"]]

    fig_heat = go.Figure(go.Bar(
        x=hs15["Country"],
        y=hs15["Rt"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        customdata=hs15[["Risk", "Growth_Rate_14d"]].values,
        hovertemplate="<b>%{x}</b><br>Rt: %{y:.2f}<br>Risk: %{customdata[0]}<br>"
                      "Growth: %{customdata[1]:.1f}%<extra></extra>",
    ))
    fig_heat.add_hline(y=1.0, line=dict(color="#e83a5e", width=1.5, dash="dash"),
                       annotation_text="Epidemic Threshold (Rt=1)",
                       annotation_font=dict(size=9, color="#e83a5e"))
    apply_theme(fig_heat, height=280, showlegend=False,
                xaxis_title=None, yaxis_title="Effective Rt",
                xaxis=dict(tickangle=-35, tickfont=dict(size=9)))
    st.plotly_chart(fig_heat, use_container_width=True)