"""
Streamlit dashboard: operator eco-score replay + fleet comparison.

Run:
    streamlit run src/dashboard/app.py
"""
import pandas as pd
import streamlit as st

from src.features.build_features import extract_cycle_features
from src.models.eco_score import compute_eco_scores, cluster_purity

st.set_page_config(page_title="Operator Eco-Driving Coach", layout="wide")
st.title("🎮 Operator Eco-Driving Coach")
st.caption(
    "Cycle-by-cycle efficiency scoring, replayed like a gamified dashboard an "
    "operator could see in the cab."
)

data_path = st.sidebar.text_input("Telemetry CSV path", "data/operator_telemetry.csv")

try:
    raw = pd.read_csv(data_path)
    cycles = extract_cycle_features(raw)
    cycles = compute_eco_scores(cycles)
    cycles, purity = cluster_purity(cycles)

    st.subheader("Fleet ranking")
    summary = (
        cycles.groupby(["operator_id", "archetype"])["eco_score"]
        .mean().round(1).reset_index()
        .sort_values("eco_score", ascending=False)
    )
    st.dataframe(summary, use_container_width=True)
    st.caption(f"Unsupervised clustering purity vs. true style: {purity:.0%}")

    st.subheader("Replay a single operator's shift")
    operator = st.selectbox("Operator", sorted(cycles.operator_id.unique()))
    op_cycles = cycles[cycles.operator_id == operator].sort_values("detected_cycle_id")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.line_chart(op_cycles.set_index("detected_cycle_id")["eco_score"])
        st.caption("Eco-score per cycle (100 = ideal)")
    with col2:
        avg = op_cycles["eco_score"].mean()
        st.metric("Average eco-score", f"{avg:.0f} / 100")
        st.metric("Avg idle % per cycle", f"{op_cycles['idle_pct'].mean():.1f}%")
        st.metric("Avg fuel / cycle", f"{op_cycles['fuel_used_gal'].mean():.3f} gal")

    low = op_cycles[op_cycles.eco_score < 50]
    if len(low) > 0:
        st.warning(f"{len(low)} cycles scored below 50 — mostly driven by excess idle time or RPM volatility.")

except FileNotFoundError:
    st.warning(
        "Data file not found. Run first:\n\n"
        "```\npython -m src.simulation.generate_operator_data --out data/operator_telemetry.csv\n```"
    )
