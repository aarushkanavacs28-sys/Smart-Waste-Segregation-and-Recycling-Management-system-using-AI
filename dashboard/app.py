from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="RobustWaste-AIoT Dashboard",
    layout="wide",
)

st.title(
    "RobustWaste-AIoT Monitoring Dashboard"
)
st.caption(
    "Virtual IoT monitoring for "
    "confidence-aware smart waste segregation."
)

csv_path = st.sidebar.text_input(
    "Event CSV path",
    "outputs/iot_events.csv",
)
auto_refresh = st.sidebar.checkbox(
    "Auto refresh",
    value=False,
)

path = Path(csv_path)

if not path.exists():
    st.info(
        f"No event log found at `{csv_path}` yet. "
        "Run inference or the MQTT subscriber."
    )
    st.stop()

df = pd.read_csv(path)

required = {
    "timestamp",
    "predicted_class",
    "confidence",
    "decision",
}
missing = required - set(df.columns)

if missing:
    st.error(
        f"CSV is missing required columns: "
        f"{sorted(missing)}"
    )
    st.stop()

df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce",
)
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Total events",
    len(df),
)
c2.metric(
    "Accepted",
    int(
        (df["decision"] == "ACCEPT")
        .sum()
    ),
)
c3.metric(
    "Rejected",
    int(
        (df["decision"] == "REJECT")
        .sum()
    ),
)
c4.metric(
    "Mean confidence",
    f"{df['confidence'].mean():.3f}",
)

st.subheader("Classification counts")
st.bar_chart(
    df["predicted_class"]
    .value_counts()
    .sort_index()
)

st.subheader("Confidence over time")
timeline = (
    df.dropna(subset=["timestamp"])
    .set_index("timestamp")[
        ["confidence"]
    ]
)
st.line_chart(timeline)

st.subheader("Decision counts")
st.bar_chart(
    df["decision"]
    .value_counts()
)

st.subheader("Latest events")
st.dataframe(
    df.sort_values(
        "timestamp",
        ascending=False,
    ).head(50),
    use_container_width=True,
)

if auto_refresh:
    import time
    time.sleep(5)
    st.rerun()
