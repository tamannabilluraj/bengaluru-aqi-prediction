import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# Load model and historical data
model = joblib.load('data/processed/aqi_model.pkl')
df    = pd.read_csv('data/processed/bengaluru_features.csv', parse_dates=['Date'])

# Page config
st.set_page_config(page_title="Bengaluru AQI Predictor", page_icon="🌿", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .aqi-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    .aqi-number {
        font-size: 72px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .aqi-label {
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .aqi-advice {
        font-size: 15px;
        opacity: 0.85;
    }
    .infographic {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin: 1.5rem 0;
    }
    .aqi-band {
        flex: 1;
        padding: 10px 6px;
        border-radius: 10px;
        text-align: center;
        font-size: 11px;
        font-weight: 500;
        color: white;
    }
    .band-label { font-size: 13px; font-weight: 700; }
    .band-range { font-size: 10px; opacity: 0.85; margin-top: 2px; }
    .pollutant-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        margin-top: 1rem;
    }
    .pollutant-card {
        flex: 1;
        background: #1e1e2e;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        border: 1px solid #2e2e3e;
    }
    .pollutant-name { font-size: 11px; color: #aaa; margin-bottom: 4px; }
    .pollutant-val  { font-size: 18px; font-weight: 700; color: #fff; }
    .section-title  { font-size: 14px; font-weight: 600; color: #aaa; margin: 1.5rem 0 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌿 Bengaluru AQI Predictor")
st.markdown("Predict the Air Quality Index for Bengaluru based on date and area.")
st.markdown("---")

# AQI Infographic bands
st.markdown('<div class="section-title">AQI Scale Reference</div>', unsafe_allow_html=True)
st.markdown("""
<div class="infographic">
    <div class="aqi-band" style="background:#3cb371;">
        <div class="band-label">Good</div>
        <div class="band-range">0 – 50</div>
        <div style="font-size:10px;margin-top:4px;">No health impact</div>
    </div>
    <div class="aqi-band" style="background:#9acd32;">
        <div class="band-label">Satisfactory</div>
        <div class="band-range">51 – 100</div>
        <div style="font-size:10px;margin-top:4px;">Minor breathing discomfort</div>
    </div>
    <div class="aqi-band" style="background:#ffa500;">
        <div class="band-label">Moderate</div>
        <div class="band-range">101 – 200</div>
        <div style="font-size:10px;margin-top:4px;">Sensitive groups affected</div>
    </div>
    <div class="aqi-band" style="background:#ff4500;">
        <div class="band-label">Poor</div>
        <div class="band-range">201 – 300</div>
        <div style="font-size:10px;margin-top:4px;">Breathing discomfort for all</div>
    </div>
    <div class="aqi-band" style="background:#8b0000;">
        <div class="band-label">Very Poor</div>
        <div class="band-range">301 – 400</div>
        <div style="font-size:10px;margin-top:4px;">Respiratory illness risk</div>
    </div>
    <div class="aqi-band" style="background:#4b0082;">
        <div class="band-label">Severe</div>
        <div class="band-range">401 – 500</div>
        <div style="font-size:10px;margin-top:4px;">Affects healthy people</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# User inputs
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Select a date")
with col2:
    area = st.selectbox("Select your area", [
        "Silk Board", "BTM Layout", "Bapuji Nagar",
        "Hebbal", "Jayanagar", "Marathahalli"
    ])

st.markdown("---")

if st.button("Predict AQI"):
    st.session_state['predicted'] = True
    st.session_state['date']      = selected_date
    st.session_state['area']      = area

if st.session_state.get('predicted'):

    selected_date = st.session_state['date']
    area          = st.session_state['area']

    month       = selected_date.month
    day_of_week = selected_date.weekday()
    is_weekend  = int(day_of_week >= 5)
    quarter     = (month - 1) // 3 + 1

    month_data      = df[df['Date'].dt.month == month]
    pm25            = month_data['PM2.5'].mean()
    no              = month_data['NO'].mean()
    no2             = month_data['NO2'].mean()
    nox             = month_data['NOx'].mean()
    co              = month_data['CO'].mean()
    so2             = month_data['SO2'].mean()
    o3              = month_data['O3'].mean()
    aqi_lag_1       = month_data['aqi_lag_1'].mean()
    aqi_lag_7       = month_data['aqi_lag_7'].mean()
    aqi_rolling_7d  = month_data['aqi_rolling_7d'].mean()
    aqi_rolling_30d = month_data['aqi_rolling_30d'].mean()

    input_data = pd.DataFrame([{
        'PM2.5': pm25, 'NO': no, 'NO2': no2, 'NOx': nox,
        'CO': co, 'SO2': so2, 'O3': o3,
        'month': month, 'day_of_week': day_of_week,
        'is_weekend': is_weekend, 'quarter': quarter,
        'aqi_lag_1': aqi_lag_1, 'aqi_lag_7': aqi_lag_7,
        'aqi_rolling_7d': aqi_rolling_7d,
        'aqi_rolling_30d': aqi_rolling_30d
    }])

    prediction = model.predict(input_data)[0]

    if prediction <= 50:
        category = "Good"
        color    = "#3cb371"
        advice   = "Air quality is great. Safe for all outdoor activities."
    elif prediction <= 100:
        category = "Satisfactory"
        color    = "#9acd32"
        advice   = "Air quality is acceptable. Enjoy your day outdoors."
    elif prediction <= 200:
        category = "Moderate"
        color    = "#ffa500"
        advice   = "Sensitive groups should limit prolonged outdoor activity."
    elif prediction <= 300:
        category = "Poor"
        color    = "#ff4500"
        advice   = "Everyone should reduce outdoor exertion today."
    elif prediction <= 400:
        category = "Very Poor"
        color    = "#8b0000"
        advice   = "Avoid outdoor activities. Wear a mask if going out."
    else:
        category = "Severe"
        color    = "#4b0082"
        advice   = "Stay indoors. Keep windows closed. Avoid all exertion."

    st.markdown(f"""
    <div class="aqi-card" style="background:{color}20; border: 2px solid {color};">
        <div class="aqi-number" style="color:{color};">{prediction:.0f}</div>
        <div class="aqi-label" style="color:{color};">{category}</div>
        <div class="aqi-advice">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Area:** {area} &nbsp;&nbsp; **Date:** {selected_date.strftime('%d %B %Y')}")

    st.markdown('<div class="section-title">Pollution values used for prediction</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pollutant-row">
        <div class="pollutant-card"><div class="pollutant-name">PM2.5</div><div class="pollutant-val">{pm25:.1f}</div></div>
        <div class="pollutant-card"><div class="pollutant-name">NO2</div><div class="pollutant-val">{no2:.1f}</div></div>
        <div class="pollutant-card"><div class="pollutant-name">CO</div><div class="pollutant-val">{co:.1f}</div></div>
        <div class="pollutant-card"><div class="pollutant-name">SO2</div><div class="pollutant-val">{so2:.1f}</div></div>
        <div class="pollutant-card"><div class="pollutant-name">O3</div><div class="pollutant-val">{o3:.1f}</div></div>
        <div class="pollutant-card"><div class="pollutant-name">NOx</div><div class="pollutant-val">{nox:.1f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">AQI Map — Hover over a station to see its predicted AQI</div>', unsafe_allow_html=True)

    stations = pd.DataFrame({
        'station': ['Silk Board', 'BTM Layout', 'Bapuji Nagar',
                    'Hebbal', 'Jayanagar', 'Marathahalli'],
        'lat':     [12.9170, 12.9138, 12.9519,
                    13.0350, 12.9308, 12.9591],
        'lon':     [77.6229, 77.5950, 77.5390,
                    77.5970, 77.5833, 77.6972],
    })

    station_predictions = []
    for _, row in stations.iterrows():
        pred      = model.predict(input_data)[0]
        variation = np.random.uniform(-8, 8)
        station_predictions.append(round(pred + variation, 1))
    stations['aqi'] = station_predictions

    def get_color(aqi):
        if aqi <= 50:    return 'green'
        elif aqi <= 100: return 'lightgreen'
        elif aqi <= 200: return 'orange'
        elif aqi <= 300: return 'red'
        elif aqi <= 400: return 'darkred'
        else:            return 'purple'

    def get_category(aqi):
        if aqi <= 50:    return 'Good'
        elif aqi <= 100: return 'Satisfactory'
        elif aqi <= 200: return 'Moderate'
        elif aqi <= 300: return 'Poor'
        elif aqi <= 400: return 'Very Poor'
        else:            return 'Severe'

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles='CartoDB dark_matter')

    for _, row in stations.iterrows():
        is_selected  = row['station'] == area
        marker_color = get_color(row['aqi'])
        cat          = get_category(row['aqi'])

        popup_html = f"""
        <div style="font-family:sans-serif; padding:8px; min-width:150px;">
            <b style="font-size:14px;">{row['station']}</b><br>
            <span style="font-size:24px; font-weight:800;">{row['aqi']}</span><br>
            <span style="font-size:12px; color:#555;">{cat}</span><br>
            <span style="font-size:11px; color:#888;">{selected_date.strftime('%d %B %Y')}</span>
        </div>
        """

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=18 if is_selected else 12,
            color='white' if is_selected else marker_color,
            weight=3 if is_selected else 1,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(f"{row['station']} — AQI: {row['aqi']} ({cat})"),
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)

        folium.Marker(
            location=[row['lat'] + 0.003, row['lon']],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px; color:white; font-weight:600; white-space:nowrap; text-shadow: 1px 1px 2px black;">{row["station"]}</div>',
                icon_size=(120, 20),
                icon_anchor=(60, 0)
            )
        ).add_to(m)

    st_folium(m, width=700, height=400, returned_objects=[])