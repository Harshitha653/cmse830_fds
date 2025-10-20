# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import time
import io

# ---- Page setup ----
st.set_page_config("Global AQ Dashboard", layout="wide")
st.sidebar.title("Navigate")
page = st.sidebar.radio("", ["Info", "Background", "Dashboard", "Export", "Contact"])

# ---- File paths ----
AQI_FILE = "C:\\Users\\harsh\\Downloads\\CMSE-830\\Project\\WHO_air_quality_data.csv"
TOP_N_CITIES_PIE = 10

# ---- Animation ----
SMOKE_HTML = """
<style>.smoke-wrap{height:80px;display:flex;align-items:flex-end;justify-content:center}
.smoke{width:12px;height:12px;border-radius:50%;background:rgba(180,180,180,0.35);
margin:0 6px;animation:rise 2.6s linear infinite}
.smoke.s1{animation-delay:0s}.smoke.s2{animation-delay:.4s}.smoke.s3{animation-delay:.8s}
@keyframes rise{0%{transform:translateY(0);opacity:.6}50%{opacity:.25}100%{transform:translateY(-90px);opacity:0}}</style>
<div class="smoke-wrap"><div class="smoke s1"></div><div class="smoke s2"></div><div class="smoke s3"></div></div>
<div style="text-align:center;color:#666">Processing data…</div>
"""

# ---- AQI calculation utilities ----
PM25_BP = [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
PM10_BP  = [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]
NO2_BP   = [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]

def subindex(conc, bps):
    if pd.isna(conc): return np.nan
    for C_lo, C_hi, I_lo, I_hi in bps:
        if C_lo <= conc <= C_hi:
            return round(((I_hi - I_lo) / (C_hi - C_lo)) * (conc - C_lo) + I_lo, 2)
    return bps[-1][3] if conc > bps[-1][1] else bps[0][2]

def aqi_from_df(df):
    df['pm25_aqi'] = df['pm25_concentration'].apply(lambda x: subindex(x, PM25_BP)) if 'pm25_concentration' in df.columns else np.nan
    df['pm10_aqi'] = df['pm10_concentration'].apply(lambda x: subindex(x, PM10_BP)) if 'pm10_concentration' in df.columns else np.nan
    df['no2_aqi']  = df['no2_concentration'].apply(lambda x: subindex(x, NO2_BP)) if 'no2_concentration' in df.columns else np.nan
    df['aqi_dominant'] = df[['pm25_aqi','pm10_aqi','no2_aqi']].max(axis=1, skipna=True)
    return df

# ---- Load & prepare (cached) ----
@st.cache_data
def load_data(aqi_path):
    raw = pd.read_csv(aqi_path)
    raw = raw.drop(columns=[c for c in ['reference','web_link','population_source','type_of_stations','who_ms'] if c in raw.columns], errors='ignore')
    if raw.shape[0] > 300 and raw.tail(300).isnull().all(axis=None):
        raw = raw[:-290]

    df = raw.copy()
    df = df.dropna(thresh=len(df)*0.5, axis=1)
    numeric_cols = [c for c in ['pm10_concentration','pm25_concentration','no2_concentration','pm10_tempcov','pm25_tempcov','no2_tempcov','population'] if c in df.columns]
    if numeric_cols:
        imputer = IterativeImputer(max_iter=3, random_state=0)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    if 'year' in df.columns:
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
    if 'population' in df.columns:
        df['population'] = df['population'].round().astype('Int64')
    if 'iso3' not in df.columns and 'country_code' in df.columns:
        df = df.rename(columns={'country_code':'iso3'})
    df = aqi_from_df(df)
    country_aqi = df.groupby(['iso3','country_name','year'], as_index=False).agg(country_aqi_mean=('aqi_dominant','mean'), population_total=('population', lambda x: x.dropna().sum()))
    unique_cities = df['city'].nunique() if 'city' in df.columns else 0
    return raw, df, country_aqi, unique_cities

# ---- Load Data ----
load_btn = st.sidebar.button("Load / Refresh Data")
if load_btn:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(SMOKE_HTML, unsafe_allow_html=True)
        time.sleep(0.8)
        raw_df, df, country_aqi, unique_cities = load_data(AQI_FILE)
    placeholder.empty()
else:
    try:
        raw_df, df, country_aqi, unique_cities = load_data(AQI_FILE)
    except Exception:
        raw_df = df = country_aqi = pd.DataFrame()
        unique_cities = 0

# ---- Pages ----
if page == "Info":
    st.header("Global Air Quality Dashboard 🌍")
    st.markdown("""
**What is Air Quality?**  
Air quality refers to the state of the air around us, specifically the presence of pollutants that can affect human health, ecosystems, and climate. Key pollutants monitored globally include **PM2.5, PM10, and NO₂**.  

**Why it matters:**  
- Poor air quality contributes to **respiratory and cardiovascular diseases**, reduced life expectancy, and premature deaths.  
- It impacts vulnerable populations such as children, the elderly, and people with pre-existing conditions.  
- Monitoring air quality is crucial for **policy-making, urban planning, and public health awareness**.  

**About this Project:**  
This dashboard visualizes global air quality data, calculates **Air Quality Index (AQI)** based on pollutant-specific values using WHO guidelines, and provides interactive tools to explore trends by country, year, and pollutant.  

**Data Source & References:**  
- [WHO — Air Quality, Energy and Health](https://www.who.int/teams/environment-climate-change-and-health/air-quality-energy-and-health)  
- [WHO — Ambient (outdoor) air quality and health (fact sheet)](https://www.who.int/news-room/fact-sheets/detail/ambient-%28outdoor%29-air-quality-and-health)  

**Health Impacts of Air Pollution (WHO):**  
- Long-term exposure to PM2.5 and PM10 can cause **lung cancer, heart disease, and stroke**.  
- NO₂ contributes to **asthma and other respiratory disorders**.  
- Air pollution is **one of the leading environmental risk factors for global mortality**.
""")

elif page == "Background":
    st.header("Background — Dataset and Imputation")
    if raw_df.empty:
        st.info("No data yet. Press Load / Refresh Data.")
    else:
        st.subheader("Dataset Overview")
        st.write("Sample of the air quality dataset:")
        st.dataframe(raw_df.head())

        st.markdown("""
**Column Descriptions:**  
- `country_name` – Name of the country  
- `city` – Monitoring site or city  
- `year` – Reporting year  
- `pm25_concentration`, `pm10_concentration`, `no2_concentration` – Pollutant concentrations in µg/m³  
- `population` – City-level population estimate  

**Missing Data:**  
- The dataset contains missing values, primarily in pollutant measurements and population.  
- Missingness type: **MCAR (Missing Completely at Random)** — the missingness is unrelated to other variables or the values themselves.  
- Handling missing data: **Iterative Imputer (MICE)** predicts missing values using other available variables, preserving relationships and patterns in the dataset better than simple mean or median imputation.

**Visualizing Missingness:**  
- Yellow = missing, Purple = present
""")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        sns.heatmap(raw_df.isnull(), cbar=False, cmap='viridis', ax=ax1)
        ax1.set_title("Missingness (Before Cleaning)")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', ax=ax2)
        ax2.set_title("After Iterative Imputer (MICE)")
        st.pyplot(fig2)

        st.markdown("### Correlation Matrix (After Imputation)")
        corr = df.select_dtypes(include='number').corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation of Numeric Columns")
        st.plotly_chart(fig_corr, use_container_width=True)


elif page == "Dashboard":
    st.header("Dashboard — Interactive Visuals")
    if df.empty:
        st.info("No data yet. Press Load / Refresh Data.")
    else:
        y_min, y_max = int(df['year'].min()), int(df['year'].max())
        sel_year = st.sidebar.slider("Year", y_min, y_max, y_max, step=1)
        countries = sorted(df['country_name'].dropna().unique())
        sel_country = st.sidebar.selectbox("Country 1", countries)
        compare_mode = st.sidebar.checkbox("Compare with another country")
        if compare_mode:
            sel_country2 = st.sidebar.selectbox("Country 2", countries, index=min(1, len(countries)-1))
        else:
            sel_country2 = None

        st.subheader("Global AQI Map")
        map_df = country_aqi[country_aqi['year']==sel_year]
        fig_map = px.choropleth(map_df, locations='iso3', color='country_aqi_mean', hover_name='country_name',
                                color_continuous_scale='RdYlBu_r', labels={'country_aqi_mean':'AQI'})
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")
        st.subheader("Country-wise AQI Trends")
        if compare_mode and sel_country2:
            ts1 = df[df['country_name']==sel_country].groupby('year',as_index=False).agg(aqi=('aqi_dominant','mean'))
            ts2 = df[df['country_name']==sel_country2].groupby('year',as_index=False).agg(aqi=('aqi_dominant','mean'))
            fig_cmp = px.line(ts1, x='year', y='aqi', markers=True, title="AQI Comparison")
            fig_cmp.add_scatter(x=ts2['year'], y=ts2['aqi'], mode='lines+markers', name=sel_country2)
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            ts = df[df['country_name']==sel_country].groupby('year',as_index=False).agg(aqi=('aqi_dominant','mean'))
            fig_l = px.line(ts, x='year', y='aqi', markers=True, title=f"{sel_country} AQI Trend")
            st.plotly_chart(fig_l, use_container_width=True)

elif page == "Export":
    st.header("Export Cleaned Dataset")
    if df.empty:
        st.info("No data yet. Press Load / Refresh Data.")
    else:
        export_df = df.drop(columns=['latitude','longitude'], errors='ignore').copy()
        export_df['AQI'] = export_df['aqi_dominant']
        csv_buf = io.StringIO()
        export_df.to_csv(csv_buf, index=False)
        st.download_button("Download Cleaned CSV", data=csv_buf.getvalue(), file_name="Cleaned_AirQuality_Data.csv", mime="text/csv")
        st.write("✅ File ready for download. Contains AQI values and essential columns.")

elif page == "Contact":
    st.header("Contact")
    st.markdown("**Developer:** Jampala, Harshitha  \n**Email:** jampalah@msu.edu")
    st.markdown("This app is intended for exploratory analysis of global air quality data.")
