# Global Air Quality Dashboard 🌍

An **interactive Streamlit dashboard** to explore global air quality data based on WHO measurements of PM2.5, PM10, and NO₂. The app visualizes trends, computes the Air Quality Index (AQI), and provides interactive maps, time series, and export functionality.

---

## Features

- **Global AQI Map:** Visualize country-level AQI on a choropleth map.  
- **Country Trends:** Interactive line charts showing AQI over years.  
- **Top Cities:** Pie charts highlighting cities with highest AQI.  
- **Missing Data Handling:** Iterative Imputer (MICE) fills missing pollutant and population data.  
- **Export Cleaned Data:** Download cleaned dataset with AQI values.  
- **Compare Countries:** Compare AQI trends between two countries.

---

## Data Source

Data comes from the **[WHO Ambient Air Quality Database](https://www.who.int/teams/environment-climate-change-and-health/air-quality-energy-and-health)**. The dashboard calculates AQI using pollutant-specific breakpoints and the dominant pollutant.

---
## Usage

1. Clone the repo.
2. Install dependencies using the Requirement.txt file.
3. Run the streamlit app. 

## Streamlit app link 
https://cmse830fds-qtaemuw9rz9aye54szenjr.streamlit.app/ 
