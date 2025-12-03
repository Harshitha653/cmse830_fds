# app.py
import streamlit as st
import pandas as pd
from feature_engineering import preprocess_tweets, preprocess_news, preprocess_stocks, vectorize_text
from visualizations import plot_stock_trends, plot_sentiment_distribution, plot_news_sentiment, plot_scatter_sentiment_vs_stock, plot_wordcloud
from utils import load_model, predict_sentiment
import os

st.set_page_config(page_title="Market & Sentiment Dashboard", layout="wide")

# ------------------------------
# 1. Load Data with Caching
# ------------------------------
@st.cache_data
def load_data():
    # Load CSVs
    df_stocks = pd.read_csv("data/all_stocks.csv", delimiter=",")
    df_tweets = pd.read_csv("data/Tweets.csv", sep=None, engine="python")
    df_news = pd.read_csv("data/abcnews-date-text.csv", delimiter=',')
    
    # Preprocess
    df_stocks = preprocess_stocks(df_stocks)
    df_tweets = preprocess_tweets(df_tweets)
    df_tweets['sentiment_label'] = df_tweets['airline_sentiment'].map({
    'positive': 1,
    'neutral': 0,
    'negative': -1
    })

    df_news = preprocess_news(df_news)
    
    return df_stocks, df_tweets, df_news

df_stocks, df_tweets, df_news = load_data()

# ------------------------------
# 2. Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Stocks", "Tweets", "News", "Sentiment Prediction"])

# ------------------------------
# 3. Overview Page
# ------------------------------
if page == "Overview":
    st.title("Market & Sentiment Dashboard")
    st.markdown("""
    This dashboard integrates **stock market data**, **tweets about airlines**, 
    and **news headlines** for analysis and visualization.
    """)
    
    st.subheader("Dataset Summary")
    st.write("Stocks Dataset")
    st.dataframe(df_stocks.head())
    st.write("Tweets Dataset")
    st.dataframe(df_tweets.head())
    st.write("News Dataset")
    st.dataframe(df_news.head())

# ------------------------------
# 4. Stocks Page
# ------------------------------
elif page == "Stocks":
    st.title("Stock Analysis")
    ticker_list = df_stocks['Ticker'].unique().tolist()
    selected_ticker = st.selectbox("Select Ticker", ticker_list)
    
    plot_stock_trends(df_stocks, selected_ticker)
    st.subheader("Stock Statistics")
    df_selected = df_stocks[df_stocks['Ticker']==selected_ticker]
    st.write(df_selected.describe())
    
    st.subheader("Stock Volatility & Returns")
    cols_to_plot = [c for c in ["Daily_Return", "Volatility"] if c in df_selected.columns]

    if cols_to_plot:
    	st.line_chart(df_selected[cols_to_plot])
    else:
    	st.warning("Daily_Return and Volatility not found in selection.")


# ------------------------------
# 5. Tweets Page
# ------------------------------
elif page == "Tweets":
    st.title("Tweets Analysis")
    st.subheader("Sentiment Distribution")
    plot_sentiment_distribution(df_tweets)
    
    st.subheader("Word Cloud of Tweets")
    plot_wordcloud(df_tweets, 'text_clean')
    
    st.subheader("Sample Tweets")
    st.dataframe(df_tweets[['text','airline_sentiment','sentiment_label']].head(20))

# ------------------------------
# 6. News Page
# ------------------------------
elif page == "News":
    st.title("News Analysis")
    st.subheader("News Count Over Time")
    plot_news_sentiment(df_news)
    
    st.subheader("Word Cloud of News Headlines")
    plot_wordcloud(df_news, 'headline_text_clean')
    
    st.subheader("Sample News Headlines")
    st.dataframe(df_news.head(20))

# ------------------------------
# 7. Sentiment Prediction Page
# ------------------------------
elif page == "Sentiment Prediction":
    st.title("Predict Tweet Sentiment")
    st.markdown("Enter a tweet to predict its sentiment (Positive=1, Neutral=0, Negative=-1)")

    input_text = st.text_area("Enter Tweet Here:")

    # Load model bundle once
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = load_model("ensemble_model.pkl")

    model = st.session_state.model_bundle["model"]
    vectorizer = st.session_state.model_bundle["vectorizer"]

    if st.button("Predict Sentiment"):
        if input_text.strip() == "":
            st.warning("Please enter a tweet to predict.")
        else:
            pred = predict_sentiment(model, vectorizer, input_text)
            sentiment_dict = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            st.success(f"Predicted Sentiment: {sentiment_dict[pred]}")

