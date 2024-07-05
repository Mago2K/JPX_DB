import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import requests
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from bs4 import BeautifulSoup

# Set page title and layout
st.set_page_config(page_title='Financial Dashboard', layout='wide')

# Apply custom CSS for Bloomberg Terminal look
st.markdown("""
    <style>
    body {
        color: #f0f2f6;
        background-color: #121212;
    }
    .main {
        background-color: #121212;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        background-color: #333;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
    }
    .stFileUploader>div>div>div>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        color: white;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 20px;
    }
    .css-1avcm0n {
        color: #4CAF50;
    }
    .css-15tx938 {
        color: white;
    }
    .css-1kyxreq {
        font-family: Arial, Helvetica, sans-serif;
    }
    .css-1adrfps {
        color: #4CAF50;
    }
    .css-2trqyj {
        color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Mapping of company codes to Yahoo Finance tickers
ticker_mapping = {
    'SHIZUOKA FINANCIAL GROUP': '5831.T',
    'CONCORDIA FINANCIAL GROUP': '7186.T',
    'AOZORA BANK': '8304.T',
    'MITSUBISHI UFJ FINANCIAL': '8306.T',
    'RESONA HOLDINGS': '8308.T',
    'SUMITOMO MITSUI TRUST': '8309.T',
    'SUMITOMO MITSUI FINANCIAL': '8316.T',
    'THE CHIBA BANK': '8331.T',
    'FUKUOKA FINANCIAL': '8354.T',
    'MIZUHO FINANCIAL': '8411.T',
    'SHINSEI BANK': '8473.T',
}

# Tickers for analysis
tickers = [
    'SHIZUOKA FINANCIAL GROUP', 'CONCORDIA FINANCIAL GROUP', 'AOZORA BANK',
    'MITSUBISHI UFJ FINANCIAL', 'RESONA HOLDINGS', 'SUMITOMO MITSUI TRUST',
    'SUMITOMO MITSUI FINANCIAL', 'THE CHIBA BANK', 'FUKUOKA FINANCIAL',
    'MIZUHO FINANCIAL', 'SPXT', 'GOVT', 'IWM', 'SHINSEI BANK'
]

# Define a mapping for units
unit_mapping = {
    '百万円': 'Million Yen',
    '円': 'Yen',
    '％': '%',
    '倍': 'Times'
}

# Fetch company news from Yahoo Finance
def fetch_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []
    for item in soup.find_all('li', class_='js-stream-content'):
        title = item.find('h3').text if item.find('h3') else None
        link = item.find('a', href=True)['href']
        link = f"https://finance.yahoo.com{link}"
        summary = item.find('p').text if item.find('p') else None
        if title and link:
            articles.append({'title': title, 'link': link, 'summary': summary})
    return articles

# File uploader
data_path = 'data/Banking data.csv'

if data_path is not None:
    # Load data
    data = pd.read_csv(data_path)

    # Translate units
    data['unit'] = data['unit'].map(unit_mapping)

    # Define years variable
    years = ['2018 FY', '2019 FY', '2020 FY', '2021 FY', '2022 FY']

    # Bank selection
    banks = data['company_name'].unique()
    selected_bank = st.selectbox('Select a bank', banks)

    if selected_bank in ticker_mapping:
        ticker = ticker_mapping[selected_bank]
        stock_data = yf.Ticker(ticker)

        # Fetch stock information
        stock_info = stock_data.info

        # Dropdown menu for selecting time frame
        time_frame = st.selectbox('Select time frame', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
        hist_data = stock_data.history(period=time_frame)

        # Add indicators
        add_sma20 = st.checkbox('Show 20-Day SMA', value=True)
        add_sma50 = st.checkbox('Show 50-Day SMA', value=True)
        add_sma200 = st.checkbox('Show 200-Day SMA', value=True)
        add_bollinger = st.checkbox('Show Bollinger Bands')

        # Calculate moving averages
        if add_sma20:
            hist_data['SMA20'] = hist_data['Close'].rolling(window=20).mean()
        if add_sma50:
            hist_data['SMA50'] = hist_data['Close'].rolling(window=50).mean()
        if add_sma200:
            hist_data['SMA200'] = hist_data['Close'].rolling(window=200).mean()
        if add_bollinger:
            hist_data['BB_upper'] = hist_data['Close'].rolling(window=20).mean() + (hist_data['Close'].rolling(window=20).std() * 2)
            hist_data['BB_lower'] = hist_data['Close'].rolling(window=20).mean() - (hist_data['Close'].rolling(window=20).std() * 2)

        # Display Highlights Section
        st.subheader('Highlights')
        col1, col3, col4 = st.columns(3)
        col1.metric(label="Market Cap", value=f"${stock_info.get('marketCap')/1e9:.2f}B")
        col3.metric(label="P/E Ratio", value=f"{stock_info.get('forwardPE'):.2f}")
        col4.metric(label="Dividend Yield", value=f"{stock_info.get('dividendYield')*100:.2f}%")

        # Display Overview Section
        st.subheader('Overview')
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write(f"**{selected_bank}**")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Industry:** {stock_info.get('industry')}")
            st.write(f"**Sector:** {stock_info.get('sector')}")
            st.write(f"**Website:** [Link]({stock_info.get('website')})")

        with col2:
            # Plot candlestick chart with plotly
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist_data.index,
                                         open=hist_data['Open'],
                                         high=hist_data['High'],
                                         low=hist_data['Low'],
                                         close=hist_data['Close'],
                                         name='Candlesticks'))

            # Add volume trace
            fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'], name='Volume', yaxis='y2', opacity=0.3))

            # Add moving average traces
            if add_sma20:
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA20'], mode='lines', name='20-Day SMA'))
            if add_sma50:
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA50'], mode='lines', name='50-Day SMA'))
            if add_sma200:
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['SMA200'], mode='lines', name='200-Day SMA'))
            if add_bollinger:
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['BB_upper'], mode='lines', name='Bollinger Upper'))
                fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['BB_lower'], mode='lines', name='Bollinger Lower'))

            # Update layout for better visualization
            fig.update_layout(
                title=f"{selected_bank} Stock Price",
                yaxis_title='Stock Price (JPY)',
                xaxis_title='Date',
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig)

        # Display General Information Section
        st.subheader('General Information')
        st.write(f"**Description:** {stock_info.get('longBusinessSummary')}")
        st.write(f"**Primary Office:** {stock_info.get('address1')}, {stock_info.get('address2')}, {stock_info.get('city')}, {stock_info.get('zip')}, {stock_info.get('country')}")
        st.write(f"**Phone:** {stock_info.get('phone')}")
        st.write(f"**Website:** [Link]({stock_info.get('website')})")
        st.write(f"**Industry:** {stock_info.get('industry')}")
        st.write(f"**Sector:** {stock_info.get('sector')}")
        st.write(f"**Employees:** {stock_info.get('fullTimeEmployees')}")
        st.write(f"**Year Founded:** {stock_info.get('founded')}")

        # Display Industry, Verticals & Keywords
        st.subheader('Industries, Verticals & Keywords')
        st.write(f"**Primary Industry:** {stock_info.get('industry')}")
        st.write(f"**Sector:** {stock_info.get('sector')}")

        # Fetch and display news
        st.subheader('Latest News')
        news_articles = fetch_news(ticker)
        if news_articles:
            for article in news_articles:
                st.write(f"**{article['title']}**")
                st.write(article.get('summary', ''))
                st.write(f"[Read more]({article['link']})")
        else:
            st.write("No news articles found.")

        # Peer comparison
        st.subheader('Peer Comparison')
        peers = st.multiselect('Select peers to compare', banks)
        peer_data = []
        for peer in peers:
            if peer in ticker_mapping:
                peer_ticker = ticker_mapping[peer]
                peer_stock_data = yf.Ticker(peer_ticker)
                peer_info = peer_stock_data.info

                peer_metrics = {
                    'Company': peer,
                    'Market Cap': peer_info.get('marketCap'),
                    'P/E Ratio': peer_info.get('forwardPE'),
                    'EPS': peer_info.get('trailingEps'),
                    'Dividend Yield': peer_info.get('dividendYield'),
                    'Price to Book': peer_info.get('priceToBook'),
                    '52 Week High': peer_info.get('fiftyTwoWeekHigh'),
                    '52 Week Low': peer_info.get('fiftyTwoWeekLow')
                }
                peer_data.append(peer_metrics)

        if peer_data:
            peer_df = pd.DataFrame(peer_data)
            st.dataframe(peer_df)

    else:
        st.error('Ticker not found for the selected bank.')

    # Filter data for the selected bank
    bank_data = data[data['company_name'] == selected_bank]

    # Line chart for yearly financial metrics
    st.write('### Yearly Financial Metrics')
    metrics = bank_data['english_description'].unique()
    selected_metric = st.selectbox('Select a metric', metrics)
    chart_type = st.selectbox('Select chart type', ['Line Chart', 'Bar Chart'])

    if selected_metric:
        # Filter data for the selected metric
        metric_data = bank_data[bank_data['english_description'] == selected_metric][years].T
        metric_data.columns = [selected_metric]

        # Add unit information
        unit = bank_data[bank_data['english_description'] == selected_metric]['unit'].values[0]
        st.write(f'### Data (Unit: {unit})')
        st.write(metric_data)

        # Plot selected chart type if data is in correct shape
        if metric_data.shape[1] == 1:
            if chart_type == 'Bar Chart':
                fig = px.bar(metric_data, x=metric_data.index, y=selected_metric, title=f'Yearly Financial Metrics ({selected_metric})')
            else:
                fig = px.line(metric_data, x=metric_data.index, y=selected_metric, title=f'Yearly Financial Metrics ({selected_metric})')
            fig.update_layout(yaxis_title=f'Value ({unit})')
            st.plotly_chart(fig)
        else:
            st.error('Error in processing the data for the selected metric.')

    # Bar chart for yearly comparison
    st.write('### Yearly Comparison')
    if all(year in bank_data.columns for year in years):
        bar_data = bank_data[['english_description'] + years].set_index('english_description')
        bar_data.columns = [year.replace(' FY', '') for year in bar_data.columns]
        bar_data = bar_data.apply(pd.to_numeric, errors='coerce')

        # Plot bar chart
        fig = go.Figure()
        for column in bar_data.columns:
            fig.add_trace(go.Bar(x=bar_data.index, y=bar_data[column], name=column))

        fig.update_layout(barmode='group', title='Yearly Financial Metrics Comparison', yaxis_title=f'Value ({unit})')
        st.plotly_chart(fig)
    else:
        st.error('One or more expected year columns are missing in the uploaded data.')

#### MCALLAN STYLE PYTHON CODE

st.title('Financial Dashboard for Banks and Diversified Portfolio')

# Tickers for analysis
tickers = [
    'SHIZUOKA FINANCIAL GROUP', 'CONCORDIA FINANCIAL GROUP', 'AOZORA BANK',
    'MITSUBISHI UFJ FINANCIAL', 'RESONA HOLDINGS', 'SUMITOMO MITSUI TRUST',
    'SUMITOMO MITSUI FINANCIAL', 'THE CHIBA BANK', 'FUKUOKA FINANCIAL',
    'MIZUHO FINANCIAL', 'SPXT', 'GOVT', 'IWM', 'SHINSEI BANK'
]

# Mapping of company codes to Yahoo Finance tickers
ticker_mapping = {
    'SHIZUOKA FINANCIAL GROUP': '5831.T',
    'CONCORDIA FINANCIAL GROUP': '7186.T',
    'AOZORA BANK': '8304.T',
    'MITSUBISHI UFJ FINANCIAL': '8306.T',
    'RESONA HOLDINGS': '8308.T',
    'SUMITOMO MITSUI TRUST': '8309.T',
    'SUMITOMO MITSUI FINANCIAL': '8316.T',
    'THE CHIBA BANK': '8331.T',
    'FUKUOKA FINANCIAL': '8354.T',
    'MIZUHO FINANCIAL': '8411.T',
    'SPXT': 'SPXT',
    'GOVT': 'GOVT',
    'IWM': 'IWM',
    'SHINSEI BANK': '8473.T',
}

# Select the start and end dates
start_date = '2017-12-31'
end_date = '2023-12-31'

# Retrieve the prices, resample to yearly
yf_tickers = [ticker_mapping[ticker] for ticker in tickers]
df_prices = yf.download(yf_tickers, start=start_date, end=end_date)['Adj Close']
df_prices.index = pd.to_datetime(df_prices.index)

# Resample to yearly data and fill missing values
df_prices = df_prices.resample('YE').last().ffill().bfill()

# Convert to returns
df_returns = df_prices.pct_change().dropna()

# Create a diversified portfolio
portfolio = (df_returns['SPXT'].mul(0.6)) + (df_returns['GOVT'].mul(0.4))
portfolio.name = 'Portfolio'

# Add the balanced portfolio returns to the dataframe
df_returns_final = pd.concat([df_returns, portfolio], axis=1).mul(100).round(2)

# Function for plotting
def calendar_year_heatmap(df_returns):
    fig, ax = plt.subplots(figsize=(12, 8))

    unique_years = df_returns.index.year.unique()
    num_years = len(unique_years)

    tickers = df_returns.columns
    color_map = {ticker: plt.cm.tab20(i % len(plt.cm.tab20.colors)) for i, ticker in enumerate(tickers)}

    for i, year in enumerate(unique_years):
        df_year = df_returns[df_returns.index.year == year].T
        df_year_sorted = df_year.sort_values(by=df_year.columns[0], ascending=False)

        for j, (ticker, row) in enumerate(df_year_sorted.iterrows()):
            return_value = row.iloc[0]
            rect = Rectangle((i, j), 1, 1, facecolor=color_map[ticker], edgecolor='black')
            ax.add_patch(rect)
            ax.text(i + 0.5, j + 0.5, f'{ticker}\n{return_value:.2f}%',
                    va='center', ha='center', fontsize=8, color='black')

    ax.set_xlim(0, num_years)
    ax.set_ylim(0, len(tickers))
    ax.set_xticks([i + 0.5 for i in range(num_years)])
    ax.set_xticklabels(unique_years, rotation=45)
    ax.set_xlabel('Years')
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title('Class Performance by Calendar Year')

    plt.gca().invert_yaxis()
    st.pyplot(fig)


# Display the heatmap
calendar_year_heatmap(df_returns_final)

# Display the final returns dataframe
st.write('### Final Returns DataFrame')
st.dataframe(df_returns_final)