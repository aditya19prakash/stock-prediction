import wikipediaapi
import re
import streamlit as st
import yahooquery as yq
import yfinance as yf
import pandas as pd
from utils import check_internet_connection, format_number
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import logging
import os
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'error_log.log'), level=logging.ERROR, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def get_symbol_from_name(company_name):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return None, 'N/A'
        search_result = yq.search(company_name)
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote and (quote['symbol'].endswith(".NS")):
                    return quote['symbol'], quote.get('shortname', 'N/A')
        st.error("This stock is not listed in NSE(National Stock Exchange).")
        return -1, 'N/A'
    except Exception as e:
        st.error(f"Error: {e}")
        logging.error(f"Error in get_symbol_from_name: {e}")
        return None, 'N/A'

def sip_calculator(symbol):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return
        st.subheader("SIP Calculator")
        start_date = "2023-01-01"
        end_date = "2024-11-30"
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.warning("No data available for the selected symbol and date range.")
            return
        sip_data = stock_data.iloc[-252:]

        if 'monthly_investment' not in st.session_state:
            st.session_state['monthly_investment'] = 1000

        monthly_investment = st.slider(
            "Monthly Investment Amount (₹)",
            min_value=500,
            max_value=10000,
            step=500,
            key='monthly_investment'
        )

        initial_price = sip_data['Close'].iloc[0]
        final_price = sip_data['Close'].iloc[-1]
        annual_return = ((final_price - initial_price) / initial_price) * 100

        if isinstance(annual_return, pd.Series):
            annual_return = annual_return.values[0]

        years = 3
        p = float(monthly_investment)
        r = float(annual_return)
        t = float(years)
        r = r / (12 * 100)
        n = t * 12

        fv = p * (((1 + r)**n - 1) / r) * (1 + r)
        
        formatted_Return = format_number(fv)
        st.markdown(
            f"<div style='font-size: 22px; color: white; background-color: #0e1117; padding: 10px; border-radius: 5px; border: 2px solid white; display: flex; flex-wrap: wrap;'>"
            f"Monthly Investment Amount (₹): {monthly_investment}<br>Estimated Annual Return: {annual_return:.2f}%<br>Future Value after {years} years based on predicted prices: ₹{formatted_Return}"
            f"</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error in SIP Calculator: {e}")
        logging.error(f"Error in sip_calculator: {e}")

def get_additional_stock_info(symbol):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return {}
        stock = yq.Ticker(symbol)
        info = stock.asset_profile[symbol]
        summary_detail = stock.summary_detail[symbol]
        return {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'fullTimeEmployees': info.get('fullTimeEmployees', 'N/A'),
            'city': info.get('city', 'N/A'),
            'state': info.get('state', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'previousClose': summary_detail.get('previousClose', 'N/A'),
            'open': summary_detail.get('open', 'N/A'),
            'bid': summary_detail.get('bid', 'N/A'),
            'ask': summary_detail.get('ask', 'N/A'),
            'daysRange': str(summary_detail.get('dayLow', 'N/A')) + ' - ' + str(summary_detail.get('dayHigh', 'N/A')),
            'fiftyTwoWeekRange': str(summary_detail.get('fiftyTwoWeekLow', 'N/A')) + ' - ' + str(summary_detail.get('fiftyTwoWeekHigh', 'N/A')),
            'volume': summary_detail.get('volume', 'N/A'),
            'avgVolume': summary_detail.get('averageVolume', 'N/A'),
            'marketCap': summary_detail.get('marketCap', 'N/A'),
            'beta': summary_detail.get('beta', 'N/A'),
            'peRatio': summary_detail.get('trailingPE', 'N/A'),
            'eps': summary_detail.get('trailingEps', 'N/A'),
            'earningsDate': summary_detail.get('earningsDate', 'N/A'),
            'forwardDividendYield': summary_detail.get('dividendYield', 'N/A'),
            'exDividendDate': summary_detail.get('exDividendDate', 'N/A'),
            'oneYearTargetEst': summary_detail.get('targetMeanPrice', 'N/A')
        }
    except Exception as e:
        st.error(f"Error fetching additional stock info: {e}")
        logging.error(f"Error in get_additional_stock_info: {e}")
        return {}

def display_additional_stock_info(info):
    try:
        def format_market_cap(market_cap):
            if market_cap == 'N/A':
                return market_cap
            market_cap = float(market_cap)
            if market_cap >= 1e12:
                return f"{market_cap / 1e12:.2f} Trillion Rs"
            elif market_cap >= 1e9:
                return f"{market_cap / 1e9:.2f} Billion Rs"
            elif market_cap >= 1e6:
                return f"{market_cap / 1e6:.2f} Million Rs"
            else:
                return f"{market_cap:.2f} Rs"

        st.markdown("<h2 style='font-size: 24px; color: white;'>Additional Stock Information:</h2>", unsafe_allow_html=True)
        st.markdown(
                    f"<div style='font-size: 22px; color: white; background-color: #0e1117; padding: 10px; border-radius: 5px; border: 2px solid white; display: flex; flex-wrap: wrap;'>"
                    f"<div style='flex: 1; min-width: 300px;'><b>Sector:</b> {info.get('sector', 'N/A')}<br>"
                    f"<b>Industry:</b> {info.get('industry', 'N/A')}<br>"
                    f"<b>Full-Time Employees:</b> {info.get('fullTimeEmployees', 'N/A')}<br>"
                    f"<b>City:</b> {info.get('city', 'N/A')}<br>"
                    f"<b>Website:</b> <a href='{info.get('website', 'N/A')}' target='_blank'>{info.get('website', 'N/A')}</a><br>"
                    f"<b>Previous Close:</b> {info.get('previousClose', 'N/A')}<br>"
                    f"<b>Open:</b> {info.get('open', 'N/A')}<br>"
                    f"<b>Bid:</b> {info.get('bid', 'N/A')}<br>"
                    f"<b>Ask:</b> {info.get('ask', 'N/A')}<br>"
                    f"<b>Day's Range:</b> {info.get('daysRange', 'N/A')}<br>"
                    f"<b>52 Week Range:</b> {info.get('fiftyTwoWeekRange', 'N/A')}</div>"
                    f"<div style='flex: 1; min-width: 300px;'><b>Volume:</b> {info.get('volume', 'N/A')}<br>"
                    f"<b>Avg. Volume:</b> {info.get('avgVolume', 'N/A')}<br>"
                    f"<b>Market Cap:</b> {format_market_cap(info.get('marketCap', 'N/A'))}<br>"
                    f"<b>Beta:</b> {info.get('beta', 'N/A')}<br>"
                    f"<b>PE Ratio:</b> {info.get('peRatio', 'N/A')}<br>"
                    f"<b>EPS:</b> {info.get('eps', 'N/A')}<br>"
                    f"<b>Earnings Date:</b> {info.get('earningsDate', 'N/A')}<br>"
                    f"<b>Forward Dividend & Yield:</b> {info.get('forwardDividendYield', 'N/A')}<br>"
                    f"<b>Ex-Dividend Date:</b> {info.get('exDividendDate', 'N/A')}<br>"
                    f"<b>1y Target Est:</b> {info.get('oneYearTargetEst', 'N/A')}</div>"
                    f"</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying additional stock info: {e}")
        logging.error(f"Error in display_additional_stock_info: {e}")

def get_wikipedia_summary(full_name):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="stock-prediction/1.0 (adyprakash19@gmail.com)"
        )
        page = wiki_wiki.page(full_name)
        if not page.exists():
            return "No Summary found for the company name."
        summary = page.summary[:60000] 
        summary = re.sub(r'\s+', ' ', summary).strip() 
        summary = re.sub(r'==\s*History\s*==', '', summary)
        summary = re.sub(r'===\s*.*?\s*===', '', summary) 
        summary = re.sub(r'\bnudity\b|\bporn\b|\bsex\b|\badult\b', '', summary, flags=re.IGNORECASE)
        return summary
    except Exception as e:
        logging.error(f"Error in get_wikipedia_summary: {e}")
        return f"Error: {e}"

def get_company_news(company_name):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return
        st.markdown("<h2 style='font-size: 24px; color: white;'>Latest News:</h2>", 
                   unsafe_allow_html=True)
                   
        if 'news_cache' not in st.session_state:
            st.session_state['news_cache'] = {}

        if company_name in st.session_state['news_cache']:
            news_items = st.session_state['news_cache'][company_name]
        else:
            query = f"{company_name} stock news"
            news_items = []

            for url in search(query, num_results=5):
                try:
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('meta', property='og:title') or soup.find('title')
                    title = title.get('content', '') if title else ''
                    if title and url:
                        news_items.append({
                            'title': title,
                            'url': url
                        })
                except Exception as e:
                    logging.error(f"Error fetching news from {url}: {e}")
                    continue

            st.session_state['news_cache'][company_name] = news_items

        for item in news_items:
            st.markdown(
                f"<div style='font-size: 18px; color: white; "
                f"background-color: #0e1117; padding: 10px; "
                f"border-radius: 5px; border: 2px solid white; margin-bottom: 10px;'>"
                f"<a href='{item['url']}' target='_blank' style='color: #00BFFF;'>"
                f"{item['title']}</a>"
                f"</div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error fetching news: {e}")
        logging.error(f"Error in get_company_news: {e}")

""" MAIN FUNCTION  """
def summaryprint(company_name, combined_predictions, symbol, signal):
    try:
        if not check_internet_connection():
            st.error("No internet connection. Please check your connection and try again.")
            return
        if 'summary_cache' not in st.session_state:
            st.session_state['summary_cache'] = {}

        if company_name in st.session_state['summary_cache']:
            cached_data = st.session_state['summary_cache'][company_name]
            combined_predictions = cached_data['combined_predictions']
            additional_info = cached_data['additional_info']
            summary = cached_data['summary']
        else:
            additional_info = get_additional_stock_info(symbol)
            summary = get_wikipedia_summary(additional_info.get('shortname', company_name))
            st.session_state['summary_cache'][company_name] = {
                'combined_predictions': combined_predictions,
                'additional_info': additional_info,
                'summary': summary
            }
        for days in [30, 60, 90, 120]:
            if len(combined_predictions) > days:
              initial_price = combined_predictions[0]
              future_price = combined_predictions[days]
              percent_change = ((future_price - initial_price) / initial_price) * 100
              color = "#39FF14" if percent_change > 0 else "#FF073A"
              icon = "📉" if percent_change < 0 else "📈"
              st.markdown(
                  f"<div style='font-size: 22px; color: white; background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid white; margin-bottom: 10px;'>"
                  f"<b>After {days} trading days, the stock listing price will be:</b> <span style='color: #FFFF33;'>₹{int(future_price)}</span><br>"
                  f"<b>{icon} Percentage Change:</b> <span style='color: {color};'>{percent_change:.2f}%</span>"
                  f"</div>", 
                  unsafe_allow_html=True
              )

        if signal:
            sip_calculator(symbol)
        
        display_additional_stock_info(additional_info)
        get_company_news(company_name)
        st.markdown("<h2 style='font-size: 24px; color: white;'>Company Summary:</h2>", unsafe_allow_html=True)
        st.markdown( f"<div style='font-size: 18px; color: white; background-color: #0e1117; padding: 10px; border-radius: 5px; border: 2px solid white;'>{summary}</div>",
        unsafe_allow_html=True)
    
        st.markdown(
            "<div style='font-size: 18px; color: red; background-color: #0e1117; padding: 10px; border-radius: 5px; border: 2px solid red; margin-top: 20px;'>"
            "<b>⚠️ Disclaimer:</b> Stock market investments carry risks. Past performance does not guarantee future returns. "
            "Please conduct thorough research and consider consulting with a financial advisor before making investment decisions."
            "</div>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Error in summaryprint: {e}")
        logging.error(f"Error in summaryprint: {e}")