import wikipediaapi
import re
import streamlit as st
import yahooquery as yq

def get_symbol_from_name(company_name):
    try:
        search_result = yq.search(company_name)
        if 'quotes' in search_result:
            for quote in search_result['quotes']:
                if 'symbol' in quote and (quote['symbol'].endswith(".NS")):
                    return quote['symbol'], quote.get('shortname', 'N/A')
        st.error("This stock is not listed in NSE(National Stock Exchange).")
        return -1, 'N/A'
    except Exception as e:
        st.error(f"Error: {e}")
        return None, 'N/A'

def get_additional_stock_info(symbol):
    try:
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
        return {}

def display_additional_stock_info(info):
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

def get_wikipedia_summary(full_name):
    try:
        print(f"Searching Wikipedia for: {full_name}")
        
        # Set a custom user agent
        wiki_wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="stock-prediction/1.0 (adyprakash19@gmail.com)"
        )
        
        page = wiki_wiki.page(full_name)

        if not page.exists():
            return "No Wikipedia page found for the company name."

        # Get the summary of the page
        summary = page.summary[:60000]  # Limit to 2000 characters

        # Clean up the summary
        summary = re.sub(r'\s+', ' ', summary).strip()  # Remove extra whitespaces
        summary = re.sub(r'==\s*History\s*==', '', summary)  # Remove 'History' section
        summary = re.sub(r'===\s*.*?\s*===', '', summary)  # Remove subheadings
        summary = re.sub(r'\bnudity\b|\bporn\b|\bsex\b|\badult\b', '', summary, flags=re.IGNORECASE)  # Remove adult content terms
        return summary
    except Exception as e:
        return f"Error: {e}"

""" MAIN FUNCTION  """
def summaryprint(company_name, combined_predictions, symbol):
    for days in [30, 60, 90, 120]:
        if len(combined_predictions) > days:
            st.markdown(
                f"<div style='font-size: 20px; color: white; background-color: #0e1117; padding: 15px; border-radius: 10px; border: 2px solid white; margin-bottom: 10px;'>"
                f"<b>After {days} days, the stock listing price will be:</b> <span style='color: #FFD700;'>[ {int(combined_predictions[days])} ]</span>"
                f"</div>", 
                unsafe_allow_html=True
            )
    additional_info = get_additional_stock_info(symbol)
    display_additional_stock_info(additional_info)
    summary = get_wikipedia_summary(additional_info.get('shortname', company_name))
    if True:
        st.markdown("<h2 style='font-size: 24px; color: white;'>Company Summary:</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size: 18px; color: white; background-color: #0e1117; padding: 10px; border-radius: 5px; border: 2px solid white;'>{summary}</div>",
            unsafe_allow_html=True
        )