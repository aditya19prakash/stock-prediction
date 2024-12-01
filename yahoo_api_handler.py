import requests

def get_symbol_from_name(company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotes_count": 5, "news_count": 0}
    
    try:
        response = _make_request(url, params=params)
        
        if not response.text:
            print("Received an empty response")
            return None, None
        
        data = response.json()
        if 'quotes' in data and data['quotes']:
            symbol = data['quotes'][0]['symbol']
            exchange = data['quotes'][0]['exchange']
            return symbol, exchange
        else:
            print("No quotes found for the given company name.")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None, None
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response")
        return None, None

def _make_request(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
