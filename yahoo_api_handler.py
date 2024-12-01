import requests  # assuming you're using requests

def get_symbol_from_name(company_name):
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    params = {'q': company_name, 'quotes_count': 1}
    
    try:
        response = requests.get(url, params=params)
        print(f"Response Status: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200 and response.text:
            data = response.json()  # assuming the response is JSON
            if 'quotes' in data and data['quotes']:
                symbol = data['quotes'][0]['symbol']
                exchange = data['quotes'][0]['exchange']
                return symbol, exchange
            else:
                print(f"No symbol found for {company_name}")
                return None, None
        else:
            print("Failed to retrieve data or empty response")
            return None, None
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None, None
