
import requests

def get_location():
    try:
        response = requests.get("https://ipinfo.io/")
        data = response.json()
        location_url = f"https://www.google.com/maps?q={data['loc']}"
        return location_url
    except Exception as e:
        return "Location not available"
