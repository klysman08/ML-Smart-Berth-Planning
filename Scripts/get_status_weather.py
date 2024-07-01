import pandas as pd
import requests
from datetime import datetime, timedelta

def get_weather_data(date_list, location="New York, US"):
    """
    Retrieves weather data for a list of dates and a given location.
    
    Parameters:
    date_list (list): A list of dates in the format 'YYYY-MM-DD'.
    location (str, optional): The location to retrieve weather data for. Defaults to 'New York, US'.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the weather data for each date.
    """
    weather_data = []
    errors = []
    
    # Loop through the list of dates
    for date_str in date_list:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Construct the API request URL
        url = f'http://api.openweathermap.org/data/2.5/weather?q={location}&dt={int(date.timestamp())}&appid=4fbaedfb6bc3921e63e511e1b5308eb9&units=metric'
        
        # Make the API request
        try:
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Extract the relevant weather data
                weather_info = {
                    'date': date_str,
                    'temperature': data['main']['temp'],
                    'weather': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed'],
                    'humidity': data['main']['humidity']
                }
                weather_data.append(weather_info)
            else:
                errors.append(f'Error retrieving weather data for {date_str}: {response.status_code} - {response.text}')
        except requests.exceptions.RequestException as e:
            errors.append(f'Error retrieving weather data for {date_str}: {e}')
    
    # Create a pandas DataFrame from the weather data
    weather_df = pd.DataFrame(weather_data)
    
    # Print any error messages
    if errors:
        print('Errors occurred while retrieving weather data:')
        for error in errors:
            print(error)
    
    return weather_df

#lista for Daily Forecast 16 days
date_list = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(16)]
weather_df = get_weather_data(date_list, location="London, UK")
print(weather_df)

# Save the weather data to a CSV file
weather_df.to_csv('weather_data.csv', index=False)
