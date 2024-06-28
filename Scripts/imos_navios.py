from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

import pandas as pd
import csv

csv_imos = '../Datasets/Imos.csv'


def extract_ship_info(imo_numbers):
    # Initialize the Chrome driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    # Initialize an empty dataframe
    df = pd.DataFrame()

    for imo_number in imo_numbers:
        # Access the page
        url = f"https://www.balticshipping.com/vessel/imo/{imo_number}"
        driver.get(url)

        # Get the complete HTML content of the page
        html_content = driver.page_source

        # Create a BeautifulSoup object to parse the HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the div with the class "ship-info-container"
        ship_info_container = soup.find("div", {"class": "ship-info-container", "style": "position: relative;"})

        if ship_info_container:
            # Find the table within the div
            ship_info_table = ship_info_container.find("table", {"class": "table ship-info", "style": "min-height: 710px;"})

            if ship_info_table:
                # Extract the data from the table
                data = {}
                rows = ship_info_table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    headers = row.find_all("th")
                    if headers and cells:
                        data[headers[0].text.strip()] = [cell.text.strip() for cell in cells]

                # Convert the data to a dataframe
                new_df = pd.DataFrame(data)

                # Concatenate the new dataframe with the main dataframe
                df = pd.concat([df, new_df], ignore_index=True)

            else:
                print(f"Could not find the ship information table for IMO {imo_number}")
        else:
            print(f"Could not find the ship information section for IMO {imo_number}")

    # Close the browser
    driver.quit()

    # Return the dataframe
    return df

def open_csv_imos(csv_imos):
    #abrir arquivo csv com os imo numbers imos.csv, para cara valor da coluna Imos, amarzenar em uma lista
    with open('Imos.csv', 'r') as file:
        reader = csv.DictReader(file)
        imo_numbers = [row['Imo'] for row in reader]
        
    imo_numbers = list(set(imo_numbers))
    #contar quantos valores tem em imo_numbers
    print(f"Total IMOs: {len(imo_numbers)}")
    print(imo_numbers[0:10])

    return imo_numbers

df_imos_caracteristicas = extract_ship_info(open_csv_imos(csv_imos))

