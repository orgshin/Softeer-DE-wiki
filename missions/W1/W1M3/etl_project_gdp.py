
import sqlite3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

# Constants
URL = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
LOG_FILE = "/Users/admin/data_engineering_course_materials/missions/W1/W1M3/etl_project_log.txt"
JSON_FILE = "/Users/admin/data_engineering_course_materials/missions/W1/W1M3/Countries_by_GDP.json"

# Because the main page does not have a GDP table for each region 
# So define a list of URL for each region
REGION_PAGES = {
    "Africa": "https://en.wikipedia.org/wiki/List_of_African_countries_by_GDP_(nominal)",
    "Arab League": "https://en.wikipedia.org/wiki/List_of_Arab_League_countries_by_GDP_(nominal)",
    "Asia-Pacific": "https://en.wikipedia.org/wiki/List_of_countries_in_Asia-Pacific_by_GDP_(nominal)",
    "Commonwealth": "https://en.wikipedia.org/wiki/List_of_Commonwealth_of_Nations_countries_by_GDP_(nominal)",
    "Latin America & Caribbean": "https://en.wikipedia.org/wiki/List_of_Latin_American_and_Caribbean_countries_by_GDP_(nominal)",
    "North America": "https://en.wikipedia.org/wiki/List_of_North_American_countries_by_GDP_(nominal)",
    "Oceania": "https://en.wikipedia.org/wiki/List_of_Oceanian_countries_by_GDP",
    "Europe": "https://en.wikipedia.org/wiki/List_of_sovereign_states_in_Europe_by_GDP_(nominal)"
}


def log_progress(message):
    """Logs a message with a timestamp.
    Args:
        message (str): The message to be logged.
    """
    timestamp_format = '%Y-%b-%d-%H-%M-%S'
    now = datetime.now()
    timestamp = now.strftime(timestamp_format)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(f"{timestamp}, {message}\n")

def extract(url):
    """
    Extracts Country and GDP data from the Wikipedia page.
    """
    log_progress("Extract phase Started")
    try:
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')

        target_table = None
        # Find the correct table by its unique caption
        for caption in soup.find_all('caption'):
            if "GDP forecast or estimate (million US$) by country" in caption.get_text():
                target_table = caption.find_parent('table')
                break

        if target_table is None:
            raise ValueError("Could not find the target table with the specific caption.")

        data = []
        # Iterate through rows, skipping the header rows (first 2 rows)
        for row in target_table.find('tbody').find_all('tr')[2:]:
            cells = row.find_all('td')
            # Ensure the row has enough cells to prevent errors
            if len(cells) > 1:
                country = cells[0].get_text(strip=True)
                # IMF Forecast is the 2nd column (index 1) in the data rows
                estimate = cells[1].get_text(strip=True)
                data.append([country, estimate])

        # Create DataFrame from the manually parsed data
        df = pd.DataFrame(data, columns=['Country', 'Estimate'])

    except Exception as e:
        log_progress(f"Error during extraction: {e}")
        raise ValueError(f"Failed to extract and parse table. Reason: {e}")

    # Clean country names (e.g., remove annotations like [n 1])
    df['Country'] = df['Country'].str.replace(r'\s*\[[^\]]*\]', '', regex=True).str.strip()
    # Remove the 'World' total row if it exists
    df = df[df['Country'] != 'World'].copy()

    log_progress("Extract phase Ended")
    return df

def extract_region_data(region_pages):
    """Extract GDP data from region-specific Wikipedia pages."""
    log_progress("Region-based Extract phase Started")
    all_data = []

    for region, url in region_pages.items():
        try:
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            table = soup.find('table', class_='wikitable')
            if not table:
                log_progress(f"No table found for region {region}")
                continue

            # Define a blacklist of non-country terms that might appear in the country column
            # These are typically aggregate rows or table headers that are not actual countries.
            blacklist = {
                "World", "European Union", "Commonwealth of Nations", "North America",
                "Arab League", "Africa", "Asia-Pacific", "Latin America & Caribbean",
                "Oceania", "Europe", "Total"
            }

            # Determine column indices based on region or table structure
            country_idx = 0
            gdp_idx = 1

            # Special handling for regions with 'Rank' column at index 0
            if region in ["Africa", "Arab League", "Asia-Pacific", "Commonwealth", "Latin America & Caribbean"]:
                country_idx = 1
                gdp_idx = 2
            # Special handling for North America due to its specific column structure
            elif region == "North America":
                country_idx = 2 # Country name is at index 2
                gdp_idx = 3   # IMF GDP is at index 3
            
            data_rows = []
            # Iterate through rows, skipping potential header rows (first few rows)
            # and focusing on rows that contain actual data cells (<td>)
            for row in table.find_all('tr'):
                # Check if the row contains <td> elements, indicating a data row
                if row.find('td'):
                    data_rows.append(row)

            if not data_rows:
                log_progress(f"No data rows found for region {region}")
                continue

            for i, row in enumerate(data_rows):
                cols = row.find_all(['td', 'th']) # Get all cells in the row

                # Ensure we have enough columns to avoid index errors
                if len(cols) <= max(country_idx, gdp_idx):
                    continue

                country = cols[country_idx].get_text(strip=True).replace('\xa0', ' ')
                gdp_text = cols[gdp_idx].get_text(strip=True).replace(',', '').replace('—', '').replace('−', '').strip()

                # Clean country names (e.g., remove annotations like [n 1])
                country = re.sub(r'\s*\[[^\]]*\]', '', country).strip()

                # Skip rows if the country name is in the blacklist or is empty
                if not country or country in blacklist:
                    continue

                # Filter out non-numeric GDP values
                if not gdp_text or not gdp_text.replace('.', '').isdigit():
                    continue

                gdp_value = float(gdp_text)
                # Convert millions to billions, but only if not already in billions (like Arab League)
                if region != "Arab League":
                    gdp = round(gdp_value / 1000, 2)  # millions → billions
                else:
                    gdp = round(gdp_value, 2) # Already in billions
                all_data.append({"Region": region, "Country": country, "GDP_USD_billion": gdp})
        except Exception as e:
            log_progress(f"Error processing region {region}: {e}")

    log_progress("Region-based Extract phase Ended")
    return pd.DataFrame(all_data)

def transform(df):
    """Transforms the extracted data into the required format.
    Assumes df has 'Country' and 'Estimate' columns for main GDP, or
    'Region', 'Country', 'GDP_USD_billion' for regional data.
    """
    log_progress("Transform phase Started")

    if 'Estimate' in df.columns:
        # For main GDP data
        df['Estimate'] = pd.to_numeric(df['Estimate'].str.replace(',', ''), errors='coerce')
        df['GDP_USD_billion'] = (df['Estimate'] / 1000).round(2)
        df = df[['Country', 'GDP_USD_billion']]
    
    # Drop rows where GDP could not be calculated
    df = df.dropna(subset=['GDP_USD_billion'])
    
    # Sort by GDP in descending order
    df = df.sort_values(by='GDP_USD_billion', ascending=False).reset_index(drop=True)
    
    log_progress("Transform phase Ended")
    return df

def load_to_json(df, filename):
    """Saves the DataFrame to a JSON file."""
    log_progress("Load phase Started")
    df.to_json(filename, orient='records', force_ascii=False, indent=4)
    log_progress("Load phase Ended")

def display_output(main_gdp_df, regional_gdp_df):
    """Prints the required outputs based on main and regional GDP data."""
    log_progress("Displaying Output Started")
    
    print("--- Countries with GDP over 100B USD (from main URL) ---")
    gdp_over_100b = main_gdp_df[main_gdp_df['GDP_USD_billion'] >= 100]
    print(gdp_over_100b.to_string())

    # Calculate and output the average GDP of top5 countries for each region
    top5_list = []
    for region, group in regional_gdp_df.groupby('Region'):
        top5 = group.nlargest(5, 'GDP_USD_billion')
        top5['Region'] = region
        top5_list.append(top5)
    
    if top5_list:
        top5_df = pd.concat(top5_list).reset_index(drop=True)
        avg_by_region = top5_df.groupby('Region')['GDP_USD_billion'].mean().round(2)
        print("\n--- Average GDP of the Top 5 countries by Region ---")
        print(avg_by_region.to_string())
    
    log_progress("Displaying Output Ended")

if __name__ == '__main__':
    log_progress("ETL Process Started")
    
    # Extract and Transform main GDP data
    main_gdp_raw = extract(URL)
    main_gdp_transformed = transform(main_gdp_raw)

    # Extract and Transform regional GDP data
    regional_gdp_raw = extract_region_data(REGION_PAGES)
    regional_gdp_transformed = transform(regional_gdp_raw)
    
    # Load main GDP data to JSON
    load_to_json(main_gdp_transformed, JSON_FILE)
    
    # Display outputs
    display_output(main_gdp_transformed, regional_gdp_transformed)
    
    log_progress("ETL Process Completed")
