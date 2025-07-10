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
DB_NAME = "/Users/admin/data_engineering_course_materials/missions/W1/W1M3/World_Economies.db"
TABLE_NAME = "Countries_by_GDP"

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
def load_to_db_with_sql(df, db_name, table_name):
    """Saves the DataFrame to a SQLite database using explicit SQL queries."""
    log_progress("DB Load phase Started (with SQL queries)")
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1. Drop table if it exists (optional, for clean slate)
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # 2. Create table using SQL CREATE TABLE statement
    # Define schema based on your DataFrame columns
    create_table_query = f"""
    CREATE TABLE {table_name} (
        Country TEXT,
        GDP_USD_billion REAL
    )
    """
    cursor.execute(create_table_query)

    # 3. Insert data using SQL INSERT statements
    # Prepare data for insertion
    data_to_insert = df[['Country', 'GDP_USD_billion']].values.tolist()

    # Use a parameterized query to prevent SQL injection
    insert_query = f"INSERT INTO {table_name} (Country, GDP_USD_billion) VALUES (?, ?)"
    cursor.executemany(insert_query, data_to_insert)

    # Commit changes and close connection
    conn.commit()
    conn.close()
    log_progress("DB Load phase Ended (with SQL queries)")


def extract_region_data(region_pages):
    """Extracts GDP data from region-specific Wikipedia pages."""
    log_progress("Region-based Extract phase Started")
    all_data = []

    for region, url in region_pages.items():
        # print(f"DEBUG: Processing region: {region} from URL: {url}") # New debug print
        try:
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            table = soup.find('table', class_='wikitable')
            if not table:
                log_progress(f"No table found for region {region}")
                # print(f"DEBUG: No table found for region {region}") # New debug print
                continue

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
            for row in table.find_all('tr'):
                if row.find('td'): # If the row contains a <td>, it's likely a data row
                    data_rows.append(row)

            if not data_rows:
                log_progress(f"No data rows found for region {region}")
                # print(f"DEBUG: No data rows found for region {region}") # New debug print
                continue

            for i, row in enumerate(data_rows): # Added enumerate for row number
                cols = row.find_all(['td', 'th']) # Get all cells in the row

                if len(cols) <= max(country_idx, gdp_idx):
                    continue

                country = cols[country_idx].get_text(strip=True).replace('\xa0', ' ')
                gdp_text = cols[gdp_idx].get_text(strip=True).replace(',', '').replace('—', '').replace('−', '').strip()

                # Clean country names (e.g., remove annotations like [n 1])
                country = re.sub(r'\s*\[[^\]]*\]', '', country).strip()

                # Skip rows if the country name is in the blacklist of is empty
                if not country or country in blacklist:
                    continue

                # Filter out non-numeric GDP values
                if not gdp_text or not gdp_text.replace('.', '').isdigit():
                    # print(f"DEBUG: Region: {region}, Row {i} skipped: Non-numeric GDP. gdp_text='{gdp_text}'") # New debug print
                    continue

                if region != "Arab League":
                    gdp = round(float(gdp_text) / 1000, 2) # millions → billions
                else:
                    gdp = round(float(gdp_text), 2)

                all_data.append({"Region": region, "Country": country, "GDP_USD_billion": gdp})

        except Exception as e:
            log_progress(f"Error processing region {region}: {e}")
            # print(f"DEBUG: Error processing region {region}: {e}") # New debug print

    log_progress("Region-based Extract phase Ended")
    return pd.DataFrame(all_data)


def transform(df):
    """Transforms the extracted data into the required format."""
    log_progress("Transform phase Started")
    
    # Convert 'Estimate' column to numeric, removing commas
    df['Estimate'] = pd.to_numeric(df['Estimate'].str.replace(',', ''), errors='coerce')
    
    # Convert GDP from millions to billions and round to 2 decimal places
    df['GDP_USD_billion'] = (df['Estimate'] / 1000).round(2)
    
    # Select and reorder the final columns
    df = df[['Country', 'GDP_USD_billion']]
    
    # Drop rows where GDP could not be calculated
    df = df.dropna(subset=['GDP_USD_billion'])
    
    # Sort by GDP in descending order
    df = df.sort_values(by='GDP_USD_billion', ascending=False).reset_index(drop=True)
    
    log_progress("Transform phase Ended")
    return df

def load_to_json(df, filename):
    """Saves the DataFrame to a JSON file."""
    log_progress("JSON Load phase Started")
    df.to_json(filename, orient=
    'records', force_ascii=False, indent=4)
    log_progress("JSON Load phase Ended")

def load_to_db(df, db_name, table_name):
    """Saves the DataFrame to a SQLite database."""
    log_progress("DB Load phase Started")
    conn = sqlite3.connect(db_name)
    # The table will only have 'Country' and 'GDP_USD_billion' attributes
    db_df = df[['Country', 'GDP_USD_billion']].copy()
    db_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    log_progress("DB Load phase Ended")

def calculate_top5_avg_by_region(df):
    """Prints average GDP of top 5 countries per region."""
    log_progress("Top5 Region GDP Average phase Started")

    top5_list = []
    for region, group in df.groupby('Region'):
        top5 = group.nlargest(5, 'GDP_USD_billion')
        top5['Region'] = region # Conserve Region
        top5_list.append(top5)

    top5_df = pd.concat(top5_list).reset_index(drop=True)

    # Calculate avg
    avg_by_region = top5_df.groupby('Region')['GDP_USD_billion'].mean().round(2)

    print("\n--- Average GDP of Top 5 Countries by Region ---")
    print(avg_by_region.to_string())

    log_progress("Top5 Region GDP Average phase Ended")


def run_queries(db_name, table_name):
    """Runs the required SQL queries and prints the results."""
    log_progress("SQL Query phase Started")
    conn = sqlite3.connect(db_name)

    print("\n--- Countries with GDP over 100B USD ---")
    query1 = f"SELECT Country, GDP_USD_billion FROM {table_name} WHERE GDP_USD_billion >= 100"
    df1 = pd.read_sql_query(query1, conn)
    print(df1.to_string())

    conn.close()
    log_progress("SQL Query phase Ended")

if __name__ == '__main__':
    log_progress("ETL Process Started")
    extracted_data = extract(URL)
    transformed_data = transform(extracted_data)
    load_to_json(transformed_data, JSON_FILE)
    load_to_db(transformed_data, DB_NAME, TABLE_NAME)
    run_queries(DB_NAME, TABLE_NAME)

    # Extract Region-based GDP and Analysis
    df_region = extract_region_data(REGION_PAGES)
    # print(df_region.groupby('Region').size())  # 각 리전별 데이터 수 확인 - Uncommented
    # print(df_region.head(10))  # 추출된 예시 확인 - Uncommented
    calculate_top5_avg_by_region(df_region)
    
    log_progress("ETL Process Completed")