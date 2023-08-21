import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
import json
import os
import re
import requests

import boto3
from dotenv import load_dotenv
import openai
import pandas as pd
from tqdm import tqdm
from litellm import completion
import litellm

from generate_docx import DocxTemplateFiller
load_dotenv()



## Set your email
os.environ["LITELLM_EMAIL"] = "ishaan1@berri.ai"
litellm.openai_key = os.getenv("OPENAI_API_KEY")
litellm.openrouter_key = os.getenv("OPENROUTER_API_KEY")

## Set debugger to true
litellm.debugger = True

API_KEY = os.getenv("API_KEY")
openai.api_key = str(API_KEY)

MAPBOX_TOKEN = str(os.getenv("MAPBOX_TOKEN"))
ZYTE_API_KEY = str(os.getenv("ZYTE_API_KEY"))

def get_location_details(upper_left_lat, upper_left_lon, lower_right_lat, lower_right_lon):
    center_lat = (upper_left_lat + lower_right_lat) / 2
    center_lon = (upper_left_lon + lower_right_lon) / 2

    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{center_lon},{center_lat}.json?access_token={MAPBOX_TOKEN}"

    response = requests.get(url)
    location_data = response.json()

    if "features" in location_data and len(location_data["features"]) > 0:
        address = location_data["features"][0]["place_name"].split(', ')
        city_name = address[0] if len(address) > 0 else None
        region_name = address[1] if len(address) > 1 else None
        country_code = address[-1].upper() if len(address) > 2 else None

        return country_code, region_name, city_name
    else:
        return None, None, None
    
def decimal_to_dms(decimal):
    """
    Convert a decimal degree value to degrees, minutes, seconds.
    """
    degree = int(decimal)
    minute_decimal = abs(decimal - degree) * 60
    minute = int(minute_decimal)
    second = int((minute_decimal - minute) * 60)
    return degree, minute, second

def latlong_to_string(lat, lon):
    lat_deg, lat_min, lat_sec = decimal_to_dms(lat)
    lon_deg, lon_min, lon_sec = decimal_to_dms(lon)

    # Determine the hemisphere for the latitude and longitude
    lat_hem = "N" if lat >= 0 else "S"
    lon_hem = "E" if lon >= 0 else "W"

    return "{}°{}'{}\"{}, {}°{}'{}\"{}".format(
        abs(lat_deg), lat_min, lat_sec, lat_hem,
        abs(lon_deg), lon_min, lon_sec, lon_hem
    )
        
def read_s3_file_to_dataframe(file_key: str) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame."""
    df_name = file_key.split('/')[-1].split('_')[0].lower()
    
    # s3_client = boto3.client('s3')
    # response = s3_client.get_object(Bucket=bucket, Key=file_key)
    # content = response['Body']
    
    # return (df_name, pd.read_csv(BytesIO(content.read())))
    
    return (df_name, pd.read_csv(file_key))

def fetch_data_from_s3(paths):
    with ThreadPoolExecutor(max_workers=len(paths)) as executor:
        futures = [executor.submit(read_s3_file_to_dataframe, path) for path in paths]
        return [future.result() for future in as_completed(futures)]

def filter_by_coordinates(df: pd.DataFrame, lat_col: str, long_col: str, upper_left: tuple, lower_right: tuple) -> pd.DataFrame:
    mask = (
        (df[lat_col] <= upper_left[0]) &
        (df[lat_col] >= lower_right[0]) &
        (df[long_col] >= upper_left[1]) &
        (df[long_col] <= lower_right[1])
    )
    return df[mask]

def extract_article_from_url(url: str) -> dict:
    """Extract article data from the given URL using the Zyte API."""
    try:
        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(ZYTE_API_KEY, ""),
            json={
                "url": url,
                "article": True
            }
        )
        article_content =  response.json().get("article").get("articleBody")
        prompt = f"Article: {article_content}\n\nAbove is a news article. Provide a succinct summary of no more than three sentences of the key issue in the article."
        response = call_gpt_batch_prompts([prompt], max_tokens=256)
        return response
    except:
        return None

def standardize_date(input_date, format='%B %dth, %Y'):
    try:
        date_obj = None
        formats_to_try = ['%d %B %Y', '%Y%m%d', '%Y-%m-%dT%H:%M:%S.%fZ']
        for fmt in formats_to_try:
            try:
                date_obj = datetime.strptime(input_date, fmt)
                break
            except ValueError:
                pass

        return date_obj

    except Exception as e:
        return f"Error: {str(e)}"

def call_gpt_batch_prompts(messages_list, model='gpt-3.5-turbo-16k', max_tokens=4000):
    import time
    prompts = [
        {
            "role": "user",
            "content": json.dumps(messages_list)
        },
        {
            "role": "system",
            "content": f"Complete all {len(messages_list)} elements of the array. Reply with an array of all {len(messages_list)} completions."
        }
    ]

    response = None
    rate_limited_models = set()
    model_expiration_times = {}

    # gpt-3.5-turbo-16k is the primary model
    fallbacks = ["gpt-3.5-turbo-16k", "openrouter/anthropic/claude-2", "j2-mid", "gpt-3.5-turbo"]

    # pick any of the supported completion models here: https://docs.litellm.ai/docs/completion/supported
    start_time = time.time()
    while response == None and time.time() - start_time < 120: # 120 seconds to try primary model + fallback
        for model in fallbacks:
        # loop thru all models
            try:
                if model in rate_limited_models: # check if model is currently cooling down
                    if model_expiration_times.get(model) and time.time() >= model_expiration_times[model]:
                        rate_limited_models.remove(model) # check if it's been 60s of cool down and remove model
                    else:
                        continue # skip model
                
                # Call LLM API 
                response = completion(model=model,
                        temperature=0.0,
                        messages=prompts, 
                        max_tokens=max_tokens,
                        force_timeout=30,
                    )
                
                # check if response == None
                if response != None:
                    return response

            except Exception as e:
                print(f"got exception {e}")
                rate_limited_models.add(model)
                model_expiration_times[model] = time.time() + 60 # cool down this selected model
                print(f"rate_limited_models {rate_limited_models}")
                pass


    if 'choices' in response and isinstance(response['choices'], list):
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            
            return response.choices[0].message.content
    else:
        print("Unexpected response format:", response)
        return []

def categorize_event_batch(contents, batch_size=10, max_tokens=512):
    themes = [
        "Diplomatic & Political Relations",
        "Information & Cybersecurity",
        "Military Affairs",
        "Economic Dynamics",
        "Social & Cultural Insights",
        "Infrastructure & Development",
        "Organizations & Governance",
        "Public & Civil Movements",
        "General Population & Human Interest"
    ]

    categorized_themes = []
    
    batch_prompts = [f"Which of the following themes best fits this content: {content}?\n" +
                          "\n".join([f"{j + 1}. {theme}" for j, theme in enumerate(themes)]) +
                          "\nPlease choose the number corresponding to the best theme."
                          for content in contents
                          ]
    batch_responses = _batched_call_helper(batch_prompts, max_tokens, batch_size)

    for response_content in batch_responses:
        try:
            chosen_index = int(response_content) - 1
            categorized_themes.append(themes[chosen_index])
        except ValueError:
            match = re.search(r'(\d+)\.', response_content)
            if match:
                chosen_index = int(match.group(1)) - 1
                categorized_themes.append(themes[chosen_index])
            else:
                categorized_themes.append(None)

    return categorized_themes

def generate_headline_batched(contents, batch_size=10, max_tokens=512):
    headlines = []
    
    batch_prompts = [
            f"Write a succinct headline on the geopolitical event described: {content}\n\nPlease provide in json format with the following key: 'headline'"
            for content in contents
        ]

    batch_responses = _batched_call_helper(batch_prompts, max_tokens, batch_size)

    for response_content in batch_responses:
        try:
            if 'headline' in response_content:
                headlines.append(response_content['headline'])
            else:
                headlines.append(None)
        except json.JSONDecodeError:
            headlines.append(None)

    return headlines

from concurrent.futures import TimeoutError

def _batched_call_helper(prompts, max_tokens, batch_size):
    results = []
    with ThreadPoolExecutor() as executor:
        futures_to_prompts = {executor.submit(call_gpt_batch_prompts, prompts[i:i+batch_size], max_tokens=max_tokens): prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)}       
        
        for future in tqdm(as_completed(futures_to_prompts), total=len(futures_to_prompts), desc="Processing batches"):
            results.extend(future.result(timeout=60))
    
    return results


def generate_summary(contents):
    prompt = f"Events Data: {contents}\n\nAbove include events data. Provide an executive summary of the most significant events, geopolitical trends, and patterns present in the given events data. Ensure the summary is presented in paragraph format adhering to AP style guidelines. Avoid transitional words like 'overall'."
    response = call_gpt_batch_prompts([prompt], model='gpt-4', max_tokens=1024)
    return response.strip('][')

def generate_conclusions(contents):
    
    prompts = [
        f"Events Data: {contents}\n\nBased on the events data, write a paragraph summary of five or less sentences on what the overall sentiment is in relation to geopolitical stability. Adhere to AP style guidelines.",
        f"Events Data: {contents}\n\nBased on the events data, write a paragraph summary of six or less sentences on implications on global or regional affairs. Avoid transitional words like 'overall'. Adhere to AP style guidelines.",
    ]

    conclusions = []
    for prompt in prompts:
        response = call_gpt_batch_prompts([prompt], model='gpt-4', max_tokens=512)
        conclusions.append(response)
    
    prompt = [f"Provide a bulleted list of recommended action points based on the sentiment and implications. \nSentiment: {conclusions[0]} \nImplications: {conclusions[1]}"]
    while True:
        response = call_gpt_batch_prompts(prompt, model='gpt-4', max_tokens=512)
        if response[0]=='-':
            break;
    
    conclusions.append(response)
    
    return conclusions

def generate_new_report(args, data):
    
    paths = [
        "./source/data/ACLED_2023-05-01-2023-07-21-Central-South_America.csv",
        "./source/data/gdelt_15m_update_20230411100229.csv",
        "./source/data//seerist_pulse_v2_scribe_southcom2023-07-19.csv"
    ]

    dataframes = fetch_data_from_s3(paths)
    dataframes = dict(dataframes)

    # Define transformations
    def preprocess_dataframe(df, coord_cols, date_col, content_col, args, url_col=None):
        df = filter_by_coordinates(df, *coord_cols, args.upper_left, args.lower_right)
        if url_col:
            urls = df[url_col].unique()  # Fetch unique URLs first to reduce number of requests
            with ThreadPoolExecutor(max_workers=4) as executor:
                contents = list(tqdm(executor.map(extract_article_from_url, urls), total=len(urls)))
            url_content_dict = dict(zip(urls, contents))
            df['content'] = df[url_col].map(url_content_dict)
            df = df.dropna(subset=['content'])
        df = df[[date_col, content_col]]
        df.columns = ['day', 'content']
        df['day'] = pd.to_datetime(df['day'].apply(standardize_date, format='%Y%m%d'))
        return df

    acled = preprocess_dataframe(dataframes['acled'], ['latitude', 'longitude'], 'event_date', 'notes', args)
    gdelt = preprocess_dataframe(dataframes['gdelt'], ['actiongeo_lat', 'actiongeo_long'], 'day', 'content', args, 'sourceurl')
    seerist = preprocess_dataframe(dataframes['seerist'], ['region_center_latitude', 'region_center_longitude'], 'event_timestamp', 'event_summary', args)

    df = pd.concat([acled, gdelt, seerist], ignore_index=True)

    start = datetime.strptime(args.dates[0], '%Y%m%d')
    end = datetime.strptime(args.dates[1], '%Y%m%d')
    df = df[(df['day'] >= start) & (df['day'] <= end)]

    contents = df['content'].tolist()
    df['headline'] = generate_headline_batched(contents, max_tokens=1000)
    df = df.dropna()
    headlines = df['headline'].tolist()
    df['theme'] = categorize_event_batch(headlines)
    df = df.drop_duplicates(subset=['headline'])
    df = df.sort_values(by=['theme', 'day'], ascending=[True, False])
    df['day'] = df['day'].apply(lambda d: d.strftime('%B %dth, %Y'))

    data['EXECUTIVE SUMMARY'] = generate_summary(headlines)
    
    print('Generated summary....')

    # Use groupby for efficiency instead of looping
    themes_content_dict = df.groupby('theme').apply(lambda group: group.apply(lambda row: f"{row['headline']} ({row['day']})", axis=1).tolist()).to_dict()
    data['theme'] = themes_content_dict

    sentiment, implications, actions = generate_conclusions(headlines)

    data['OVERALL SENTIMENT'] = sentiment
    data['IMPLICATIONS'] = implications
    data['RECOMMENDED ACTIONS'] = actions
    
    print('Generated conclusions....')

    file_path = 'output/' + data['LOCATION'] + f'_{args.dates[0]}-{args.dates[1]}.json'
    
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
        
    print('Generated JSON....')


def main(args):

    data = {}

    data['COORDINATES']=f'({latlong_to_string(args.upper_left[0],args.upper_left[1])}), ({latlong_to_string(args.lower_right[0], args.lower_right[1])})'
    country_code, region_name, city_name = get_location_details(args.upper_left[0], args.upper_left[1], args.lower_right[0], args.lower_right[1])
    city_name = None if region_name == city_name else city_name
    data['LOCATION'] = "/".join([loc for loc in [country_code, region_name, city_name] if loc is not None])
    
    json_path = 'output/'+ data['LOCATION'] + f'_{args.dates[0]}-{args.dates[1]}.json'

    if os.path.exists(json_path):
        pass
    else:
        generate_new_report(args, data)
    
    template_path = 'source/demo-template.docx'
    output_path = json_path.split('.')[0]+'-report.docx'
    
    filler = DocxTemplateFiller(template_path, json_path, output_path)
    filler.fill_placeholders()
    filler.save_output()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter datasets based on latitude and longitude coordinates.")
    parser.add_argument("--upper_left", type=float, nargs=2, metavar=('LAT', 'LONG'), required=True, help="Upper left coordinates (latitude and longitude).")
    parser.add_argument("--lower_right", type=float, nargs=2, metavar=('LAT', 'LONG'), required=True, help="Lower right coordinates (latitude and longitude).")
    parser.add_argument("--dates", type=str, nargs=2, metavar=('BEG', 'END'), required=True, help="Beginning and end dates (YYYYMMDD)")
    args = parser.parse_args()

    main(args)
