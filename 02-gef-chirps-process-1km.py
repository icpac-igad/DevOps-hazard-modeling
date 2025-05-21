from prefect import flow, task
from dotenv import load_dotenv
import os
import shutil
import glob
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Optional, List

from utils import (
    gefs_chrips_list_tiff_files,
    gefs_chrips_download_files,
    gefs_chrips_process,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df,
    gefs_chirps_update_input_data
)

load_dotenv()

def get_last_date_from_rain(zone_dir):
    """
    Read the existing rain.txt file and determine the last date in the file.
    
    Parameters:
    ----------
    zone_dir : str
        Path to the zone directory containing rain.txt
        
    Returns:
    -------
    datetime
        The last date in the file, or None if the file doesn't exist or can't be read
    """
    rain_file = os.path.join(zone_dir, 'rain.txt')
    
    if not os.path.exists(rain_file):
        print(f"No existing rain.txt found at {rain_file}")
        return None
    
    try:
        # Read the rain.txt file
        df = pd.read_csv(rain_file, sep=",")
        
        # Check if NA column exists (which contains the dates in YYYYDDD format)
        if 'NA' not in df.columns:
            print(f"Invalid format in rain.txt - missing 'NA' column")
            return None
        
        # Convert the last date to datetime
        last_date_str = df['NA'].iloc[-1]
        last_date = datetime.strptime(str(last_date_str), '%Y%j')
        
        print(f"Last date in existing rain.txt: {last_date.strftime('%Y-%m-%d')} (Day {last_date_str})")
        return last_date
        
    except Exception as e:
        print(f"Error reading existing rain.txt: {e}")
        return None

@task
def validate_tiff_files(date_dir):
    """
    Validate all TIFF files in the specified directory.
    
    Args:
        date_dir: Directory containing TIFF files
        
    Returns:
        bool: True if all files are valid, False otherwise
    """
    import rasterio
    
    print(f"Validating TIFF files in {date_dir}")
    tiff_files = glob.glob(f"{date_dir}/*.tif")
    
    if not tiff_files:
        print(f"No TIFF files found in {date_dir}")
        return False
    
    all_valid = True
    corrupted_files = []
    
    for file_path in tiff_files:
        try:
            # Attempt to open the file with rasterio to verify it's readable
            with rasterio.open(file_path) as src:
                # Just checking a property to verify the file can be read
                shape = src.shape
                print(f"Validated: {os.path.basename(file_path)} - shape: {shape}")
        except Exception as e:
            all_valid = False
            corrupted_files.append((file_path, str(e)))
            print(f"Corrupted file: {os.path.basename(file_path)} - Error: {e}")
    
    if not all_valid:
        print(f"Found {len(corrupted_files)} corrupted TIFF files in {date_dir}")
        
        # Rename the corrupted directory for investigation if needed
        backup_dir = f"{date_dir}_corrupted"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        
        print(f"Moving corrupted directory to {backup_dir} for investigation")
        shutil.move(date_dir, backup_dir)
        
        # Create a fresh directory for the new download
        os.makedirs(date_dir, exist_ok=True)
    
    return all_valid

@task
def get_current_date():
    """Get the current date in YYYYMMDD format."""
    return datetime.now().strftime('%Y%m%d')

@task
def check_data_availability(base_url, date_string):
    """
    Check if GEFS-CHIRPS data is available for the specified date.
    
    Args:
        base_url: Base URL for the GEFS-CHIRPS data
        date_string: Date in YYYYMMDD format
        
    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        # Parse the date string
        year = date_string[:4]
        month = date_string[4:6]
        day = date_string[6:]
        
        # Construct the URL for the specific date
        url = f"{base_url}{year}/{month}/{day}/"
        
        # Send a request to check if the URL exists
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content to check if there are TIFF files available
            soup = BeautifulSoup(response.text, 'html.parser')
            tiff_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.tif')]
            
            # If there are TIFF files, data is available
            return len(tiff_files) > 0
        
        return False
    except Exception as e:
        print(f"Error checking data availability for {date_string}: {e}")
        return False

@task
def get_best_available_date(base_url, days_to_check=7):
    """
    Get the most recent date for which data is available.
    Start with today and go back up to 'days_to_check' days.
    
    Args:
        base_url: Base URL for the GEFS-CHIRPS data
        days_to_check: Number of days to check backward
        
    Returns:
        str: Date in YYYYMMDD format, or None if no data is available
    """
    today = datetime.now()
    
    for i in range(days_to_check):
        check_date = today - timedelta(days=i)
        date_string = check_date.strftime('%Y%m%d')
        print(f"Checking data availability for {date_string}...")
        
        if check_data_availability(base_url, date_string):
            print(f"Data found for {date_string}")
            return date_string
    
    print(f"No data found for the last {days_to_check} days")
    return None

@task
def setup_environment():
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    download_dir = f'{data_path}geofsm-input/gefs-chirps'
    params = get_dask_client_params()
    client = Client(**params)
    print(f"Environment setup: data_path={data_path}, download_dir={download_dir}")
    return data_path, download_dir, client

@task
def get_gefs_files(base_url, date_string):
    all_files = gefs_chrips_list_tiff_files(base_url, date_string)
    print(f"Found {len(all_files)} files for date {date_string}")
    return all_files

@task
def download_gefs_files(url_list, date_string, download_dir, force_redownload=False):
    date_dir = f"{download_dir}/{date_string}"
    
    # If force_redownload is True or the directory doesn't exist or is empty, download the files
    if force_redownload or not os.path.exists(date_dir) or not os.listdir(date_dir):
        if os.path.exists(date_dir):
            print(f"Removing existing directory: {date_dir}")
            shutil.rmtree(date_dir)
        
        print(f"Downloading data for {date_string}...")
        gefs_chrips_download_files(url_list, date_string, download_dir)
    else:
        print(f"Data for {date_string} already exists in {date_dir}, validating files...")
        
        # Validate the files
        if not validate_tiff_files(date_dir):
            print(f"Files in {date_dir} are corrupted, redownloading...")
            # Redownload the files
            return download_gefs_files(url_list, date_string, download_dir, True)
            
    return date_dir

@task
def process_gefs_data(input_path):
    print(f"Processing GEFS-CHIRPS data from {input_path}")
    try:
        return gefs_chrips_process(input_path)
    except Exception as e:
        print(f"Error processing GEFS-CHIRPS data: {e}")
        # Check if this is a file corruption issue
        if "TIFFReadDirectory:Failed" in str(e):
            print("TIFF file corruption detected. Try redownloading the data.")
            raise RuntimeError(f"Corrupted TIFF file detected in {input_path}. Please delete this directory and try again.")
        raise

@task
def process_zone(data_path, pds, zone_str):
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
    print(f"Processed zone {zone_str}")
    return z1ds, pdsz1, zone_extent

@task
def regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent):
    return regrid_dataset(
        pdsz1,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )

@task
def calculate_zone_means(regridded_data, zone_ds):
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_gefs_chirps_results(results_df, data_path, zone_str, date_string):
    """
    Save processed GEFS-CHIRPS results and update input data.
    This will merge with existing rain.txt data and create all necessary output files.
    """
    try:
        # Create output directory
        output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{zone_input_path}{zone_str}", exist_ok=True)
        
        # Format dates
        start_date = pd.to_datetime(results_df['time'].min())
        end_date = pd.to_datetime(results_df['time'].max())
        
        # Save CSV file for future reference
        csv_file = f"{output_dir}/gefs_chirps_{date_string}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
        
        # Update GEFS-CHIRPS input data (this will update the rain.txt file and create all other required files)
        output_file = gefs_chirps_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date)
        print(f"GEFS-CHIRPS data processed and rain.txt updated: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error saving GEFS-CHIRPS results: {e}")
        raise

@flow
def process_single_zone_date(data_path, pds, zone_str, date_string):
    """Process a single zone for a specific date"""
    print(f"Processing zone {zone_str} for date {date_string}...")
    
    # Check if data for this zone and date has already been processed
    date_obj = datetime.strptime(date_string, '%Y%m%d')
    date_ddd = date_obj.strftime('%Y%j')
    output_dir = f"{data_path}geofsm-input/processed/{zone_str}"
    processed_file = f"{output_dir}/rain_{date_ddd}.txt"
    
    if os.path.exists(processed_file):
        print(f"Data for zone {zone_str} and date {date_string} already processed. Skipping.")
        return processed_file
    
    z1ds, pdsz1, zone_extent = process_zone(data_path, pds, zone_str)
    input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(pdsz1, input_chunk_sizes, output_chunk_sizes, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    
    # Use the function that integrates CSV saving and updating input data
    txt_file = save_gefs_chirps_results(zone_means, data_path, zone_str, date_string)
    
    return txt_file

@flow
def process_single_zone(data_path, download_dir, base_url, zone_str, start_date, end_date):
    """Process a single zone for a date range"""
    
    print(f"\n===== Processing {zone_str} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} =====")
    
    # For GEFS-CHIRPS, we now only process the latest available date since it includes the forecast
    latest_date_str = end_date.strftime('%Y%m%d')
    
    # Get files for this date
    url_list = get_gefs_files(base_url, latest_date_str)
    
    # Download and validate files
    input_path = download_gefs_files(url_list, latest_date_str, download_dir)
    
    # Process the data
    try:
        print(f"Processing data for {latest_date_str}...")
        pds = process_gefs_data(input_path)
        
        # Process for this specific date
        txt_file = process_single_zone_date(data_path, pds, zone_str, latest_date_str)
        return [txt_file] if txt_file else []
    except Exception as e:
        print(f"Error processing {zone_str} for date {latest_date_str}: {e}")
        # If we get a file corruption error, try to redownload
        if "Corrupted TIFF file detected" in str(e):
            print(f"Attempting to redownload and process {latest_date_str} data...")
            # Force redownload by removing the directory
            if os.path.exists(input_path):
                shutil.rmtree(input_path)
            # Try again with fresh download
            input_path = download_gefs_files(url_list, latest_date_str, download_dir, True)
            pds = process_gefs_data(input_path)
            txt_file = process_single_zone_date(data_path, pds, zone_str, latest_date_str)
            return [txt_file] if txt_file else []
        return []

@flow
def gefs_chirps_all_zones_workflow(process_all_zones: bool = True, specific_zone: Optional[str] = None, days_to_process: int = 30):
    """
    Main workflow for processing GEFS-CHIRPS data for all zones.
    
    Args:
        process_all_zones: Whether to process all zones or just a specific one
        specific_zone: Zone to process if process_all_zones is False (can be None)
        days_to_process: Maximum number of days back to process if no existing data is found
        
    Returns:
        Dict containing the paths to the generated txt files
    """
    data_path, download_dir, client = setup_environment()
    
    try:
        base_url = "https://data.chc.ucsb.edu/products/EWX/data/forecasts/CHIRPS-GEFS_precip_v12/daily_16day/"
        
        # Get the most recent available date
        latest_available_date = get_best_available_date(base_url)
        if latest_available_date is None:
            print("No data available for processing. Exiting workflow.")
            return {'txt_files': []}
        
        # Convert to datetime for comparisons
        latest_date_obj = datetime.strptime(latest_available_date, '%Y%m%d')
        print(f"Most recent available data: {latest_date_obj.strftime('%Y-%m-%d')}")
        
        # Process all zones from the shapefile
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        if not os.path.exists(master_shapefile):
            print(f"ERROR: Master shapefile not found at {master_shapefile}")
            raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
        
        all_zones = gp.read_file(master_shapefile)
        output_files = []
        
        # Determine which zones to process
        if process_all_zones:
            zones_to_process = all_zones['zone'].unique()
        else:
            if specific_zone is None:
                print("No specific zone provided. Exiting workflow.")
                return {'txt_files': []}
            zones_to_process = [specific_zone]
        
        # Process each zone separately
        for zone_str in zones_to_process:
            try:
                # Standardize zone string format
                if not isinstance(zone_str, str):
                    zone_str = str(zone_str)
                
                # Check the last date in existing rain.txt for this zone
                zone_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
                os.makedirs(zone_dir, exist_ok=True)
                
                last_date = get_last_date_from_rain(zone_dir)
                
                # If we have a last date, start from the next day
                # Otherwise, use a default start date (e.g., 30 days ago)
                if last_date:
                    start_date = last_date + timedelta(days=1)
                    print(f"Starting data collection from {start_date.strftime('%Y-%m-%d')}")
                else:
                    start_date = latest_date_obj - timedelta(days=days_to_process)
                    print(f"No existing data found. Using default start date: {start_date.strftime('%Y-%m-%d')}")
                
                # End date is the latest available data date
                end_date = latest_date_obj
                
                # If start date is after end date, skip this zone
                if start_date > end_date:
                    print(f"No new data to process for {zone_str}. Last data date {last_date.strftime('%Y-%m-%d')} is after or equal to latest available {end_date.strftime('%Y-%m-%d')}")
                    continue
                
                # Process the zone for the date range (which now only processes the latest date)
                zone_files = process_single_zone(data_path, download_dir, base_url, zone_str, start_date, end_date)
                output_files.extend(zone_files)
                
            except Exception as e:
                print(f"Error processing {zone_str}: {e}")
        
        print(f"Workflow completed successfully! Processed {len(output_files)} files")
        return {'txt_files': output_files}
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process GEFS-CHIRPS data for hydrological modeling')
    parser.add_argument('--specific-zone', type=str, default=None, 
                      help='Process only a specific zone (default: process all zones)')
    parser.add_argument('--days-to-process', type=int, default=30, 
                      help='Maximum number of days to process if no existing data is found (default: 30)')
    parser.add_argument('--force-redownload', action='store_true',
                      help='Force redownloading of data even if it exists locally')
    
    args = parser.parse_args()
    
    # If specific_zone is provided, set process_all_zones to False
    process_all_zones = args.specific_zone is None
    
    print(f"Starting GEFS-CHIRPS processing workflow")
    print(f"Processing {'all zones' if process_all_zones else f'zone {args.specific_zone}'}")
    print(f"Maximum days to process: {args.days_to_process}")
    
    # Note: We're not passing specific_zone if it's None
    if process_all_zones:
        result = gefs_chirps_all_zones_workflow(
            process_all_zones=True,
            days_to_process=args.days_to_process
        )
    else:
        result = gefs_chirps_all_zones_workflow(
            process_all_zones=False,
            specific_zone=args.specific_zone,
            days_to_process=args.days_to_process
        )
    
    print(f"Generated files: {result['txt_files']}")