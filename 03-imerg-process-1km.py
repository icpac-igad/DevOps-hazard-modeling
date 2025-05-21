from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import numpy as np
import rioxarray
from typing import Optional, List

from utils import (
    imerg_list_files_by_date,
    imerg_download_files,
    imerg_read_tiffs_to_dataset,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df,
    imerg_update_input_data
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
def setup_environment():
    """Set up the processing environment"""
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    imerg_store = f'{data_path}geofsm-input/imerg'
    params = get_dask_client_params()
    client = Client(**params)
    print(f"Environment setup: data_path={data_path}, imerg_store={imerg_store}")
    return data_path, imerg_store, client

@task
def get_imerg_files(start_date, end_date):
    """Get a list of IMERG files for the specified date range"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    url = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg/gis/early/"
    flt_str = '-S233000-E235959.1410.V07B.1day.tif'
    username = os.getenv("imerg_username")
    password = os.getenv("imerg_password")
    
    if not username or not password:
        raise ValueError("IMERG credentials not found in environment variables")
    
    file_list = imerg_list_files_by_date(url, flt_str, username, password, start_date_str, end_date_str)
    print(f"Found {len(file_list)} IMERG files for date range {start_date_str} to {end_date_str}")
    return file_list

@task
def download_imerg_files(file_list, imerg_store):
    """Download IMERG files"""
    download_dir = f"{imerg_store}"
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if files already exist
    existing_files = set(os.listdir(download_dir))
    to_download = []
    
    for url in file_list:
        filename = os.path.basename(url)
        if filename not in existing_files:
            to_download.append(url)
    
    if not to_download:
        print(f"All IMERG files already exist in {download_dir}, skipping download.")
    else:
        print(f"Downloading {len(to_download)} new IMERG files...")
        username = os.getenv("imerg_username")
        password = os.getenv("imerg_password")
        imerg_download_files(to_download, username, password, download_dir)
    
    return download_dir

@task
def process_imerg_data(input_path, start_date, end_date):
    """Process IMERG data into xarray format"""
    print(f"Processing IMERG data from {input_path} for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    data = imerg_read_tiffs_to_dataset(input_path, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
    
    # If the data is a DataArray, assign a name to it
    if isinstance(data, xr.DataArray) and not data.name:
        data = data.rename('precipitation')
        print(f"Assigned name 'precipitation' to DataArray")
    
    return data

@task
def rename_coordinates(imerg_data):
    """
    Rename 'x' and 'y' coordinates to 'lon' and 'lat' if they exist.
    This ensures compatibility with other functions expecting lat/lon.
    """
    # Print the object type and dimensions to help with debugging
    print(f"Object type: {type(imerg_data).__name__}")
    print(f"Dimensions: {list(imerg_data.dims)}")
    
    # Check if x and y are present in the dimensions
    if 'x' in imerg_data.dims and 'y' in imerg_data.dims:
        # Create a new object with renamed coordinates
        renamed_data = imerg_data.rename({'x': 'lon', 'y': 'lat'})
        print("Renamed 'x' to 'lon' and 'y' to 'lat'")
    else:
        renamed_data = imerg_data
        print("No renaming needed or coordinates not found")
    
    return renamed_data

@task
def process_zone(data_path, imerg_data, zone_str):
    """Process a zone from the combined shapefile"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    km_str = 1
    z1ds, zone_subset_ds, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, imerg_data)
    print(f"Processed zone {zone_str}")
    return z1ds, zone_subset_ds, zone_extent

@task
def regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent):
    """Regrid the precipitation data to match the zone extent at 1km resolution"""
    print(f"Input to regridding - type: {type(zone_subset_ds).__name__}, name: {getattr(zone_subset_ds, 'name', 'unnamed')}")
    
    # Get the result from regrid_dataset function
    result = regrid_dataset(
        zone_subset_ds,
        input_chunk_sizes,
        output_chunk_sizes,
        zone_extent,
        regrid_method="bilinear"
    )
    
    # Ensure the result has a name if it's a DataArray
    if isinstance(result, xr.DataArray) and not result.name:
        result = result.rename('precipitation')
        print("Named regridded DataArray as 'precipitation'")
    
    return result

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate zonal means for the regridded data"""
    # Ensure the input DataArray has a name
    if isinstance(regridded_data, xr.DataArray) and not regridded_data.name:
        print("WARNING: Received unnamed DataArray, renaming to 'precipitation'")
        regridded_data = regridded_data.rename('precipitation')
    
    # Call zone_mean_df with the properly named data
    return zone_mean_df(regridded_data, zone_ds)

@task
def save_imerg_results(results_df, data_path, zone_str, date_string):
    """
    Save processed IMERG results and update input data.
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
        csv_file = f"{output_dir}/imerg_{date_string}.csv"
        results_df.to_csv(csv_file, index=False)
        print(f"CSV results saved to {csv_file}")
        
        # Update IMERG input data (this will update the rain.txt file and create all other required files)
        output_file = imerg_update_input_data(results_df, zone_input_path, zone_str, start_date, end_date)
        print(f"IMERG data processed and rain.txt updated: {output_file}")
        
        return output_file
    except Exception as e:
        print(f"Error saving IMERG results: {e}")
        raise

@flow
def process_single_zone(data_path, imerg_data, zone_str, date_string):
    """Process a single zone for IMERG data"""
    print(f"Processing zone {zone_str} for IMERG data...")
    
    z1ds, zone_subset_ds, zone_extent = process_zone(data_path, imerg_data, zone_str)
    
    # Adjust input_chunk_sizes based on the dimensions in the data
    if 'lat' in zone_subset_ds.dims and 'lon' in zone_subset_ds.dims:
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
    else:
        input_chunk_sizes = {'time': 10, 'y': 30, 'x': 30}
    
    output_chunk_sizes = {'lat': 300, 'lon': 300}
    regridded_data = regrid_precipitation_data(zone_subset_ds, input_chunk_sizes, output_chunk_sizes, zone_extent)
    zone_means = calculate_zone_means(regridded_data, z1ds)
    
    # Use the function that integrates CSV saving and updating input data
    txt_file = save_imerg_results(zone_means, data_path, zone_str, date_string)
    
    return txt_file

@flow
def imerg_all_zones_workflow(process_all_zones: bool = True, specific_zone: Optional[str] = None, days_to_process: int = 30):
    """
    Main workflow for processing IMERG data for all zones.
    
    Args:
        process_all_zones: Whether to process all zones or just a specific one
        specific_zone: Zone to process if process_all_zones is False (can be None)
        days_to_process: Maximum number of days back to process if no existing data is found
        
    Returns:
        Dict containing the paths to the generated txt files
    """
    data_path, imerg_store, client = setup_environment()
    
    try:
        # Get today's date for reference
        today = datetime.now()
        yesterday = today - timedelta(days=1)  # IMERG data is usually available for yesterday
        end_date = yesterday
        
        print(f"Processing IMERG data up to {end_date.strftime('%Y-%m-%d')}")
        
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
                
                print(f"\n===== Processing {zone_str} =====")
                
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
                    start_date = end_date - timedelta(days=days_to_process)
                    print(f"No existing data found. Using default start date: {start_date.strftime('%Y-%m-%d')}")
                
                # If start date is after end date, skip this zone
                if start_date > end_date:
                    print(f"No new data to process for {zone_str}. Last data date {last_date.strftime('%Y-%m-%d')} is after or equal to latest available {end_date.strftime('%Y-%m-%d')}")
                    continue
                
                # Get IMERG files for the date range
                file_list = get_imerg_files(start_date, end_date)
                
                if not file_list:
                    print(f"No IMERG files found for {zone_str} in the date range.")
                    continue
                
                # Download IMERG files
                download_dir = download_imerg_files(file_list, imerg_store)
                
                # Process IMERG data
                imerg_data = process_imerg_data(download_dir, start_date, end_date)
                
                # Rename coordinates if needed
                imerg_data = rename_coordinates(imerg_data)
                
                # Process the zone with the IMERG data
                txt_file = process_single_zone(data_path, imerg_data, zone_str, end_date.strftime('%Y%m%d'))
                
                if txt_file:
                    output_files.append(txt_file)
                    print(f"Successfully processed {zone_str}")
                
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
    
    parser = argparse.ArgumentParser(description='Process IMERG data for hydrological modeling')
    parser.add_argument('--specific-zone', type=str, default=None, 
                      help='Process only a specific zone (default: process all zones)')
    parser.add_argument('--days-to-process', type=int, default=30, 
                      help='Maximum number of days to process if no existing data is found (default: 30)')
    
    args = parser.parse_args()
    
    # If specific_zone is provided, set process_all_zones to False
    process_all_zones = args.specific_zone is None
    
    print(f"Starting IMERG processing workflow")
    print(f"Processing {'all zones' if process_all_zones else f'zone {args.specific_zone}'}")
    print(f"Maximum days to process: {args.days_to_process}")
    
    # Note: We're not passing specific_zone if it's None
    if process_all_zones:
        result = imerg_all_zones_workflow(
            process_all_zones=True,
            days_to_process=args.days_to_process
        )
    else:
        result = imerg_all_zones_workflow(
            process_all_zones=False,
            specific_zone=args.specific_zone,
            days_to_process=args.days_to_process
        )
    
    print(f"Generated files: {result['txt_files']}")