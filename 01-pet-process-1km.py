from prefect import flow, task
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import xarray as xr
from dask.distributed import Client
import geopandas as gp
import pandas as pd
import numpy as np
import glob

from utils import (
    pet_list_files_by_date,
    pet_download_extract_bilfile,
    pet_bil_netcdf,
    pet_read_netcdf_files_in_date_range,
    get_dask_client_params,
    process_zone_from_combined,
    regrid_dataset,
    zone_mean_df
)

load_dotenv()

@task
def get_current_date():
    """Get the current date in YYYYMMDD format."""
    return datetime.now().strftime('%Y%m%d')

@task
def setup_environment():
    """Set up the environment for data processing"""
    data_path = os.getenv("data_path", "./data/")  # Default to ./data/ if not set
    output_dir = f'{data_path}geofsm-input/pet/dir/'
    netcdf_path = f'{data_path}geofsm-input/pet/netcdf/'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(netcdf_path, exist_ok=True)
    
    params = get_dask_client_params()
    client = Client(**params)
    
    print(f"Environment setup complete. Using data_path: {data_path}")
    return data_path, output_dir, netcdf_path, client

def get_last_date_from_evap(zone_dir):
    """
    Read the existing evap.txt file and determine the last date in the file.
    
    Parameters:
    ----------
    zone_dir : str
        Path to the zone directory containing evap.txt
        
    Returns:
    -------
    datetime
        The last date in the file, or None if the file doesn't exist or can't be read
    """
    evap_file = os.path.join(zone_dir, 'evap.txt')
    
    if not os.path.exists(evap_file):
        print(f"No existing evap.txt found at {evap_file}")
        return None
    
    try:
        # Read the evap.txt file
        df = pd.read_csv(evap_file, sep=",")
        
        # Check if NA column exists (which contains the dates in YYYYDDD format)
        if 'NA' not in df.columns:
            print(f"Invalid format in evap.txt - missing 'NA' column")
            return None
        
        # Convert the last date to datetime
        last_date_str = df['NA'].iloc[-1]
        last_date = datetime.strptime(str(last_date_str), '%Y%j')
        
        print(f"Last date in existing evap.txt: {last_date.strftime('%Y-%m-%d')} (Day {last_date_str})")
        return last_date
        
    except Exception as e:
        print(f"Error reading existing evap.txt: {e}")
        return None

def pet_extend_forecast_improved(df, date_column, days_to_add=16):
    """
    Add a forecast extension by copying the last 15 days of data and appending it
    to create a 16-day forecast.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    date_column (str): Name of the column containing dates in 'YYYYDDD' format
    days_to_add (int): Number of days to add for forecast (default is 16)
    
    Returns:
    pd.DataFrame: DataFrame with additional forecast rows
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()
    
    # Function to safely convert date string to datetime
    def safe_to_datetime(date_str):
        try:
            return datetime.strptime(str(date_str), '%Y%j')
        except ValueError:
            return None

    # Convert date column to datetime for processing
    df['_temp_date'] = df[date_column].apply(safe_to_datetime)
    
    # Remove any rows where the date conversion failed
    df = df.dropna(subset=['_temp_date'])
    
    if df.empty:
        print(f"No valid dates found in the '{date_column}' column.")
        return df
        
    # Sort by date to ensure correct order
    df = df.sort_values('_temp_date')
    
    # Get the last 15 days of data (or fewer if less available)
    days_to_copy = min(15, len(df))
    historical_pattern = df.iloc[-days_to_copy:].copy()
    
    # Create new rows for forecast
    new_rows = []
    last_date = df['_temp_date'].iloc[-1]
    
    for i in range(days_to_add):
        # Calculate the new date
        new_date = last_date + timedelta(days=i+1)
        
        # Get corresponding historical row (cycling through the pattern)
        historical_idx = i % len(historical_pattern)
        new_row = historical_pattern.iloc[historical_idx].copy()
        
        # Update the date
        new_row['_temp_date'] = new_date
        new_rows.append(new_row)
    
    # Convert new_rows to a DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    
    # Concatenate the new rows to the original DataFrame
    result_df = pd.concat([df, new_rows_df], ignore_index=True)
    
    # Convert date column back to the original string format and remove temp column
    result_df[date_column] = result_df['_temp_date'].dt.strftime('%Y%j')
    result_df = result_df.drop(columns=['_temp_date'])
    
    return result_df

def pet_update_input_data(z1a, zone_input_path, zone_str, start_date, end_date):
    """
    Processes evaporation data and generates only the standard evap.txt and 
    zone-specific evap_zone*.txt files. Uses historical pattern for forecast extension.
    
    Parameters:
    ----------
    z1a : pandas.DataFrame
        Dataframe containing PET data that needs to be adjusted, pivoted, and formatted.
    zone_input_path : str
        Base path for input and output data files related to specific zones.
    zone_str : str
        Identifier for the specific zone, used for file naming and directory structure.
    start_date : datetime
        Start date for filtering the dataset.
    end_date : datetime
        End date for filtering the dataset.

    Returns:
    -------
    tuple
        Paths to the two generated files (standard evap.txt and zone-specific evap file).
    """
    # Ensure zone_wise directory exists
    zone_dir = f'{zone_input_path}{zone_str}'
    os.makedirs(zone_dir, exist_ok=True)
    
    # Adjust the 'pet' column by a factor of 10
    z1a['pet'] = z1a['pet'] / 10
    
    # Pivot the DataFrame
    zz1 = z1a.pivot(index='time', columns='group', values='pet')
    
    # Apply formatting to the pivoted DataFrame
    zz1 = zz1.apply(lambda row: row.map(lambda x: f'{x:.1f}' if isinstance(x, (int, float)) and pd.notna(x) else x), axis=1)
    
    # Reset the index and adjust columns
    azz1 = zz1.reset_index()
    azz1['NA'] = azz1['time'].dt.strftime('%Y%j')
    azz1.columns = [str(col) if isinstance(col, int) else col for col in azz1.columns]
    azz1 = azz1.rename(columns={'time': 'date'})
    
    # Path to standard evap.txt file in zone_wise directory
    evap_file = f'{zone_dir}/evap.txt'
    
    # Check if the evap.txt file exists
    if os.path.exists(evap_file):
        # If file exists, read and merge with new data
        try:
            ez1 = pd.read_csv(evap_file, sep=",")
            ez1['date'] = pd.to_datetime(ez1['NA'], format='%Y%j')
            
            # Create a mask for filtering data
            mask = (ez1['date'] < start_date) | (ez1['date'] > end_date)
            aez1 = ez1[mask]
            
            # Concatenate DataFrames
            bz1 = pd.concat([aez1, azz1], axis=0)
            
            # Reset index and drop unnecessary columns
            bz1.drop(['date'], axis=1, inplace=True)
            bz1.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"Error reading existing evap.txt: {e}")
            print("Creating new evap.txt file instead")
            bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    else:
        # If file doesn't exist, just use the new data
        print(f"No existing evap.txt found at {evap_file}. Creating new file.")
        bz1 = azz1.drop(['date'], axis=1).reset_index(drop=True)
    
    # Use the improved forecast extension that copies the last 15 days
    bz2 = pet_extend_forecast_improved(bz1, 'NA')
    
    # Create only the two required files
    
    # 1. Standard evap.txt file
    bz2.to_csv(evap_file, index=False)
    print(f"Created/updated standard evap.txt file: {evap_file}")
    
    # 2. Zone-specific evap file (evap_zone1.txt)
    zone_specific_file = f'{zone_dir}/evap_{zone_str}.txt'
    bz2.to_csv(zone_specific_file, index=False)
    print(f"Created zone-specific evap file: {zone_specific_file}")
    
    return evap_file, zone_specific_file

@task
def get_pet_files(url, start_date, end_date):
    """Get the list of PET files for the date range"""
    try:
        print(f"Getting PET files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        pet_list = pet_list_files_by_date(url, start_date, end_date)
        print(f"Found {len(pet_list)} PET files in date range")
        return pet_list
    except Exception as e:
        print(f"Error fetching PET files: {e}")
        raise

@task
def process_pet_files(pet_list, output_dir, netcdf_path):
    """Download and process PET files"""
    print(f"Processing {len(pet_list)} PET files")
    processed_files = 0
    
    for file_url, date in pet_list:
        try:
            date_str = date.strftime('%Y%m%d')
            nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
            
            if os.path.exists(nc_file):
                print(f"NetCDF file already exists for {date_str}, skipping download and conversion")
                processed_files += 1
                continue
            
            print(f"Processing PET file for {date_str}")
            pet_download_extract_bilfile(file_url, output_dir)
            pet_bil_netcdf(file_url, date, output_dir, netcdf_path)
            processed_files += 1
        except Exception as e:
            print(f"Error processing PET file {file_url}: {e}")
    
    print(f"Processed {processed_files} PET files")
    return processed_files

@task
def process_zone(data_path, pds, zone_str):
    """Process zone from combined shapefile and subset data"""
    master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
    
    if not os.path.exists(master_shapefile):
        print(f"Master shapefile not found: {master_shapefile}")
        raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
    
    # Standardize zone string format
    if not isinstance(zone_str, str):
        zone_str = str(zone_str)
        
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    
    print(f"Processing {zone_str} from combined shapefile")
    km_str = 1  # 1km resolution
    
    try:
        z1ds, pdsz1, zone_extent = process_zone_from_combined(master_shapefile, zone_str, km_str, pds)
        print(f"Processed zone {zone_str}")
        return z1ds, pdsz1, zone_extent
    except Exception as e:
        print(f"Error processing zone {zone_str}: {e}")
        raise

@task
def regrid_pet_data(pdsz1, zone_extent):
    """Regrid PET data to match zone resolution"""
    print("Regridding PET data")
    try:
        input_chunk_sizes = {'time': 10, 'lat': 30, 'lon': 30}
        output_chunk_sizes = {'lat': 300, 'lon': 300}
        
        # Ensure data is contiguous
        for var in pdsz1.data_vars:
            pdsz1[var] = pdsz1[var].copy(data=np.ascontiguousarray(pdsz1[var].data))
            
        return regrid_dataset(
            pdsz1,
            input_chunk_sizes,
            output_chunk_sizes,
            zone_extent,
            regrid_method="bilinear"
        )
    except Exception as e:
        print(f"Error regridding PET data: {e}")
        raise

@task
def calculate_zone_means(regridded_data, zone_ds):
    """Calculate mean PET values for each zone"""
    print("Calculating zone means")
    try:
        return zone_mean_df(regridded_data, zone_ds)
    except Exception as e:
        print(f"Error calculating zone means: {e}")
        raise

@task
def save_pet_results(results_df, data_path, zone_str, start_date, end_date):
    """Save processed PET results and update input data"""
    try:
        # Format dates to ensure they are datetime objects
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)
            
        # Create zone input path
        zone_input_path = f"{data_path}zone_wise_txt_files/"
        
        # Update PET input data - only generate the two required files
        evap_file, zone_specific_file = pet_update_input_data(
            results_df, zone_input_path, zone_str, start_date, end_date
        )
        
        print(f"PET input data updated: {evap_file} and {zone_specific_file}")
        
        return evap_file, zone_specific_file
    except Exception as e:
        print(f"Error saving PET results: {e}")
        raise

@task
def read_and_process_single_pet_file(netcdf_path, file_date):
    """Read and process a single PET netCDF file"""
    date_str = file_date.strftime('%Y%m%d')
    nc_file = os.path.join(netcdf_path, f"{date_str}.nc")
    
    if not os.path.exists(nc_file):
        print(f"Warning: NetCDF file {nc_file} does not exist")
        return None
    
    try:
        # Open the single file
        ds = xr.open_dataset(nc_file)
        
        # Process the dataset
        if 'spatial_ref' in ds.variables:
            ds = ds.drop_vars('spatial_ref')
        
        if 'band' in ds.variables:
            ds = ds.drop_vars('band')
        
        if 'date' in ds.variables:
            ds = ds.drop_vars('date')
        
        if 'band' in ds.dims:
            ds = ds.squeeze('band')
        
        if '__xarray_dataarray_variable__' in ds.data_vars:
            ds = ds.rename_vars({'__xarray_dataarray_variable__': 'pet'})
        
        # Add time dimension
        ds = ds.expand_dims(time=[file_date])
        
        # Rename coordinates if needed
        rename_dict = {}
        if 'x' in ds.dims and 'lon' not in ds.dims:
            rename_dict['x'] = 'lon'
        if 'y' in ds.dims and 'lat' not in ds.dims:
            rename_dict['y'] = 'lat'
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        return ds
    
    except Exception as e:
        print(f"Error processing {nc_file}: {e}")
        return None

@flow
def process_zone_pet_for_date(data_path, netcdf_path, zone_str, file_date):
    """Process PET data for a single zone and date"""
    try:
        # Read the single file for the date
        ds = read_and_process_single_pet_file(netcdf_path, file_date)
        
        if ds is None:
            return None, None
        
        # Process the single-date dataset for the zone
        z1ds, pdsz1, zone_extent = process_zone(data_path, ds, zone_str)
        regridded_data = regrid_pet_data(pdsz1, zone_extent)
        zone_means = calculate_zone_means(regridded_data, z1ds)
        
        # Return the results for later aggregation
        return file_date, zone_means
    
    except Exception as e:
        print(f"Error processing {zone_str} for date {file_date.strftime('%Y-%m-%d')}: {e}")
        return None, None

@flow
def process_single_zone_pet(data_path, netcdf_path, zone_str, start_date, end_date):
    """Process PET data for a single zone across multiple dates"""
    print(f"Processing zone {zone_str} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Standardize zone string format
    if not isinstance(zone_str, str):
        zone_str = str(zone_str)
        
    if zone_str.isdigit():
        zone_str = f'zone{zone_str}'
    elif not zone_str.startswith('zone'):
        zone_str = f'zone{zone_str}'
    
    try:
        # Generate a date range for all dates in the period
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Process each date individually
        all_results = []
        for file_date in date_range:
            date_str = file_date.strftime('%Y%m%d')
            print(f"Processing {zone_str} for date {date_str}")
            
            result_date, result_df = process_zone_pet_for_date(data_path, netcdf_path, zone_str, file_date)
            
            if result_date is not None and result_df is not None:
                all_results.append(result_df)
        
        # Combine results if we have any
        if all_results:
            # Concatenate all dataframes
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Save the combined results
            evap_file, zone_specific_file = save_pet_results(
                combined_results, data_path, zone_str, start_date, end_date
            )
            
            return evap_file, zone_specific_file
        else:
            print(f"No valid results found for {zone_str} in date range")
            return None, None
    
    except Exception as e:
        print(f"Error in process_single_zone_pet for {zone_str}: {e}")
        return None, None

@flow
def pet_all_zones_workflow():
    """
    Main workflow for processing PET data for all zones, starting from the last date
    in the existing evap.txt file and filling in until the latest available data.
    Uses a pattern of the last 15 days of data for the 16-day forecast extension.
    
    Returns:
        Dict containing the paths to the generated txt files
    """
    data_path, output_dir, netcdf_path, client = setup_environment()
    
    try:
        # Base URL for PET data
        url = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/pet/downloads/daily/"
        
        # Check if master shapefile exists before continuing
        master_shapefile = f'{data_path}WGS/geofsm-prod-all-zones-20240712.shp'
        if not os.path.exists(master_shapefile):
            print(f"ERROR: Master shapefile not found at {master_shapefile}")
            raise FileNotFoundError(f"Master shapefile not found: {master_shapefile}")
        else:
            print(f"Found master shapefile: {master_shapefile}")
        
        # Process all zones from the shapefile
        all_zones = gp.read_file(master_shapefile)
        unique_zones = all_zones['zone'].unique()
        
        # Initialize variables for collecting output files
        output_files = []
        
        # Process each zone separately
        for zone_str in unique_zones:
            try:
                # Standardize zone string format
                if not isinstance(zone_str, str):
                    zone_str = str(zone_str)
                    
                if zone_str.isdigit():
                    zone_str = f'zone{zone_str}'
                elif not zone_str.startswith('zone'):
                    zone_str = f'zone{zone_str}'
                
                print(f"\n===== Processing {zone_str} =====")
                
                # Check the last date in existing evap.txt for this zone
                zone_dir = f"{data_path}zone_wise_txt_files/{zone_str}"
                os.makedirs(zone_dir, exist_ok=True)
                
                last_date = get_last_date_from_evap(zone_dir)
                
                # If we have a last date, start from the next day
                # Otherwise, use a default start date (e.g., 30 days ago)
                if last_date:
                    start_date = last_date + timedelta(days=1)
                    print(f"Starting data collection from {start_date.strftime('%Y-%m-%d')}")
                else:
                    start_date = datetime.now() - timedelta(days=30)
                    print(f"No existing data found. Using default start date: {start_date.strftime('%Y-%m-%d')}")
                
                # End date is today
                end_date = datetime.now()
                
                # Get PET files for the date range
                print(f"Searching for PET files from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                pet_files = get_pet_files(url, start_date, end_date)
                
                if not pet_files:
                    print(f"No new PET files found for the date range")
                    continue
                
                print(f"Found {len(pet_files)} PET files to process")
                
                # Process all files - download and convert to NetCDF
                process_pet_files(pet_files, output_dir, netcdf_path)
                
                # Process this zone using the approach that handles each date separately
                evap_file, zone_specific_file = process_single_zone_pet(
                    data_path, netcdf_path, zone_str, start_date, end_date
                )
                
                if evap_file and zone_specific_file:
                    output_files.extend([evap_file, zone_specific_file])
                    print(f"Successfully processed {zone_str}")
                
            except Exception as e:
                print(f"Error processing {zone_str}: {e}")
        
        print(f"Workflow completed successfully! Processed {len(output_files)//2} zones")
        return {'txt_files': output_files}
    
    except Exception as e:
        print(f"Error in workflow: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PET data for hydrological modeling')
    
    args = parser.parse_args()
    
    print(f"Processing PET data from last available date forward")
    result = pet_all_zones_workflow()
    print(f"Generated files: {result['txt_files']}")