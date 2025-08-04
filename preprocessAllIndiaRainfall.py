import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch


def preprocess_rainfall_data(year, type):
    yearnum = int(year)
    jjas = 0;

    leap_year = (yearnum % 4 == 0 and yearnum % 100 != 0) or (yearnum % 400 == 0)
    months = [31, 28 if not leap_year else 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    cur = 0
    ds = xr.open_dataset(filename_or_obj="./rain_data/rain_0p25_" + year+".nc", engine="netcdf4")
    if type == 1:
        param = "RAINFALL"
        tparam = "TIME"
        latitude = "LATITUDE"
        longitude = "LONGITUDE"
    else:
        param = "rf"
        tparam = "time"
        latitude = "lat"
        longitude = "lon"
    var = ds[param] 

    param_units = var.attrs.get("units", "unknown")

    for i,month in enumerate(months):
        l = cur
        r = cur + month
        data = None
        if type == 1:
            data = var.isel(TIME=slice(l, r)).mean(dim=[tparam, latitude, longitude]).values
        else:
            data = var.isel(time=slice(l, r)).mean(dim=[tparam, latitude, longitude]).values
        
        if(i >= 5 and i <= 8):
            jjas += data
        cur += month
    
    return jjas


def get_jjas_rainfall(start_year, end_year, test):

    # Get JJAS rainfall for all years from start_year to end_year
    data = torch.load("torch_objects/jjas_percentage_deviation_south_peninsular.pt")
    start = start_year - 1901
    end = end_year - 1901

    filtered_data = data[start:end+1]

    filename = ""
    if test:
        filename = "torch_objects/test_annual_jjas_rainfall_data_south_peninsular.pt"
    else:
        filename = "torch_objects/train_annual_jjas_rainfall_data_south_peninsular.pt"
    
    torch.save(filtered_data, filename)
    print(f"JJAS rainfall data saved to {filename}")
    return filename

    all_years_rainfall = []
    for year in range(start_year, end_year + 1):
        all_years_rainfall.append(preprocess_rainfall_data(str(year), year != 2024))

    filename = ""
    if test:
        filename = "test_annual_jjas_rainfall_data.nc"
    else:
        filename = "train_annual_jjas_rainfall_data.nc"

    filetensor = f"torch_objects/{filename}.pt"
    min = np.min(all_years_rainfall)
    max = np.max(all_years_rainfall)
    print(f"JJAS rainfall data for {start_year}-{end_year} processed. Min: {min}, Max: {max}")
    rainfall_tensor = torch.tensor(all_years_rainfall, dtype=torch.float32)
    torch.save(rainfall_tensor, filetensor)

    return filetensor



