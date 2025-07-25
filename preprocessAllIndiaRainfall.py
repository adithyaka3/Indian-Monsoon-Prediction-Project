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


def get_jjas_rainfall(start_year, end_year):

    all_years_rainfall = []
    for year in range(start_year, end_year + 1):
        all_years_rainfall.append(preprocess_rainfall_data(str(year), year != 2024))

    years = np.arange(start_year, end_year +1)
    # Create xarray DataArray
    rainfall_da = xr.DataArray(
        data=all_years_rainfall,
        coords={"year": years},
        dims=["year"],
        name="rainfall",  # Name of the variable
        attrs={"units": "mm", "description": "Annual Rainfall over India"}
    )

    # Create Dataset
    ds = xr.Dataset({"rainfall": rainfall_da})

    # Save to NetCDF
    filename = "all_india_rainfall/annual_jjas_rainfall_data.nc"
    if os.path.exists(filename):
        os.remove(filename)
    ds.to_netcdf(filename)

    filetensor = "torch_objects/annual_jjas_rainfall_data.pt"
    rainfall_tensor = torch.tensor(all_years_rainfall, dtype=torch.float32)
    torch.save(rainfall_tensor, "torch_objects/annual_jjas_rainfall_data.pt")

    return filetensor



