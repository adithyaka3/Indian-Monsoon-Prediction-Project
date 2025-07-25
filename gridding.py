import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def griddata(param, level, gridsize, starttime: str, endtime: str):
    """This function grids the data of the entire world into bins so that the amount of data handled is reduced which can then be fed into ML models for analysis."""

    #Open the NetCFT file for the given file name
    # Assumed format: {param}.mon.mean.nc
    ds = xr.open_dataset(filename_or_obj=f"raw_data/{param}.mon.mean.nc", engine="netcdf4")
    var = ds[param]  

    param_units = var.attrs.get("units", "unknown")

    lon_intervals = np.arange(0, 360, 2*gridsize)
    lat_intervals = np.arange(-90, 90, gridsize)

    #select variables within the time range
    var_capped = var.sel(time=slice(starttime, endtime))
    time_intervals = var_capped.sizes["time"]


    grid_data = np.full((time_intervals, len(lat_intervals), len(lon_intervals)), np.nan)

    for i,lon in enumerate(lon_intervals):
        for j,lat in enumerate(lat_intervals):
            data = None
            if param == "mslp":
                data = var_capped.sel(lon=slice(lon, lon + 2*gridsize), lat=slice( lat + gridsize, lat)) #use for mslp
            else:
                data = var_capped.sel(level = level, lon=slice(lon, lon + 2*gridsize), lat=slice( lat + gridsize, lat)) #use for uwnd
            # Subtract monthly mean (i.e., anomaly)
            data = data.groupby("time.month") - data.groupby("time.month").mean(dim="time")
            average_value = data.mean(dim=["lon", "lat"])
            grid_data[:, j, i] = average_value.values

    #creating centers for the grid
    lat_centers = lat_intervals + gridsize/2
    lon_centers = lon_intervals + gridsize

    #create a DataArray with the gridded data
    mslp_grid = xr.DataArray(
        data=grid_data,
        dims=["time", "lat_index", "lon_index"],
        coords={
            "time": var_capped.time,
            "lat_index": np.arange(len(lat_intervals)),
            "lon_index": np.arange(len(lon_intervals)),
            "lat": ("lat_index", lat_centers),
            "lon": ("lon_index", lon_centers),
        },
        name=param,
    )
    # save the file
    if os.path.exists(f"gridded_data/{param}_grid.nc"):
        os.remove(f"gridded_data/{param}_grid.nc")
    mslp_grid.to_netcdf(f"gridded_data/{param}_grid.nc")


