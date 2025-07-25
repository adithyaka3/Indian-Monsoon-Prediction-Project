import xarray as xr
import numpy as np
import pandas as pd
import cftime
import os

# Load dataset
def save_homogenous_rainfall_data(filename, startyear, endyear):
    """
    This script processes a NetCDF dataset containing homogenous rainfall data
    and saves each region's rainfall data from 1980 to 2024 into separate files.
    """
    ds = xr.open_dataset(f"raw_data/{filename}", decode_times=False)

    # Convert time index to cftime datetime
    months_since = ds["T"].values
    dates = cftime.num2date(
        months_since,
        units="months since 1960-01-01",
        calendar="360_day"
    )

    # Create mask for 1980-01-01 to 2024-12-31

    mask = np.array([(d.year >= int(startyear) and d.year <= int(endyear)) for d in dates])
    filtered_dates = np.array(dates)[mask]  # cftime objects
    filtered_indices = np.where(mask)[0]    # integer indices

    # Get region names
    region_names = [
        "".join(chr(c) for c in row if c > 0).strip()
        for row in ds["Name"].values
    ]

    # Extract rainfall data (DIV, T)
    rainfall = ds["PCPN"]

    # Save each region to a new file
    for i, region in enumerate(region_names):
        rain = rainfall.sel(DIV=i+1).values  # shape (T,)
        rain_filtered = rain[filtered_indices]  # shape (~540 months)

        # Create time-based DataArray
        rain_da = xr.DataArray(
            data=rain_filtered,
            coords={"time": filtered_dates},
            dims=["time"],
            name="rainfall",
            attrs={
                "units": "mm",
                "long_name": f"Monthly rainfall in {region} (1980–2024)"
            },
        )

        # Save to new NetCDF file
        ds_out = xr.Dataset({"rainfall": rain_da})
        filename = f"homogenous_rain_data/{region.replace(' ', '_').lower()}_rainfall.nc"
        if os.path.exists(filename):
            os.remove(filename)
        ds_out.to_netcdf(filename)
        print(f"Saved {filename}")



