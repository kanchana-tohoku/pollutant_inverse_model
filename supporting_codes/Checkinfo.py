import xarray as xr

file = "C:/Users/kanch/Hourly_Data/sntr/ERA5_sntr_2016_04.nc"
ds = xr.open_dataset(file, engine='netcdf4')  # or engine='h5netcdf'
print(ds)
