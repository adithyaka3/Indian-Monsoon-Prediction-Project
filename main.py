from gridding import griddata
from singleAutoencoder import getEncodeddata
from doubleAutoencoder import getEncodeddataDouble
from preprocessAllIndiaRainfall import get_jjas_rainfall
from rankedPredictors import getRankedPredictors
from predict import prediction

#First step is to grid the data into bins
print("Gridding the data...")
griddata(param = "mslp", level = 200, gridsize= 10, starttime = "1980-01-01", endtime = "2020-12-31")
print("Gridding completed.")

#second step if to process the rainfall data. 
# For all india rainfall, we need to get the JJAS rainfall for all years
print("Processing rainfall data...")
rain_file_tensor = get_jjas_rainfall(start_year=1980, end_year=2020) # Inclusive of the end year, i.e 31-12-end_year
print("Rainfall data processed.")

#Next step is to encode the data using the autoencoder
print("Encoding the data using single autoencoder...")
getEncodeddata(variable_name="mslp", epochs=500)
print("Encoding completed using single autoencoder.")

#Next step is to get the ranked predictors
print("Getting ranked predictors...")
getRankedPredictors(rain_file_tensor, variables=["mslp"])
print("Ranked predictors obtained.")

#Next step is to predict the rainfall using the ranked predictors
print("Predicting rainfall using ranked predictors...")
prediction(rain_file_tensor, "torch_objects/features_h3_mslp_top_5_predictors.pt")
print("Prediction completed.")





