from gridding import griddata
from singleAutoencoder import getEncodeddata, testGetEncodeddata
from doubleAutoencoder import getEncodeddataDouble
from preprocessAllIndiaRainfall import get_jjas_rainfall
from rankedPredictors import getRankedPredictors, testGetRankedPredictors
from predict import prediction

param = "mslp"  # Example parameter, can be changed as needed
#First step is to grid the data into bins

print("Gridding the data...")
griddata(param = param, level = 200, gridsize= 10, starttime = "1980-01-01", endtime = "2010-12-31")
print("Gridding completed.")

#second step if to process the rainfall data. 
# For all india rainfall, we need to get the JJAS rainfall for all years
print("Processing rainfall data...")
rain_file_tensor = get_jjas_rainfall(start_year=1980, end_year=2010, test = False) # Inclusive of the end year, i.e 31-12-end_year
print("Rainfall data processed.")

#Next step is to encode the data using the autoencoder
print("Encoding the data using single autoencoder...")
W1, b1, W2, b2, W3, b3 = getEncodeddata(variable_name=param, epochs=200)
print("Encoding completed using single autoencoder.")

#Next step is to get the ranked predictors
print("Getting ranked predictors...")
all_correlations = getRankedPredictors(rain_file_tensor, variables=[param])
print("Ranked predictors obtained.")



#TESTING:
print("Test gridding the data...")
griddata(param = param, level = 200, gridsize= 10, starttime = "2011-01-01", endtime = "2015-12-31")
print("Test gridding completed.")

print("Test rainfall data processing...")
test_rain_file_tensor = get_jjas_rainfall(start_year=2011, end_year=2015, test = True) # Inclusive of the end year, i.e 31-12
print("Test rainfall data processed.")

print("Test encoding the data using single autoencoder...")
testGetEncodeddata(W1, b1, W2, b2, W3, b3, variable_name=param)
print("Test encoding completed using single autoencoder.")

print("Test ranked predictors...")
testGetRankedPredictors(all_correlations, variables=[param])
print("Test ranked predictors obtained.")


#Next step is to predict the rainfall using the ranked predictors
print("Predicting rainfall using ranked predictors...")
layer = 3
top = 5
for layer in range(1, 4):
    for top in [4, 5, 6, 8, 10]:
        print(f"Predicting using layer {layer} and top {top} predictors...")
        # Load the feature tensors for training and testing
        train_feature_data_path = f"torch_objects/train_features_h{layer}_{param}_top_{top}_predictors.pt"
        test_feature_data_path = f"torch_objects/test_features_h{layer}_{param}_top_{top}_predictors.pt"

        prediction(rain_file_tensor, test_rain_file_tensor, train_feature_data_path, test_feature_data_path)
        print("------------------------------------------------")

print("Prediction completed.")





