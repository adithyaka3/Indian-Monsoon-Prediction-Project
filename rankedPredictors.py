import torch
import os

def pearson_corr(x: torch.Tensor, y: torch.Tensor):
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")

    x_mean = x.mean()
    y_mean = y.mean()
    xm = x - x_mean
    ym = y - y_mean
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2)) * torch.sqrt(torch.sum(ym ** 2))
    if r_den == 0:
        return torch.tensor(0.0)
    r = r_num / r_den
    return r


def getRankedPredictors(filetensor, variables):

    """ Rank all the predictors from each variable encoded data. If the encoding is coming from 2 variables, then separate them as var1_var2"""

    rain_data = torch.load(filetensor)  # Load the rainfall data tensor


    for var in variables:
        print(f"--- Processing variable: {var} ---")
        for layer in range(1, 4):
            layer_features = torch.load(f"torch_objects/encoded_h{layer}_{var}.pt")
            layer_features = layer_features.T


            all_correlations = []
            for p,predictor in enumerate(layer_features):
                for lead in range(0, 13):
                    cur_month = 5-lead
                    var_array = []
                    lag = cur_month < 0
                    cur_month = (cur_month + 12) % 12
                    for j in range(cur_month, predictor.shape[0], 12):
                        var_array.append(predictor[j])

                    if lag:
                        var_array = var_array[:-1]
                        corr = pearson_corr(torch.tensor(var_array), rain_data[1:])
                    else:
                        corr = pearson_corr(torch.tensor(var_array), rain_data)
                    all_correlations.append([p, lead, corr.item()])


                #sort in descending order of correlation ie the 4th element of all_correlations
            all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

            #need to create 5 prediction sets by choosing top 4 5 6 8 10
            prediction_sets_sizes = [4, 5, 6, 8, 10]
            prediction_set = []
            for k in prediction_sets_sizes:
                print(f"--- Top {k} predictors for variable {var} and file encoded_h{layer}_{var}.pt---")
                predictors = [all_correlations[i][0] for i in range(k)]
                leads = [all_correlations[i][1] for i in range(k)]
                leads = [5 - lead for lead in leads]
                leads = [(lead + 12) % 12 for lead in leads]
                top_correlations = [all_correlations[i][2] for i in range(k)]
                
                features = [layer_features[predictors[i]][torch.arange(leads[i], layer_features.shape[1], step= 12, dtype = torch.long)] for i in range(k)]
                features_torch = torch.stack(features)
                features_torch = features_torch.T
                print("Top Correlations:", top_correlations) 
                #save the features tensor
                torch.save(features_torch, f"torch_objects/features_h{layer}_{var}_top_{k}_predictors.pt")
                print()






