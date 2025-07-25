import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataload(var1, var2):

    file1 = f"gridded_data/{var1}_grid.nc"
    file2 = f"gridded_data/{var2}_grid.nc"

    #data1
    ds = xr.open_dataset(file1)
    data = ds[var1].values

    #converting into tensor
    data = torch.tensor(data, dtype=torch.float32)
    #normalizing
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to avoid division by 0

    data = data.reshape(data.shape[0], -1)  # Flatten the 18x18 into a single dimension
    time = data.shape[0]

    #data2
    ds = xr.open_dataset(file2)
    data2 = ds[var2].values

    #converting into tensor
    data2 = torch.tensor(data2, dtype=torch.float32)
    #normalizing
    min_val = data2.min()
    max_val = data2.max()
    data2 = (data2 - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to avoid division by 0

    data2 = data2.reshape(data2.shape[0], -1)  # Flatten the 18x18 into a single dimension
    time2 = data2.shape[0]

    return data, time, data2, time2

class LinearAutoencoder_comb(nn.Module):
    def __init__(self):
        super(LinearAutoencoder_comb, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(648, 194),
            nn.Tanh(),
            nn.Linear(194, 58),
            nn.Tanh(),
            nn.Linear(58, 17),  # Bottleneck layer
        )

        # Decoder (adjusted to return (1, 18, 18))
        self.decoder = nn.Sequential(
            nn.Linear(17, 58),
            nn.Tanh(),
            nn.Linear(58, 194),
            nn.Tanh(),
            nn.Linear(194, 648),  # Output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def thresholding(model, dataloader, var1, var2):

    W1 = model.encoder[0].weight.data.clone()
    b1 = model.encoder[0].bias.data.clone()
    W2 = model.encoder[2].weight.data.clone()
    b2 = model.encoder[2].bias.data.clone()
    W3 = model.encoder[4].weight.data.clone()
    b3 = model.encoder[4].bias.data.clone()

    def get_thresholded_w(W, std_factor=2.0):
        means = W.mean(dim=1, keepdim=True)
        stds = W.std(dim=1, keepdim=True)
        thresholds = means + std_factor * stds
        mask = W > thresholds
        W_thresholded = W * mask.float()
        assert W_thresholded.shape == W.shape, "Thresholded weights shape mismatch"
        return W_thresholded

    def thresholded_forward(x, W1, b1, W2, b2, W3, b3):
        # Layer 1
        h1 = torch.matmul(x, W1.T) + b1
        h1 = torch.tanh(h1)

        # Layer 2
        h2 = torch.matmul(h1, W2.T) + b2
        h2 = torch.tanh(h2)

        # Layer 3
        h3 = torch.matmul(h2, W3.T) + b3
        h3 = torch.tanh(h3)

        return h1, h2, h3

    W1 = get_thresholded_w(W1)
    W2 = get_thresholded_w(W2)
    W3 = get_thresholded_w(W3)

    all_h1 = []
    all_h2 = []
    all_h3 = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            h1, h2, h3 = thresholded_forward(
                inputs,
                W1.to(device), b1.to(device), 
                W2.to(device), b2.to(device), 
                W3.to(device), b3.to(device),
            )
            all_h1.append(h1.cpu())
            all_h2.append(h2.cpu())
            all_h3.append(h3.cpu())

    encoded_dataset_h1 = torch.cat(all_h1, dim=0)
    encoded_dataset_h2 = torch.cat(all_h2, dim=0)
    encoded_dataset_h3 = torch.cat(all_h3, dim=0)

    # Save the encoded datasets
    torch.save(encoded_dataset_h1, f"torch_objects/encoded_h1_{var1}_{var2}.pt")
    torch.save(encoded_dataset_h2, f"torch_objects/encoded_h2_{var1}_{var2}.pt")
    torch.save(encoded_dataset_h3, f"torch_objects/encoded_h3_{var1}_{var2}.pt")

def getEncodeddataDouble(var1, var2, epochs):
    """This function loads the gridded data and then uses an autoencoder to encode the data into a lower dimension."""
    # Load the dataset and create a DataLoader
    data, time, data2, time2 = dataload(var1, var2)

    data = torch.cat((data, data2), dim=1)  # Concatenate along the feature dimension
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LinearAutoencoder_comb().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
                inputs = batch[0].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader.dataset):.4f}")
    
    # Thresholding the weights and getting the encoded data
    thresholding(model, dataloader, var1, var2)
    


