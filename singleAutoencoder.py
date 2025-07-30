import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloading(variable_name):
    file_path = f"gridded_data/{variable_name}_grid.nc"
    # Load and normalize NetCDF variable
    ds = xr.open_dataset(file_path)
    data = ds[variable_name].values

    #converting into tensor
    data = torch.tensor(data, dtype=torch.float32)

    #normalizing
    min_val = data.min()
    max_val = data.max()
    data = (data - min_val) / (max_val - min_val + 1e-8)  # Avoid division by zero

    data = data.reshape(data.shape[0], -1)  # Flatten the 18x18 into a single dimension
    time = data.shape[0]
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(324, 97),
            nn.Tanh(),
            nn.Linear(97, 29),
            nn.Tanh(),
            nn.Linear(29, 9),  # Bottleneck layer
            nn.Tanh(),
        )

        # Decoder (adjusted to return (1, 18, 18))
        self.decoder = nn.Sequential(
            nn.Linear(9, 29),
            nn.Tanh(),
            nn.Linear(29, 97),
            nn.Tanh(),
            nn.Linear(97, 324),  # Output layer
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def thresholding(model, dataloader, variable_name):
    
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
    torch.save(encoded_dataset_h1, f"torch_objects/train_encoded_h1_{variable_name}.pt")
    torch.save(encoded_dataset_h2, f"torch_objects/train_encoded_h2_{variable_name}.pt")
    torch.save(encoded_dataset_h3, f"torch_objects/train_encoded_h3_{variable_name}.pt")

    return W1, b1, W2, b2, W3, b3

def getEncodeddata(variable_name, epochs):
    """This function loads the gridded data and then uses an autoencoder to encode the data into a lower dimension."""
    # Load the dataset and create a DataLoader
    dataloader = dataloading(variable_name)

    model = LinearAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset = dataloader.dataset

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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/ len(dataset):.4f}")
    
    return thresholding(model, dataloader, variable_name)

def testGetEncodeddata(W1, b1, W2, b2, W3, b3, variable_name):
    dataloader = dataloading(variable_name)
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
    torch.save(encoded_dataset_h1, f"torch_objects/test_encoded_h1_{variable_name}.pt")
    torch.save(encoded_dataset_h2, f"torch_objects/test_encoded_h2_{variable_name}.pt")
    torch.save(encoded_dataset_h3, f"torch_objects/test_encoded_h3_{variable_name}.pt")
