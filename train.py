import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext import create_mednext_v1
import data_loader
import yaml
import argparse 
    
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('cfig_path',  type = str)
parser.add_argument('--phase', default = 'train', type = str)
args = parser.parse_args()

cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ data loader -----------------#
loaders = data_loader.GetLoader(cfig = cfig['loader_params'])
train_loader =loaders.train_dataloader()
val_loader = loaders.val_dataloader()

# ------------- Network ------------------ # 
model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
  num_classes = cfig['model_params']['out_channels'],
  model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
  kernel_size = cfig['model_params']['kernel_size'],   # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = cfig['model_params']['deep_supervision']   
).to(device)

# ------------ loss -----------------------# 
optimizer = optim.Adam(model.parameters(), lr=cfig['lr'])
criterion = nn.L1Loss()

# -----------Training loop --------------- #
for epoch in range(cfig['num_epochs']):
    model.train()
    for i, data_dict in enumerate(train_loader):
        # Forward pass
        outputs = model(data_dict['data'].to(device))
        loss = criterion(outputs, data_dict['label'].to(device))
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{cfig['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
