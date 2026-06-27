import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import predictor structure
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import SimpleFusionNetwork

class MultiSourceCoatingDataset(Dataset):
    """
    A custom PyTorch Dataset that loads aligned RGB, LWIR thermal, and 3D height map samples
    from file paths and merges them into a 5-channel tensor.
    """
    def __init__(self, metadata_df, data_dir, transform=None):
        self.df = metadata_df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve paths
        rgb_path = os.path.join(self.data_dir, row['optical_path'])
        thermal_path = os.path.join(self.data_dir, row['thermal_path'])
        height_path = os.path.join(self.data_dir, row['height_path'])
        
        # Load mock inputs (substitute random data if files don't physically exist)
        if os.path.exists(rgb_path):
            # Normal load (mocked logic)
            rgb = np.zeros((1024, 1024, 3), dtype=np.float32)
        else:
            rgb = np.random.rand(1024, 1024, 3).astype(np.float32)

        if os.path.exists(thermal_path):
            thermal = np.zeros((1024, 1024, 1), dtype=np.float32)
        else:
            thermal = np.random.rand(1024, 1024, 1).astype(np.float32)

        if os.path.exists(height_path):
            height = np.zeros((1024, 1024, 1), dtype=np.float32)
        else:
            height = np.random.rand(1024, 1024, 1).astype(np.float32)

        # Concatenate channels to construct 5-Channel tensor
        fused = np.concatenate([rgb, thermal, height], axis=-1)
        
        # PyTorch format: (Channels, Height, Width)
        tensor_data = torch.tensor(fused).permute(2, 0, 1)
        
        label = int(row.get('has_defect', 0))
        
        # Mask segmentation target (mock class mask)
        mask = torch.zeros((1024, 1024), dtype=torch.long)
        if label > 0:
            # Draw a simulated circle mask in the center
            mask[400:600, 400:600] = 1 # defect class
            
        return tensor_data, label, mask

def train(config_path, epochs=3, batch_size=2, lr=0.001):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config.get("model", {}).get("num_classes", 5)
    
    print(f"Initializing Fusion Baseline Training on {device}...")
    model = SimpleFusionNetwork(in_channels=5, num_classes=num_classes)
    model.to(device)
    
    # Loss functions & optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Generate dummy batch inputs to simulate training
    print("Creating simulated training dataloader...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Simulate 5 iterations per epoch
        for batch_idx in range(5):
            inputs = torch.randn(batch_size, 5, 512, 512).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            masks = torch.randint(0, num_classes, (batch_size, 512, 512)).to(device)
            
            optimizer.zero_grad()
            cls_out, seg_out = model(inputs)
            
            loss_cls = criterion_cls(cls_out, labels)
            loss_seg = criterion_seg(seg_out, masks)
            
            loss = loss_cls + loss_seg
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / 5
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    print("Training finished! Saving baseline checkpoint model...")
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/baseline_model.pth")
    print("Checkpoint saved: outputs/baseline_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Train Pipeline")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()
    
    train(args.config, epochs=args.epochs)
