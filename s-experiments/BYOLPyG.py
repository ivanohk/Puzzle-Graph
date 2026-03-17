import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import BatchNorm
import copy

from torch_geometric.utils import dropout_edge
import torch.nn.functional as F
import torch.optim as optim
import math

from torch_geometric.datasets import Planetoid

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 1. Twin Architectures
class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        # Message Passing
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        # We return the node-level representations directly
        out = self.conv2(h, edge_index)
        return out
    

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class BYOLGeometric(nn.Module):
    def __init__(self, encoder, rep_dim, proj_hidden, proj_dim, pred_hidden):
        super().__init__()

        # Online Network
        self.online_encoder = encoder
        self.online_projector = MLP(in_dim=rep_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.online_predictor = MLP(in_dim=proj_dim, hidden_dim=pred_hidden, out_dim=proj_dim)

        # Target Network
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # No Backpropagation
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward_online(self, x, edge_index):
        representation = self.online_encoder(x, edge_index)
        projection = self.online_projector(representation)
        prediction = self.online_predictor(projection)
        return prediction
    
    @torch.no_grad()
    def forward_target(self, x, edge_index):
        representation = self.target_encoder(x, edge_index)
        projection = self.target_projector(representation)
        return projection
    
    def forward(self, x1, edge_index1, x2, edge_index2):        
        # 1. Online sees View 1, Target sees View 2
        pred_1 = self.forward_online(x1, edge_index1)
        with torch.no_grad():
            targ_2 = self.forward_target(x2, edge_index2)
            targ_2 = targ_2.detach()

        # 2. Online sees View 2, Target sees View 1
        pred_2 = self.forward_online(x2, edge_index2)
        with torch.no_grad():
            targ_1 = self.forward_target(x1, edge_index1)
            targ_1 = targ_1.detach()

        return pred_1, targ_2, pred_2, targ_1
    
    @torch.no_grad()
    def update_target_network(self, curr_step, tot_steps, base_tau=0.99):
        # Dynamic cosine update
        tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * curr_step / tot_steps) + 1.0) / 2.0
        # EMA
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = tau * target_param.data + (1.0 - tau) * online_param.data

        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = tau * target_param.data + (1.0 - tau) * online_param.data


# 2. Graph Augmentations
def augment_graph(x, edge_index, drop_feat_rate=0.2, drop_edge_rate=0.2):
    # Node Feature Masking
    # Random matrix with the exact same shape as x (Nodes x Features)
    feat_mask = torch.rand_like(x, device=x.device) < drop_feat_rate

    x_aug = x.clone()

    x_aug[feat_mask] = 0

    # Edge Dropping
    edge_index_aug, _ = dropout_edge(edge_index, p=drop_edge_rate, force_undirected=True, training=True)

    return x_aug, edge_index_aug


# 3. Forward Pass Pipeline (Handled inside BYOLGeometric)

# 4. Asymmetric Loss
def byol_loss(p, z):
    z = z.detach()

    # FIXED: Changed dim=1 to dim=-1. Safer for varying batch sizes.
    similarity = F.cosine_similarity(p, z, dim=-1)

    loss = 2 - 2 * similarity.mean()

    return loss

# 5. Exponential Moving Average


# 6. Inference and execution
if __name__ == "__main__":
    print("Downloading/Loading Cora dataset...")
    # This will automatically download Cora to a local folder named 'data'
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]  # The entire citation network is in data[0]

    # Setup pre training
    print("Setting up BYOL for Node Classification...")
    # Cora has 1433 features per node
    encoder = GNNEncoder(in_channels=dataset.num_node_features, hidden_channels=256, out_channels=128)
    
    # Initialize BYOL with node-level dimensions
    model = BYOLGeometric(encoder, rep_dim=128, proj_hidden=256, proj_dim=128, pred_hidden=256)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    byol_losses = []

    # Pre training loop
    total_epochs = 200
    print(f"Running BYOL Pre-Training ({total_epochs} Epochs)...")

    model.train()
    for epoch in range(1, total_epochs + 1):
        optimizer.zero_grad()

        # Augment the entire Cora graph twice
        x1, edge_index1 = augment_graph(data.x, data.edge_index, drop_feat_rate=0.3, drop_edge_rate=0.3)
        x2, edge_index2 = augment_graph(data.x, data.edge_index, drop_feat_rate=0.3, drop_edge_rate=0.3)

        # Forward pass (outputs are now node-level predictions/projections)
        p1, z2, p2, z1 = model(x1, edge_index1, x2, edge_index2)

        # Calculate symmetric loss across all nodes
        loss = byol_loss(p1, z2) + byol_loss(p2, z1)
        loss.backward()
        optimizer.step()

        byol_losses.append(loss.item())

        # Update Target Network
        model.update_target_network(curr_step=epoch, tot_steps=total_epochs, base_tau=0.99)

        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | BYOL Loss: {loss.item():.4f}")

    print("Generating t-SNE visualization of BYOL embeddings...")
    # Extract embeddings without tracking gradients
    model.eval()
    with torch.no_grad():
        # Pass the clean graph through the trained encoder
        raw_embeddings = model.online_encoder(data.x, data.edge_index)
    
    # Compress from 128 dimensions down to 2 dimensions for plotting
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(raw_embeddings.cpu().numpy())
    labels = data.y.cpu().numpy()

    # Plot the clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                          c=labels, cmap='tab10', s=15, alpha=0.8)
    
    plt.title("t-SNE of BYOL Node Embeddings (Pre-Tuning)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # Add a legend for the 7 Cora classes
    plt.legend(handles=scatter.legend_elements()[0], 
               labels=[f"Class {i}" for i in range(dataset.num_classes)],
               title="Paper Categories")
    plt.show()

    # Fine-tuning
    print("\nSetting up Fine-Tuning Protocol...")
    pretrained_encoder = copy.deepcopy(model.online_encoder)

    # Ensure weights are unfrozen for fine-tuning
    for param in pretrained_encoder.parameters():
        param.requires_grad = True

    class FineTuneClassifier(nn.Module):
        def __init__(self, encoder, hidden_dim, num_classes):
            super().__init__()
            self.encoder = encoder
            self.classifier_head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, edge_index):
            embeddings = self.encoder(x, edge_index)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            logits = self.classifier_head(embeddings)
            return logits

    # Cora has 7 classes (paper categories)
    fine_tune_model = FineTuneClassifier(pretrained_encoder, hidden_dim=128, num_classes=dataset.num_classes)
    optimizer_ft = optim.Adam([
        {'params': fine_tune_model.encoder.parameters(), 'lr': 1e-4, 'weight_decay': 5e-4},       
        {'params': fine_tune_model.classifier_head.parameters(), 'lr': 0.01, 'weight_decay': 5e-4} 
    ])

    print(f"Running Fine-Tuning ({total_epochs} Epochs)...")
    for epoch in range(1, total_epochs + 1):
        # 1. Training phase
        fine_tune_model.train()
        optimizer_ft.zero_grad()
        
        # Single forward pass
        logits = fine_tune_model(data.x, data.edge_index)
        
        # Calculate loss only on the training nodes
        loss_ft = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        
        loss_ft.backward()
        optimizer_ft.step()
        
        # Calculate Training Accuracy "for free" using the existing logits
        train_preds = logits[data.train_mask].argmax(dim=1)
        train_correct = (train_preds == data.y[data.train_mask]).sum().item()
        train_acc = train_correct / data.train_mask.sum().item()
        
        # 2. Evaluation phase
        if epoch % 25 == 0:
            fine_tune_model.eval()
            with torch.no_grad():
                # Clean forward pass (no dropout) for the test set
                eval_logits = fine_tune_model(data.x, data.edge_index)
                
                # Calculate test accuracy
                test_preds = eval_logits[data.test_mask].argmax(dim=1)
                test_correct = (test_preds == data.y[data.test_mask]).sum().item()
                test_acc = test_correct / data.test_mask.sum().item()
            
            print(f"Epoch {epoch:03d} | Train Loss: {loss_ft.item():.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    print("\nCora evaluation complete!")

    # Plot BYOL Loss
    plt.figure(figsize=(8, 5))
    plt.plot(byol_losses, label='BYOL Asymmetric Loss', color='blue', linewidth=2)
    plt.title("Self-Supervised Pre-Training Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
