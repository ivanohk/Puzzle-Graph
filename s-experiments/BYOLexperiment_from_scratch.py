import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt


# 1. Custom Convolutional Backbone
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 2. Multi-Layer Perceptron 
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3. Loss Function
def manual_loss(p, z):
    # Calcolo norma
    p_norm = p.norm(p=2, dim=1, keepdim=True)
    z_norm = z.norm(p=2, dim=1, keepdim=True)

    # Normalizzo 
    p_normalized = p / (p_norm + 1e-8) # Aggiungo Epsilon per evitare divisioni per zero
    z_normalized = z / (z_norm + 1e-8)

    # Prodotto scalare (cosine similarity)
    dot_product = (p_normalized * z_normalized).sum(dim=1)

    loss = 2 - 2 * dot_product.mean()
    return loss

# 4. BYOL Architecture
class BYOLFromScratch(nn.Module):
    def __init__(self, image_channels=3, feature_dim=512, hidden_dim=4096, proj_dim=256, m=0.99):
        super().__init__()
        self.m = m

        # ONLINE NETWORK (gradienti)
        self.online_encoder = SimpleConvNet(in_channels=image_channels, out_dim=feature_dim)
        self.online_projector = MLP(feature_dim, hidden_dim, proj_dim)
        self.online_predictor = MLP(proj_dim, hidden_dim, proj_dim)

        # TARGET NETWORK (EMA)
        self.target_encoder = SimpleConvNet(in_channels=image_channels, out_dim=feature_dim)
        self.target_projector = MLP(feature_dim, hidden_dim, proj_dim)

        # Inizializzazione pesi Target per renderli uguali a quelli Online
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())

        # Gradienti di Target disattivati
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        # Implementazione Exponential Moving Average (EMA)
        
        # Aggiornamento encoder
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

        # Aggiornamento proiettore
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)


    def forward(self, view1, view2, visualize=False, step=None):
        # Passaggio Online (con gradiente)
        online_enc1 = self.online_encoder(view1)
        online_enc2 = self.online_encoder(view2)
        z1 = self.online_projector(online_enc1)
        z2 = self.online_projector(online_enc2)
        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        # Passaggio Target (no gradiente)
        with torch.no_grad():
            self.update_target_network()
            target_z1 = self.target_projector(self.target_encoder(view1))
            target_z2 = self.target_projector(self.target_encoder(view2))

        # Visualizzazione istogrammi
        if visualize and step is not None:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs[0, 0].hist(p1[0].detach().cpu().numpy(), bins=30, color='blue', alpha=0.7)
            axs[0, 0].set_title(f'p1 embedding (step {step})')
            axs[0, 1].hist(p2[0].detach().cpu().numpy(), bins=30, color='orange', alpha=0.7)
            axs[0, 1].set_title(f'p2 embedding (step {step})')
            axs[1, 0].hist(z1[0].detach().cpu().numpy(), bins=30, color='green', alpha=0.7)
            axs[1, 0].set_title(f'z1 projection (step {step})')
            axs[1, 1].hist(z2[0].detach().cpu().numpy(), bins=30, color='red', alpha=0.7)
            axs[1, 1].set_title(f'z2 projection (step {step})')
            plt.tight_layout()
            plt.show()

        # Calcolo loss (asimmetrica e incrociata)
        loss_1 = manual_loss(p1, target_z2.detach())
        loss_2 = manual_loss(p2, target_z1.detach())

        return loss_1 + loss_2

# 5. Inizializzazione

# Scelta device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inizializzo il modello custom
model = BYOLFromScratch().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Genero due tensori random per simulare due batch di 64 immagini distorte
view_1 = torch.rand(64, 3, 64, 64).to(device)
view_2 = torch.rand(64, 3, 64, 64).to(device)

losses = []
model.train()

for step in range(10):
    optimizer.zero_grad()

    # Forward pass e calcolo della loss, con visualizzazione
    loss = model(view_1, view_2, visualize=True, step=step+1)

    # Backward pass e ottimizzazione
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f"Step {step + 1} | Loss: {loss.item():.4f}")

plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss trend")
# plt.show()

# 6. Training Loop