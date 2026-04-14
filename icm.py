import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    def __init__(self,z_dim=32,action_dim=4,feature_dim=32):
        super(ICM,self).__init__()
        self.z_dim=z_dim

        #For Feature Encoder
        self.fc1=nn.Linear(z_dim,64)                #32->64->32
        self.fc2=nn.Linear(64,feature_dim)

        #For Forward Model
        self.fc3=nn.Linear(feature_dim+action_dim,64)     #36->64->32
        self.fc4=nn.Linear(64,feature_dim)

        #For Inverse Model
        self.fc5=nn.Linear(feature_dim+feature_dim,64)
        self.fc6=nn.Linear(64,action_dim)

    def feature_encoder(self,z):
        h=F.relu(self.fc1(z))
        phi_t=self.fc2(h)
        return phi_t

    def forward(self,z_t,a_t,z_next):
        phi_t=self.feature_encoder(z_t)
        phi_next=self.feature_encoder(z_next)

        #Forward predicts next features
        phi_hat_next=torch.cat([phi_t,a_t],dim=-1)
        phi_hat_next=F.relu(self.fc3(phi_hat_next))
        phi_hat_next=self.fc4(phi_hat_next)

        #Inverse Model
        a_hat=torch.cat([phi_t,phi_next],dim=-1)
        a_hat=F.relu(self.fc5(a_hat))
        a_hat=self.fc6(a_hat)

        #Compute loss
        forward_loss=F.mse_loss(phi_hat_next,phi_next.detach())
        inverse_loss=F.cross_entropy(a_hat,a_t.argmax(dim=-1))

        #Intrinsice Reward
        r_intrinsic=(phi_hat_next-phi_next.detach()).pow(2).sum(dim=-1).mean()
        return forward_loss,inverse_loss,r_intrinsic

# ---- Test ICM ----
z_dim = 32
action_dim = 4
batch_size = 16

icm = ICM(z_dim=z_dim, action_dim=action_dim, feature_dim=32)

z_t = torch.randn(batch_size, z_dim)
a_t = F.one_hot(torch.randint(0, 4, (batch_size,)), num_classes=4).float()
z_next = torch.randn(batch_size, z_dim)

forward_loss, inverse_loss, r_intrinsic = icm(z_t, a_t, z_next)

print("=== Shapes ===")
print(f"z_t:     {z_t.shape}")         # [16, 32]
print(f"a_t:     {a_t.shape}")         # [16, 4]
print(f"z_next:  {z_next.shape}")      # [16, 32]

print("\n=== Losses ===")
print(f"Forward loss:    {forward_loss.item():.4f}")
print(f"Inverse loss:    {inverse_loss.item():.4f}")
print(f"R intrinsic:     {r_intrinsic.item():.4f}")
print(f"All finite: {torch.isfinite(forward_loss).item() and torch.isfinite(inverse_loss).item()}")

print(f"\n=== ICM Loss ===")
beta = 0.2
icm_loss = (1 - beta) * inverse_loss + beta * forward_loss
print(f"ICM loss (0.8*inv + 0.2*fwd): {icm_loss.item():.4f}")

print(f"\nTotal params: {sum(p.numel() for p in icm.parameters()):,}")