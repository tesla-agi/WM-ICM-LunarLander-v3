import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,input_dim,z_dim):
        super(VAE,self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        #Encoder
        self.fc1=nn.Linear(input_dim,64)
        self.fc2=nn.Linear(64,128)
        self.fc3=nn.Linear(128,256)

        self.mu=nn.Linear(256,z_dim)
        self.log_var=nn.Linear(256,z_dim)

        #Decoder
        self.fc4=nn.Linear(z_dim,256)
        self.fc5=nn.Linear(256,128)
        self.fc6=nn.Linear(128,64)
        self.fc7=nn.Linear(64,input_dim)

    def encode(self,x):
        h=F.relu(self.fc1(x))
        h=F.relu(self.fc2(h))
        h=F.relu(self.fc3(h))

        mu=self.mu(h)
        log_var=self.log_var(h)
        return mu,log_var

    def reparameterize(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)
        z=mu+std*eps
        return z

    def decode(self,z):
        h=F.relu(self.fc4(z))
        h=F.relu(self.fc5(h))
        h=F.relu(self.fc6(h))

        out=self.fc7(h)
        return out

    def forward(self,x):
        mu,log_var=self.encode(x)
        z=self.reparameterize(mu,log_var)
        x_hat=self.decode(z)
        return x_hat,mu,log_var


def vae_loss(x,x_hat,mu,log_var,beta=1.0):
    recon_loss = F.mse_loss(x_hat,x,reduction='sum')/x.size(0)
    kl=-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())/x.size(0)
    total=recon_loss+beta*kl
    return total,recon_loss,kl

'''
# During training:
for epoch in range(num_epochs):
    beta = min(1.0, epoch / 100)  # warmup over first 100 epochs
    total, recon, kl = vae_loss(x, x_hat, mu, log_var, beta=beta)
'''

# ---- Test VAE ----
input_dim = 8
z_dim = 32
batch_size = 16

vae = VAE(input_dim=input_dim, z_dim=z_dim)
dummy = torch.randn(batch_size, input_dim)

# Forward pass
x_hat, mu, log_var = vae(dummy)

# Loss
beta = 0.5
total, recon, kl = vae_loss(dummy, x_hat, mu, log_var, beta=beta)

# Print shapes
print("=== Shapes ===")
print(f"Input:    {dummy.shape}")       # [16, 8]
print(f"mu:       {mu.shape}")          # [16, 32]
print(f"log_var:  {log_var.shape}")     # [16, 32]
print(f"z:        {vae.reparameterize(mu, log_var).shape}")  # [16, 32]
print(f"x_hat:    {x_hat.shape}")       # [16, 8]

# Print losses
print("\n=== Losses ===")
print(f"Recon loss: {recon.item():.4f}")
print(f"KL loss:    {kl.item():.4f}")
print(f"Beta:       {beta}")
print(f"Total loss: {total.item():.4f}")

# Verify dimensions match
print("\n=== Checks ===")
print(f"Input == Output dim: {dummy.shape == x_hat.shape}")
print(f"Latent dim is 32:    {mu.shape[1] == 32}")
print(f"Total params: {sum(p.numel() for p in vae.parameters()):,}")