import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self,z_dim=32,hidden_dim=256,action_dim=4):
        super(Controller,self).__init__()

        self.fc=nn.Linear(z_dim+hidden_dim,action_dim)

    def forward(self,z_t,h_t):
        x=torch.cat([z_t,h_t],dim=1)
        logits=self.fc(x)
        action=torch.argmax(logits,dim=1)

        return action,logits


# ---- Test Controller ----
z_dim = 32
hidden_dim = 256
action_dim = 4
batch_size = 16

controller = Controller(z_dim, hidden_dim, action_dim)
z_t = torch.randn(batch_size, z_dim)
h_t = torch.randn(batch_size, hidden_dim)

action, logits = controller(z_t, h_t)

print("=== Shapes ===")
print(f"z_t:     {z_t.shape}")        # [16, 32]
print(f"h_t:     {h_t.shape}")        # [16, 256]
print(f"logits:  {logits.shape}")     # [16, 4]
print(f"action:  {action.shape}")     # [16]

print("\n=== Checks ===")
print(f"Actions in range [0,3]: {(action >= 0).all().item() and (action <= 3).all().item()}")
print(f"Sample actions: {action[:5].tolist()}")
print(f"Total params: {sum(p.numel() for p in controller.parameters()):,}")
