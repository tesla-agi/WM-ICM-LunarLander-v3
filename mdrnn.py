import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSTM(nn.Module):
    def __init__(self,z_dim,action_dim,hidden_dim):
        super(LSTM,self).__init__()

        input_dim=z_dim+action_dim
        self.hidden_dim=hidden_dim

        #Input Weights
        self.xf=nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.xi=nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.xc=nn.Linear(input_dim+hidden_dim,hidden_dim)
        self.xo=nn.Linear(input_dim+hidden_dim,hidden_dim)

    def forward(self,x_t,h_prev,c_prev):
        combined=torch.cat([x_t,h_prev],dim=-1)
        f_t=torch.sigmoid(self.xf(combined))
        i_t=torch.sigmoid(self.xi(combined))
        g_t=torch.tanh(self.xc(combined))
        o_t=torch.sigmoid(self.xo(combined))

        c_t=f_t*c_prev+i_t*g_t
        h_t=o_t*torch.tanh(c_t)
        return h_t,c_t

'''

# ---- Test LSTM ----
z_dim = 32
action_dim = 4
hidden_dim = 256
batch_size = 16

lstm = LSTM(z_dim=z_dim, action_dim=action_dim, hidden_dim=hidden_dim)

x_t = torch.randn(batch_size, z_dim + action_dim)  # (16, 36)
h_prev = torch.zeros(batch_size, hidden_dim)         # (16, 256)
c_prev = torch.zeros(batch_size, hidden_dim)         # (16, 256)

h_t, c_t = lstm(x_t, h_prev, c_prev)

print("=== Shapes ===")
print(f"Input x_t:  {x_t.shape}")      # [16, 36]
print(f"h_prev:     {h_prev.shape}")    # [16, 256]
print(f"c_prev:     {c_prev.shape}")    # [16, 256]
print(f"h_t:        {h_t.shape}")       # [16, 256]
print(f"c_t:        {c_t.shape}")       # [16, 256]

print("\n=== Checks ===")
print(f"h_t range: [{h_t.min().item():.4f}, {h_t.max().item():.4f}]")  # should be in (-1, 1)
print(f"c_t range: [{c_t.min().item():.4f}, {c_t.max().item():.4f}]")
print(f"Total params: {sum(p.numel() for p in lstm.parameters()):,}")


'''

class MDN(nn.Module):
    def __init__(self,hidden_dim,z_dim,num_mixtures):
        super(MDN,self).__init__()

        self.num_mixtures = num_mixtures
        self.z_dim=z_dim

        self.pi=nn.Linear(hidden_dim,num_mixtures)
        self.mu=nn.Linear(hidden_dim,num_mixtures*z_dim)
        self.sigma=nn.Linear(hidden_dim,num_mixtures*z_dim)

    def forward(self,h_t):
        pi=torch.softmax(self.pi(h_t),dim=-1)
        mu=self.mu(h_t).view(-1,self.num_mixtures,self.z_dim)
        sigma = F.softplus(self.sigma(h_t)) + 1e-6
        sigma = sigma.view(-1, self.num_mixtures,self.z_dim)

        return pi,mu,sigma

'''
# ---- Test MDN ----
hidden_dim = 256
z_dim = 32
num_mixtures = 5
batch_size = 16

mdn = MDN(hidden_dim=hidden_dim, z_dim=z_dim, num_mixtures=num_mixtures)
h_t = torch.randn(batch_size, hidden_dim)

pi, mu, sigma = mdn(h_t)

print("=== Shapes ===")
print(f"h_t:    {h_t.shape}")       # [16, 256]
print(f"pi:     {pi.shape}")        # [16, 5]
print(f"mu:     {mu.shape}")        # [16, 5, 32]
print(f"sigma:  {sigma.shape}")     # [16, 5, 32]

print("\n=== Checks ===")
print(f"pi sums to 1: {pi[0].sum().item():.4f}")
print(f"pi all positive: {(pi > 0).all().item()}")
print(f"sigma all positive: {(sigma > 0).all().item()}")
print(f"sigma range: [{sigma.min().item():.4f}, {sigma.max().item():.4f}]")
print(f"Total params: {sum(p.numel() for p in mdn.parameters()):,}")
'''


class MDRNN(nn.Module):
    def __init__(self,z_dim,action_dim,hidden_dim,num_mixtures):
        super(MDRNN,self).__init__()

        self.hidden_dim=hidden_dim

        self.lstm=LSTM(z_dim,action_dim,hidden_dim)
        self.mdn=MDN(hidden_dim,z_dim,num_mixtures)

    def init_hidden(self,batch_size):
        h=torch.zeros(batch_size,self.hidden_dim)
        c=torch.zeros(batch_size,self.hidden_dim)

        return h,c


    def forward(self,z_seq,a_seq):
        batch_size,seq_len,_=z_seq.shape
        h,c=self.init_hidden(batch_size)

        pis=[]
        mus=[]
        sigmas=[]

        for t in range(seq_len):
            z_t=z_seq[:,t]
            a_t=a_seq[:,t]

            x_t=torch.cat([z_t,a_t],dim=-1)
            h,c=self.lstm(x_t,h,c)
            pi,mu,sigma=self.mdn(h)
            pis.append(pi)
            mus.append(mu)
            sigmas.append(sigma)

        pis=torch.stack(pis,dim=1)
        mus=torch.stack(mus,dim=1)
        sigmas=torch.stack(sigmas,dim=1)
        return pis,mus,sigmas

'''
# ---- Test MDRNN ----
z_dim = 32
action_dim = 4
hidden_dim = 256
num_mixtures = 5
batch_size = 16
seq_len = 10

mdrnn = MDRNN(z_dim, action_dim, hidden_dim, num_mixtures)
z_seq = torch.randn(batch_size, seq_len, z_dim)
a_seq = torch.randn(batch_size, seq_len, action_dim)

pis, mus, sigmas = mdrnn(z_seq, a_seq)

print("=== Shapes ===")
print(f"z_seq:   {z_seq.shape}")     # [16, 10, 32]
print(f"a_seq:   {a_seq.shape}")     # [16, 10, 4]
print(f"pis:     {pis.shape}")       # [16, 10, 5]
print(f"mus:     {mus.shape}")       # [16, 10, 5, 32]
print(f"sigmas:  {sigmas.shape}")    # [16, 10, 5, 32]

print("\n=== Checks ===")
print(f"pi sums to 1: {pis[0, 0].sum().item():.4f}")
print(f"sigma all positive: {(sigmas > 0).all().item()}")
print(f"Total params: {sum(p.numel() for p in mdrnn.parameters()):,}")

'''


def mdrnn_loss(pis,mus,sigmas,z_target):                    #z_target=batch,seq_len,32
    z_target=z_target.unsqueeze(2).expand_as(mus)           #z_target=batch,seq_len,num_mixture,32
    log_prob=-0.5*math.log(2*math.pi)-torch.log(sigmas)-0.5*((z_target-mus)/sigmas)**2  #log probability per component per distributions
    log_prob=torch.sum(log_prob,dim=-1)
    log_pi=torch.log(pis+1e-8)
    log_mix=log_pi+log_prob
    log_p=torch.logsumexp(log_mix,dim=-1)
    loss=-log_p.mean()
    return loss

# ---- Test MDRNN Loss ----
z_dim = 32
action_dim = 4
hidden_dim = 256
num_mixtures = 5
batch_size = 16
seq_len = 10

mdrnn = MDRNN(z_dim, action_dim, hidden_dim, num_mixtures)
z_seq = torch.randn(batch_size, seq_len, z_dim)
a_seq = torch.randn(batch_size, seq_len, action_dim)
z_target = torch.randn(batch_size, seq_len, z_dim)

pis, mus, sigmas = mdrnn(z_seq, a_seq)
loss = mdrnn_loss(pis, mus, sigmas, z_target)

print(f"Loss: {loss.item():.4f}")
print(f"Loss is finite: {torch.isfinite(loss).item()}")


