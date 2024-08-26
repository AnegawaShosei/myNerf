import torch
import numpy as np
from torch import nn
import time

class NeRF(nn.Module):
    def __init__(self, L_pos = 10, L_dir = 4, gpu = torch.device('cuda')):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gpu = gpu
        
        self.pos_input = nn.Linear(3 + self.L_pos * 2 * 3, 256).to(gpu)
        self.posnet = nn.ModuleList([nn.Linear(256, 256) for i in range(7)]).to(gpu)
        self.alpha_output = nn.Linear(256, 1).to(gpu)

        self.view_input = nn.Linear(256 + 3 + self.L_dir * 2 * 3, 128).to(gpu)
        self.rgb_output = nn.Linear(128, 3).to(gpu)

    def forward(self, pos, view):
        enct = time.time()
        pos = self.encode(pos, self.L_pos).to(self.gpu)
        view = self.encode(view, self.L_dir).to(self.gpu)
        #print(f'Time taken for encoding: {time.time() - enct}')
        #print(f'pos mean: {pos.mean()}')
        modeltime = time.time()

        out = self.relu(self.pos_input(pos))
        #print(f'L1 mean: {out.mean()}')
        #print(f'First layer time: {time.time() - modeltime}')
        for lin in self.posnet:
            out = self.relu(lin(out))
            #print(f'Ln mean: {out.mean()}')

        #print(f'Loop layer time: {time.time() - modeltime}')
        
        alpha = self.relu(self.alpha_output(out))
        #print(f'Alpha mean: {alpha.mean()}')
        out = torch.cat((out, view), axis = -1)
        out = self.relu(self.view_input(out))
        #print(f'Lout mean: {out.mean()}')
        out = self.sigmoid(self.rgb_output(out))

        #print(f'Time taken for running model: {time.time() - modeltime}')
        
        return out, alpha
        
    def encode(self, vec, L):
        enc = [vec] 
        
        for i in range(0, L):
            for f in [torch.sin, torch.cos]:
                enc.append(f(vec * np.pi * (2 ** i))) #Encoding scheme from paper
        
        return torch.cat(enc, -1)

def run_model(start, stop, N_samples, model, rays_o, rays_d, use_chunks = False, chunk_size = 4096):
    st = time.time()
    samples = (torch.linspace(start, stop, N_samples) + torch.rand(list(rays_o.shape[:-1]) + [N_samples]) * (stop-start)/N_samples)
    xyz = (samples[..., :, None] * rays_d[..., None, :] + rays_o[..., None, :]).view(-1, 3)
    views = rays_d[..., None, :].tile(1, N_samples, 1).view(-1, 3)
    
    #print(f'Time taken to sample: {time.time() - st}')
    #print("Finished sampling rays")
    #print("Running model")

    rgb = None
    alpha = None
    if use_chunks:
        rgbs = []
        alphas = []
        for i in range(0, rays_o.shape[0]*N_samples, chunk_size):
            if i + chunk_size < xyz.shape[0]:
                rgb, alpha = model.forward(xyz[i:i+chunk_size, :], views[i:i+chunk_size, :])
            else:
                rgb, alpha = model.forward(xyz[i:, :], views[i:, :])
    
            rgbs.append(rgb)
            alphas.append(alpha)

        rgb = torch.cat(rgbs, 0)
        alpha = torch.cat(alphas, 0)

    else:
        mtt = time.time()
        #print(f'XYZ mean: {xyz.mean()}')
        #print(f'views mean: {views.mean()}')
        rgb, alpha = model.forward(xyz, views)
        #print(f'Time taken to run model: {time.time() - mtt}')
        rgb = rgb.to(torch.device('cpu'))
        #print(f'RGB mean: {rgb.mean()}')
        alpha = alpha.to(torch.device('cpu'))
        #print(f'Alpha mean: {alpha.mean()}')
    #print(rgb.shape, alpha.shape)

    #print("Done Running model")
    
    #Get distances between samples, and append 1e10 to the end to simulate infinitely extending ray
    ppt = time.time()
    dists = torch.cat([samples[..., 1:] - samples[..., :-1], torch.tensor([1e10]).expand(samples[...,:1].shape)], -1).view(-1, 1)
    alpha_i = 1 - torch.exp(-dists * alpha)
    weights = alpha_i * torch.cumprod(1-alpha + 1e-10, -1) / ((1-alpha + 1e-10)[0])
    #print(f'Weights mean: {weights.mean()}')
    
    rgb_vals = torch.sum((rgb * weights).view(rays_o.shape[0], N_samples, -1), -2)
    depth_map = torch.sum((weights * samples.view(-1, 1)).view(rays_o.shape[0], N_samples, -1), -2)
    acc_map = torch.sum(weights.view(rays_o.shape[0], N_samples, -1), -2)
    #print(f'Time taken to process outputs: {time.time() - ppt}')

    #print(f'RGB_vals mean {rgb_vals.mean()}')

    return rgb_vals, depth_map, acc_map


def batched_get_rays(H, W, F, c2w):
    i, j = torch.meshgrid(torch.arange(0, W, dtype = torch.int32), torch.arange(0, H, dtype=torch.int32), indexing = 'xy') 
    dirs = torch.stack([(i - W * 0.5) / F, -(j - H * .5)/F, -torch.ones_like(i)], -1) #(H, W, 3)
    dirs = dirs.expand(c2w.shape[0], H, W, 3)   #(B, H, W, 3)
    rays_d = torch.einsum('bij,bhwj->bhwi', c2w[:, :3, :3], dirs)  #(B, H, W, 3)
    rays_o = torch.broadcast_to(torch.Tensor(c2w[:, :3, -1]), rays_d.shape) #(B, H, W, 3)
    
    rays_d /= torch.linalg.vector_norm(rays_d, dim = 2).view(rays_d.shape[0], H, W, 1)

    return rays_o, rays_d
    
def get_rays(H, W, F, c2w):
    i, j = torch.meshgrid(torch.arange(0, W, dtype = torch.int32), torch.arange(0, H, dtype=torch.int32), indexing = 'xy') #Index all pixels
    dirs = torch.stack([(i - W * 0.5) / F, -(j - H * .5)/F, -torch.ones_like(i)], -1) #Center on (0, 0, 0)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) #Rotate using c2w mat
    rays_o = torch.broadcast_to(torch.Tensor(c2w[:3, -1]), rays_d.shape) #All ray origins are at camera location

    rays_d /= torch.linalg.vector_norm(rays_d, dim = 2).view(H, W, 1) #Normalize viewing directions

    return rays_o, rays_d

class PositionalEncoding():
    def __init__(self, L_pos = 10, L_dir = 4):
        '''
        L_pos: Position (xyz) coordinates will be mapped to L_pos * 2 dimentions
        L_dir: Viewing direction unit vector will be mapped to L_dir * 2 dimentions
        '''
        self.L_pos = 10
        self.L_dir = 4

    def encode(self, pos, vdir):
        posenc = [pos] #Try including base pos and dir
        direnc = [vdir]
        
        for i in range(0, self.L_pos):
            for f in [torch.sin, torch.cos]:
                posenc.append(f(pos * np.pi * (2 ** i)))
                              
        for j in range(0, self.L_dir):
            for f in [torch.sin, torch.cos]:
                direnc.append(f(vdir * np.pi * (2 ** i)))
        
        posenc = torch.cat(posenc, -1)
        direnc = torch.cat(direnc, -1)
        
        return posenc, direnc