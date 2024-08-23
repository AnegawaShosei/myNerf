import torch
import numpy as np

class NeRF(nn.Module):
    def __init__(self, L_pos = 10, L_dir = 4):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid
        
        self.pos_input = nn.Linear(3 + self.L_pos * 2 * 3, 256)
        self.posnet = nn.ModuleList([nn.Linear(256, 256) for i in range(7)])
        self.alpha_output = nn.Linear(256, 1)

        self.view_input = nn.Linear(256 + 3 + self.L_dir * 2 * 3, 128)
        self.rgb_output = nn.Linear(128, 3)

    def forward(self, pos, view):
        pos, view = self.encode(pos, view)

        out = self.relu(self.pos_input(pos))
        for lin in self.posnet:
            out = self.relu(lin(out))

        alpha = self.alpha_output(out) 
        out = torch.cat((out, view), axis = -1)

        out = self.relu(self.view_input(out))
        rgb = self.rgb_output(out)

        return torch.cat((rgb, alpha), axis = -1)
        

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

def batched_get_rays(H, W, F, c2w):
    i, j = torch.meshgrid(torch.arange(0, W, dtype = torch.int32), torch.arange(0, H, dtype=torch.int32), indexing = 'xy') 
    dirs = torch.stack([(i - W * 0.5) / F, -(j - H * .5)/F, -torch.ones_like(i)], -1) #(H, W, 3)
    dirs = dirs.expand(c2w.shape[0], H, W, 3)   #(B, H, W, 3)
    rays_d = torch.einsum('bij,bhwj->bhwi', c2w[:, :3, :3], dirs)  #(B, H, W, 3)
    rays_o = torch.broadcast_to(torch.Tensor(c2w[:, :3, -1]), rays_d.shape) #(B, H, W, 3)
    
    rays_d /= torch.linalg.vector_norm(rays_d, dim = 2).reshape(rays_d.shape[0], H, W, 1)

    return rays_o, rays_d
    
def get_rays(H, W, F, c2w):
    i, j = torch.meshgrid(torch.arange(0, W, dtype = torch.int32), torch.arange(0, H, dtype=torch.int32), indexing = 'xy')
    dirs = torch.stack([(i - W * 0.5) / F, -(j - H * .5)/F, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = torch.broadcast_to(torch.Tensor(c2w[:3, -1]), rays_d.shape)

    rays_d /= torch.linalg.vector_norm(rays_d, dim = 2).reshape(H, W, 1)

    return rays_o, rays_d