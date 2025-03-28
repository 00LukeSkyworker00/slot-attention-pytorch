import argparse
import os
import torch
from dataset import *
from plyfile import PlyData, PlyElement
import seaborn as sns
import itertools

def SlotAttention(gs: torch.Tensor, slots: torch.Tensor, iters=3):
    """
    gs: (G, D) tensor of Gaussians
    slots: (N_S, D) tensor of k-means initialized slots
    """
    k,v = gs, gs
    
    for i in range(iters):

        # Compute attention weights
        q = slots
        q *= slots.size(1) ** -0.5  # Normalization.
        attn = k @ q.T # (G,N_S)
        attn = torch.softmax(attn, dim=-1)

        # Weighted means of values
        attn += 1e-8
        attn /= attn.sum(dim=-2, keepdim=True)
        updates = attn.T @ v # (N_S,D)

        # Update query
        slots = slots + updates

    return slots

def compute_attention(gs, slots):
    """
    gs: (G, D) tensor of Gaussians
    slots: (N_S, D) tensor of attended slots
    """
    # Compute attention weights
    q = slots
    q *= slots.size(1) ** -0.5  # Normalization.
    attn = gs @ q.T # (G, N_S)
    attn = torch.softmax(attn, dim=-1)

    # Weighted means of values
    # attn += 1e-8
    # attn /= attn.sum(dim=-2, keepdim=True)

    return attn # (G, N_S)

def kmeans(gs, k, iters=50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    gs: input tensor of shape 
    k: number of clusters (slots)
    iters: number of iterations
    """  
    n_g, d = gs.shape
    print("start kmeans!")

    # Initialize centroids randomly
    centroids = gs[torch.randperm(n_g)[:k]].to(gs[0].device)
    last = centroids.clone()

    for _ in range(iters):
        # Compute the distances from each point to each centroid
        distances = torch.cdist(gs, centroids)        
        # Assign each point to the nearest centroid (min distance)
        assignments = torch.argmin(distances, dim=1)
        # Update centroids by computing the mean of the assigned points
        for i in range(k):
            centroids[i] = gs[assignments == i].mean(dim=0)

        improvement = (last - centroids).sum()
        print('kmeans improvements: ',improvement)
        if improvement == 0.:
            print("No significant improvement in kmeans!")
            break
        last = centroids.clone()

    print("end kmeans!")

    return centroids, assignments

def export_slots_ply(gs, slots):
    """
    gs: (G, D) tensor of Gaussians
    slots: (N_S, D) tensor of attended slots
    """
    attn = compute_attention(gs, slots)# (N_S, G)
    means = gs[:, :3] # (G, 3)
    rots = gs[:, 3:7] # (G, 4)
    scales = gs[:, 7:10] # (G, 3)
    colors = np.array(sns.color_palette('husl', slots.shape[0])) # (N_S, 3)

    for i in range(attn):
        opacities = attn[i].unsqueeze(-1) # (G, 1)
        vis_color = colors[i].unsqueeze(0) # (1, 3)
        slot_gs = {
            'means': means,
            'rots': rots,
            'scales': scales,
            'opacities': opacities,
            'colors': vis_color
        }
        export_ply(slot_gs, f'./tmp/slot_{i}.ply')
    
        
def export_ply(gs, path):
        # mkdir_p(os.path.dirname(path))

        gs = gs.detach().cpu().numpy()

        xyz = gs['means']
        normals = np.zeros_like(xyz)
        opacities = gs['opacities']
        scale = gs['scales']
        rotation = gs['quats']

        f_dc = gs['colors']
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gs)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        print(xyz.shape, normals.shape, f_dc.shape, opacities.shape, scale.shape, rotation.shape)
        print(xyz[0], normals[0], f_dc[0], opacities[0], scale[0], rotation[0])
        attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


def construct_list_of_attributes(gs):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        # l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(gs['scales'].shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gs['quats'].shape[1]):
        l.append('rot_{}'.format(i))
    return l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument
    parser.add_argument('--data_dir', default='./data', type=str, help='where to find the dataset' )
    parser.add_argument('--output_dir', default='./tmp', type=str, help='where to save models' )
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--frame', default=10, type=int, help='which frame to inference.')
    parser.add_argument('--seed', default=0, type=int, help='Set seed for reproducibility.')

    opt = parser.parse_args()

    print(opt.data_dir)
    print(opt.output_dir)

    torch.manual_seed(opt.seed)

    # Load the dataset
    train_set = ShapeOfMotion(os.path.join(opt.data_dir), opt.num_slots)
  
    frame = train_set[opt.frame]
    inputs = frame['fg_gs']
    # inputs = frame['fg_gs'][:,:3] # only xyz
    # inputs = frame['fg_gs'][:,:7] # only xyz, rots
    # inputs = frame['fg_gs'][:,:10] # only xyz, rots, scale
    # inputs = frame['fg_gs'][:,11:14] # only color
    # inputs = frame['fg_gs'][:,10:14] # only color and opac
    # inputs = torch.cat((frame['fg_gs'][:,:3],frame['fg_gs'][:,11:14]), dim=1)

    # kmeans = KMeans(n_clusters=opt.num_slots, random_state=opt.seed, n_init='auto')
    # labels = kmeans.fit_predict(inputs.detach().cpu().numpy())
    # slots = torch.tensor(kmeans.cluster_centers_,device=inputs.device)

    slots, labels = kmeans(inputs, opt.num_slots, iters=200)

    slots = SlotAttention(inputs, slots, opt.num_iterations)
    # print(slots[0])
    attn = compute_attention(inputs, slots) # (G, N_S)
    
    # colors = torch.tensor(sns.color_palette('husl', opt.num_slots), device=inputs.device) # (N_S, 3)
    # colors = ((colors * 10.0) - 10.0)
    # colors = torch.ones((opt.num_slots, 3),device=attn.device) * -5

    values = [5., -5.]
    colors = torch.tensor(list(itertools.product(values, repeat=3)),device=inputs.device)

    frame_col = attn @ colors # (G, 3)
    # print(attn.shape)
    # print(colors.shape)
    # colors = colors.unsqueeze(1)
    # colors = attn.unsqueeze(-1).permute(1,0,2) * colors # (N_S, G, 3)
    # exit()
    # frame_col = colors[labels] # (G, 3)
    # print(frame_col[0])

    # colors = attn.permute(1,0).unsqueeze(-1) * 10 - 5
    # colors = colors.expand(-1,-1,3) * -1
    # for i in range(colors.size(0)):
    #     new_ckpt = train_set.modify_ckpt(colors[i])
    #     torch.save(new_ckpt, f"{opt.output_dir}/checkpoints/last_{i}.ckpt")

    # new_ckpt = train_set.modify_ckpt(frame_col)
    # torch.save(new_ckpt, f"{opt.output_dir}/checkpoints/last.ckpt")
    
    print('Save Checkpoint!!')


if __name__ == '__main__':
    main()