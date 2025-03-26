import argparse
import os
import torch
from dataset import *

def SlotAttention(gs: torch.Tensor, slots: torch.Tensor, iters=3):
    """
    gs: (G, D) tensor of Gaussians
    slots: (N_S, D) tensor of k-means initialized slots
    """
    k,v = gs, gs
    
    for _ in range(iters):
        q = slots
        # Compute attention weights
        attn = torch.softmax(q @ k.transpose(-2,-1), dim=-1)
        
        # Weighted means of values
        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = attn @ v
        
        # Update query
        slots += updates

    return slots

def kmeans(gs, k, iters=10):
    """
    gs: input tensor of shape 
    k: number of clusters (slots)
    iters: number of iterations
    """  
    n_g, d = gs.shape

    # Initialize centroids randomly
    centroids = gs[torch.randperm(n_g)[:k]]

    for _ in range(iters):
        # Compute the distances from each point to each centroid
        distances = torch.cdist(gs, centroids)        
        # Assign each point to the nearest centroid (min distance)
        assignments = torch.argmin(distances, dim=1)        
        # Update centroids by computing the mean of the assigned points
        for i in range(k):
            centroids[i] = gs[assignments == i].mean(dim=0)

    return torch.tensor(centroids, device=gs[0].device)

def export_slots(gs, slots):
    """
    gs: (G, D) tensor of Gaussians
    slots: (N_S, D) tensor of attended slots
    """
    # Compute attention -> (N_S, G)
    attn = torch.softmax(slots @ gs.transpose(-2,-1), dim=-1)
    means = gs[:, :3] # (G, 3)
    rots = gs[:, 3:7] # (G, 4)
    scales = gs[:, 7:10] # (G, 3)

    for slot in attn:
        opacity = slot.unsqueeze(-1) # (G, 1)
        

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument
    parser.add_argument('--data_dir', default='./data', type=str, help='where to find the dataset' )
    parser.add_argument('--output_dir', default='./tmp', type=str, help='where to save models' )
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--frame', default=0, type=int, help='which frame to inference.')

    opt = parser.parse_args()

    print(opt.data_dir)
    print(opt.output_dir)

    # Load the dataset
    train_set = ShapeOfMotion(os.path.join(opt.data_dir), opt.num_slots)

    # Load the frame
    frame = train_set[opt.frame]

    slots = kmeans(frame['all_gs'], opt.num_slots)
    slots = SlotAttention(frame['all_gs'], slots, opt.num_iterations)


if __name__ == '__main__':
    main()