import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch

import torch.multiprocessing as mp
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.utils.rnn import pack_padded_sequence

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Trainer(rank, world_size, opt):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    resolution = (128, 128)

    train_list = []

    for i in range(10, 51):
        train_set = ShapeOfMotion(os.path.join(opt.data_dir,f'movi_a_00{i}_anoMask'))
        train_list.append(train_set)
        # print(train_set[0]['fg_gs'].shape)
    
    train_set = ConcatDataset(train_list)
    # train_set = ShapeOfMotion(opt.data_dir)

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, 21)
    # model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
    # model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])
    model = model.to(device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Loss function
    criterion = nn.MSELoss()

    # Define optimizer
    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=opt.learning_rate)

    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(opt.output_dir, 'last.ckpt')

    if os.path.exists(checkpoint_path):
        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)  # Default to 0 if not saved

    # Create DataLoader with DistributedSampler
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
        shuffle=True, seed=opt.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=opt.batch_size, num_workers=opt.num_workers,
        sampler=train_sampler, collate_fn=collate_fn_padd
        )

    start = time.time()
    i = start_epoch * len(train_dataloader)  # Resume step count

    for epoch in range(start_epoch, opt.num_epochs):  # Resume from the saved epoch
        model.train()

        total_loss = 0

        for sample in tqdm(train_dataloader):
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            # print(sample['gt_imgs'].shape)

            # Get inputs and lengths
            gt_imgs = sample['gt_imgs'].to(device)
            fg_gs = sample['fg_gs'].to(device)
            # lengths = sample['fg_lengths']  # Sequence lengths

            # Pack the sequence
            # packed_fg_gs = pack_padded_sequence(fg_gs, lengths, batch_first=True, enforce_sorted=False)

            # Forward pass through model
            recon_combined, recons, masks, slots = model(fg_gs, gt_imgs)
            
            # Loss calculation
            loss = criterion(recon_combined, gt_imgs)
            # print(loss.item())
            total_loss += loss.item()

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)

        print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
            datetime.timedelta(seconds=time.time() - start)))

        if rank == 0:  # Only rank 0 saves checkpoints
            print(f"Epoch: {epoch}, Loss: {total_loss}, Time: {datetime.timedelta(seconds=time.time() - start)}")

            if epoch % 100 == 0:
                os.makedirs(opt.output_dir, exist_ok=True)

                torch.save({
                    'epoch': epoch,  # Save the current epoch
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(opt.output_dir, f'{epoch}.ckpt'))

                torch.save({
                    'epoch': epoch,  # Save the last epoch for resuming
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

    # Clean up process group
    destroy_process_group()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument
    parser.add_argument('--data_dir', default='./data', type=str, help='where to find the dataset' )
    parser.add_argument('--output_dir', default='./tmp', type=str, help='where to save models' )
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_img_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=1000, type=int, help='number of training epochs.')

    opt = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(opt.seed)

    world_size = torch.cuda.device_count()
    mp.spawn(Trainer, args=(world_size, opt), nprocs=world_size, join=True)

        
if __name__ == '__main__':
    main()