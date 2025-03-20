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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Trainer(rank, world_size, opt):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Set device for each process
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(rank, device)
    
    # Initialize process group for DDP
    init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Load dataset
    resolution = (128, 128)    
    train_set = PARTNET(opt.data_dir,'train')

    model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim)
    model = model.to(device)
    # model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])

    # print(rank, model.encoder_cnn.encoder_pos.grid.device)
    # print(rank, model.encoder_cnn.conv1.weight.device)
    # print(rank, model.decoder_cnn.conv6.weight.device)
    # print(rank, model.slot_attention.slots_mu.device)

    # Wrap model in DDP
    model = DDP(model, device_ids=[rank])

    # Loss function
    criterion = nn.MSELoss()

    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=opt.learning_rate)

    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(opt.output_dir, 'last.ckpt')

    if os.path.exists(checkpoint_path):
        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(checkpoint['model_state_dict'].keys())
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)  # Default to 0 if not saved

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers, sampler=train_sampler)


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
            
            image = sample['image'].to(device)
            recon_combined, recons, masks, slots = model(image)
            loss = criterion(recon_combined, image)
            total_loss += loss.item()

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)

        if rank == 0:   # Print and save only from rank 0
            print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
                datetime.timedelta(seconds=time.time() - start)))

            if not epoch % 10:
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
            
    # Clean up the process group after training
    destroy_process_group()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument
    parser.add_argument('--data_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
    parser.add_argument('--output_dir', default='./tmp/model10.ckpt', type=str, help='where to save models' )
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--learning_rate', default=0.0004, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=1000, type=int, help='number of training epochs.')

    opt = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(opt.seed)

    # Spawn processes for DDP
    world_size = torch.cuda.device_count()
    mp.spawn(Trainer, args=(world_size, opt), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()