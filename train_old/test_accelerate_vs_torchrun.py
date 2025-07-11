#!/usr/bin/env python3
"""
Comparison of Accelerate vs Manual Distributed Training
Shows the key differences in data sharding and setup
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from accelerate import Accelerator
from datasets import Dataset as HFDataset

# Set environment variables
os.environ['HF_HOME'] = '/mnt/nvme4/dipika/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nvme4/dipika/hf_cache'

class DummyDataset(Dataset):
    """Simple dummy dataset for testing"""
    def __init__(self, size=1000):
        self.size = size
        self.data = [f"Sample {i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"text": self.data[idx], "idx": idx}

def test_manual_distributed():
    """Test manual distributed training setup"""
    print("\n" + "="*50)
    print("MANUAL DISTRIBUTED TRAINING")
    print("="*50)
    
    # Check if we're in distributed mode
    if not dist.is_initialized():
        print("❌ Not in distributed mode!")
        print("   Run with: torchrun --nproc_per_node=8 test_accelerate_vs_torchrun.py")
        return
    
    # Get distributed info
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"Process {rank}/{world_size} on GPU {local_rank}")
    
    # Create dataset
    dataset = DummyDataset(1000)
    print(f"[Rank {rank}] Original dataset size: {len(dataset)}")
    
    # Manual distributed sampler
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        shuffle=False
    )
    
    print(f"[Rank {rank}] Dataloader batches: {len(dataloader)}")
    print(f"[Rank {rank}] Expected samples per rank: {len(dataset) // world_size}")
    print(f"[Rank {rank}] Actual samples per rank: {len(dataloader) * 4}")
    
    # Process a few batches to show data sharding
    print(f"[Rank {rank}] First 3 batch indices:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        indices = batch['idx'].tolist()
        print(f"[Rank {rank}] Batch {i}: indices {indices}")

def test_accelerate_distributed():
    """Test Accelerate distributed training setup"""
    print("\n" + "="*50)
    print("ACCELERATE DISTRIBUTED TRAINING")
    print("="*50)
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    print(f"Process {accelerator.process_index}/{accelerator.num_processes}")
    print(f"Device: {accelerator.device}")
    print(f"Is main process: {accelerator.is_main_process}")
    
    # Create dataset (HuggingFace Dataset for better Accelerate compatibility)
    data = [{"text": f"Sample {i}", "idx": i} for i in range(1000)]
    dataset = HFDataset.from_list(data)
    print(f"[Process {accelerator.process_index}] Original dataset size: {len(dataset)}")
    
    # Create dataloader (no manual sampler needed!)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )
    
    print(f"[Process {accelerator.process_index}] Before prepare - batches: {len(dataloader)}")
    
    # Prepare with Accelerate (automatic data sharding!)
    dataloader = accelerator.prepare(dataloader)
    
    print(f"[Process {accelerator.process_index}] After prepare - batches: {len(dataloader)}")
    print(f"[Process {accelerator.process_index}] Samples per process: {len(dataloader) * 4}")
    
    # Process a few batches to show data sharding
    print(f"[Process {accelerator.process_index}] First 3 batch indices:")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        indices = batch['idx'].tolist()
        print(f"[Process {accelerator.process_index}] Batch {i}: indices {indices}")

def test_accelerate_gathering():
    """Test Accelerate's gathering functionality"""
    print("\n" + "="*50)
    print("ACCELERATE GATHERING")
    print("="*50)
    
    accelerator = Accelerator()
    
    # Create some dummy loss values
    local_loss = torch.tensor(float(accelerator.process_index + 1)).to(accelerator.device)
    print(f"[Process {accelerator.process_index}] Local loss: {local_loss.item()}")
    
    # Gather losses from all processes
    gathered_losses = accelerator.gather_for_metrics(local_loss)
    
    if accelerator.is_main_process:
        print(f"[Main Process] Gathered losses: {gathered_losses.tolist()}")
        print(f"[Main Process] Average loss: {gathered_losses.mean().item()}")
    
    # Test gathering with different sizes
    local_tensor = torch.randn(2, 3).to(accelerator.device) * (accelerator.process_index + 1)
    gathered_tensor = accelerator.gather_for_metrics(local_tensor)
    
    if accelerator.is_main_process:
        print(f"[Main Process] Gathered tensor shape: {gathered_tensor.shape}")
        print(f"[Main Process] Expected shape: {(accelerator.num_processes * 2, 3)}")

def main():
    """Main function"""
    print("ACCELERATE vs MANUAL DISTRIBUTED TRAINING COMPARISON")
    print("="*60)
    
    # Test manual distributed (only works with torchrun)
    test_manual_distributed()
    
    # Test Accelerate (works with both torchrun and accelerate launch)
    test_accelerate_distributed()
    
    # Test Accelerate gathering
    test_accelerate_gathering()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Manual Distributed Training:")
    print("  ✓ Requires torchrun")
    print("  ✓ Manual DistributedSampler setup")
    print("  ✓ Manual device placement")
    print("  ✓ Manual gradient synchronization")
    print("  ✓ Manual result gathering")
    
    print("\nAccelerate:")
    print("  ✓ Works with both torchrun and accelerate launch")
    print("  ✓ Automatic data sharding via prepare()")
    print("  ✓ Automatic device placement")
    print("  ✓ Automatic gradient synchronization")
    print("  ✓ Easy result gathering with gather_for_metrics()")
    print("  ✓ Handles mixed precision automatically")
    print("  ✓ Unified API for different backends")

if __name__ == "__main__":
    main() 