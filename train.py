import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from test import Diffusion3DGenerator  # Import your model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion3DGenerator")
    parser.add_argument("--image_size", type=int, default=32, help="Size of input images")
    parser.add_argument("--voxel_size", type=int, default=32, help="Size of output voxel grid")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension of the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save model every n epochs")
    return parser.parse_args()

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger.add(f"{output_dir}/train.log", rotation="10 MB")


def load_and_preprocess_data(args):
    dataset = load_dataset("cifar10", split="train")
    
    preprocess = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    all_images = []
    for example in tqdm(dataset, desc="Preprocessing"):
        image = preprocess(example["img"].convert("RGB"))
        all_images.append(image)
    
    return torch.stack(all_images)




def train(args, model, train_dataloader, optimizer, device):
    save_every = 1
    
    model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        
        for batch in progress_bar:
            images = batch[0].to(device)  # Extract images from the batch
            batch_size = images.size(0)
            time_steps = torch.randint(0, args.num_timesteps, (batch_size,), device=device)
            
            optimizer.zero_grad()
            
            # Ensure images have the correct shape (B, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)  # Add batch dimension if missing
            
            outputs = model(images, time_steps)
            
            # Assuming outputs is [B, 1, D, H, W], we'll project it to 2D
            # by taking the mean along the depth dimension
            outputs_2d = outputs.mean(dim=2)  # Result: [B, 1, H, W]
            
            # Ensure outputs_2d and images have the same number of channels
            if outputs_2d.size(1) != images.size(1):
                outputs_2d = outputs_2d.repeat(1, images.size(1), 1, 1)
            
            # Ensure outputs_2d and images have the same spatial dimensions
            outputs_2d = F.interpolate(outputs_2d, size=images.shape[2:], mode='bilinear', align_corners=False)
            
            loss = F.mse_loss(outputs_2d, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint and model
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            model_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pth")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save model
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
            
def main():
    args = parse_args()
    setup_logging(args.output_dir)
    logger.info(f"Starting training with args: {args}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = Diffusion3DGenerator(
        image_size=args.image_size,
        voxel_size=args.voxel_size,
        num_timesteps=args.num_timesteps,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    dataset = load_and_preprocess_data(args)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    train(args, model, train_dataloader, optimizer, device)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()