import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from PIL import Image


def visualize_reconstruction(model, data_loader, save_dir, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            break  # Get the first batch for visualization
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axes[0, i].imshow(x[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off') 
        axes[1, i].imshow(x_recon[i].cpu().numpy().squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction.png'))
    plt.close()
    
    
def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()
    
#function written by chatGPT
def save_cam(image, gcam):
    # shape gradcam:[64, 64]
    # Normalize the Grad-CAM values
    gcam = (gcam - np.min(gcam)) / (np.max(gcam) - np.min(gcam))

    # Resize Grad-CAM to match the image dimensions
    # h, w =[64, 64]
    h, w, _ = image.shape
    gcam_resized = np.array(Image.fromarray(gcam).resize((w, h), Image.BILINEAR))

    # Apply a colormap similar to cv2.applyColorMap
    gcam_colored = plt.cm.jet(gcam_resized)[:, :, :3] * 255
    gcam_colored = gcam_colored.astype(np.uint8)

    # Add Grad-CAM on top of the original image
    heatmap = gcam_colored.astype(np.float64)
    
    overlaid_image = (heatmap + image.astype(np.float64)) / 2
    overlaid_image = (overlaid_image / np.max(overlaid_image) * 255).astype(np.uint8)
    
    return(overlaid_image)

    # Save the overlaid image
    #Image.fromarray(overlaid_image).save(filename)
    
# visualize heatmaps
def visualize_heatmap(x, gcam_map, heatmap_dir, args):
    test_index = 0
    if x.size(1) == 1:  # Check if the input is grayscale
        x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels if necessary
    
    for i in range(x.size(0)):
        if i % 10 == 0:    
            raw_image = (x[i] * 255.0).clamp(0, 255)  # Ensure values are within range
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))
            
            im_path = os.path.join(heatmap_dir, f'{args.exp}_{args.latentvar}_{str(args.latentcls)}')
            os.makedirs(im_path, exist_ok=True)
            
            im.save(os.path.join(im_path, "{}-origin.png".format(test_index)))
            attmap_path = os.path.join(im_path, "{}-attmap.png".format(test_index))
            
            save_cam(ndarr, attmap_path, gcam_map[i])  # Use ndarr directly
            
        print(f'\nHeatmaps were saved for index {test_index}')
        test_index += 10  # Increment by 1 to process all images

# visualzie grid of heatmaps
def visualize_heatmap_grid(x, gcam_map, heatmap_dir, batch_idx, num_images=10):
    if x.size(1) == 1:  # Check if the input is grayscale
        x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels if necessary
    
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        raw_image = (x[i] * 255.0).clamp(0, 255)  # Ensure values are within range
        ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
        
        # Original image
        axs[0, i].imshow(ndarr)
        axs[0, i].axis('off')
        
        # Grad-CAM heatmap overlaid on the image
        overlaid_image = save_cam(ndarr, gcam_map[i])
        axs[1, i].imshow(overlaid_image)
        axs[1, i].axis('off')
    
    # Save the grid of images and heatmaps
    #grid_path = os.path.join(heatmap_dir,batch_idx)
    #os.makedirs(grid_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir,f'{batch_idx}_grid.png'))
    plt.show()
    
# Visualise single heatmap
def overlay_heatmap_single(x, heatmap, overlay_dir, img_idx):
    # x:      [78208, 1, 64, 64]
    # heatmap:[78208, 64, 64]
    
    if x.shape[1] == 1:  # Check if the input is grayscale
        # Use np.repeat to expand the single channel to three channels
        x = np.repeat(x, 3, axis=1)  # Repeat along the channel dimension
    
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    
    raw_image = (x[img_idx] * 255.0).clip(0, 255)  # Ensure values are within range
    ndarr = raw_image.squeeze().transpose(1, 2, 0).astype(np.uint8)  # Convert to HWC format
        
    # Original image
    axs[0].imshow(ndarr)
    axs[0].axis('off')
    
    # Heatmap
    axs[1].imshow(heatmap[img_idx])
    axs[1].axis('off')
        
    # Grad-CAM heatmap overlaid on the image
    overlaid_image = save_cam(ndarr, heatmap[img_idx])
    axs[2].imshow(overlaid_image)
    axs[2].axis('off')
    
    file_path = os.path.join(overlay_dir, f'overlay_{img_idx}.png')
    fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    
    
    

