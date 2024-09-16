import torch

def save_checkpoint(generator, discriminator, opt_gen, opt_disc, scaler, epoch, file_name='checkpoint.pth.tar'):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
        'scaler': scaler.state_dict()  # Optional if using mixed precision
    }, file_name)


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, model_type='generator'):
    checkpoint = torch.load(checkpoint_path)
    if model_type == 'generator':
        model.load_state_dict(checkpoint['generator_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['opt_gen_state_dict'])
    elif model_type == 'discriminator':
        model.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        if optimizer:
            optimizer.load_state_dict(checkpoint['opt_disc_state_dict'])
    
    if scaler and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    print("=> Checkpoint loaded successfully")



# def save_checkpoint(state, filename="checkpoint.pth"):
#     """
#     Save the model checkpoint.
    
#     Args:
#     - state (dict): A dictionary containing the model's state_dict, optimizer's state_dict, and other metadata.
#     - filename (str): The filename for saving the checkpoint.
#     """
#     print(f"=> Saving checkpoint to {filename}")
#     torch.save(state, filename)
#     print("=> Checkpoint saved successfully")

# def load_checkpoint(checkpoint_path, model, optimizer=None):
#     """
#     Load a model checkpoint.
    
#     Args:
#     - checkpoint_path (str): The path to the saved checkpoint file.
#     - model (torch.nn.Module): The model instance where the weights will be loaded.
#     - optimizer (torch.optim.Optimizer, optional): The optimizer instance where the state will be loaded.
    
#     Returns:
#     - checkpoint (dict): The entire checkpoint that was loaded.
#     """
#     print(f"=> Loading checkpoint from {checkpoint_path}")
    
#     # Load the entire checkpoint, handle map_location to ensure compatibility with GPU/CPU
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
#     # Load the model state_dict
#     model.load_state_dict(checkpoint['generator_state_dict'], strict=False)
    
#     # Load the optimizer state_dict if provided
#     if optimizer and 'optimizer_state_dict' in checkpoint:
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         print("=> Optimizer state loaded")
    
#     print("=> Checkpoint loaded successfully")
#     return checkpoint


