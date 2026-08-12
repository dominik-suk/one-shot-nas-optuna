import os
import shutil
import torch

def save_supernet(model, epoch, optimizer, validation_loss=None, is_best=False, save_dir="../../models/supernet"):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if validation_loss is not None:
        state['validation_loss'] = validation_loss

    filepath = os.path.join(save_dir, f'supernet_epoch_{epoch}.pth')
    torch.save(state, filepath)

    if is_best:
        dirname = os.path.dirname(save_dir)
        best_filepath = os.path.join(dirname, f'supernet_best_{validation_loss}.pth')
        shutil.copyfile(filepath, best_filepath)


def load_supernet(model, filepath, optimizer=None, device="cuda"):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint["epoch"]
