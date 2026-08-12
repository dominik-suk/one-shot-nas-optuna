import os
import glob
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from optuna.samplers import RandomSampler

from external.sample_blocks import Sampler
from src.models.supernet import Supernet
from src.utils.supernet_io_utils import load_supernet, save_supernet


def train_supernet(
        supernet: Supernet,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        search_space: dict,
        validation_search_space: dict,
        epochs: int = 100,
        device: str = "cuda",
        save_dir: str = "models/supernet",
):
    supernet.to(device)
    optimizer = optim.Adam(supernet.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_validation_loss = float('inf')
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'supernet_epoch_*.pth'))

    if existing_checkpoints:
        latest_checkpoint = max(existing_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_validation_loss = torch.load(latest_checkpoint, map_location=device, weights_only=False)['validation_loss']
        start_epoch = load_supernet(supernet, latest_checkpoint, optimizer, device) + 1
        for _ in range(start_epoch):
            scheduler.step()

    for epoch in range(start_epoch, epochs):
        supernet.train()
        total_train_loss = 0.0
        train_study = optuna.create_study(sampler=RandomSampler())
        train_correct = 0
        train_total = 0

        for batch_index, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            trial = train_study.ask()
            sampler = Sampler(trial)
            architecture_config = sampler.construct_sample(search_space)
            output = supernet(batch_x, architecture_config)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=1)
            optimizer.step()
            total_train_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            train_correct += torch.sum(predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        # avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        # current_learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        supernet.eval()
        total_validation_loss = 0.0
        validation_correct = 0
        validation_total = 0
        validation_study = optuna.create_study(sampler=RandomSampler())
        validation_trial = validation_study.ask()
        validation_sampler = Sampler(validation_trial)
        # fixed_architecture_config = validation_sampler.construct_sample(validation_search_space)

        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                architecture = validation_sampler.construct_sample(search_space)
                output = supernet(batch_x, architecture)
                # output = supernet(batch_x, fixed_architecture_config)
                total_validation_loss += criterion(output, batch_y).item()
                predictions = torch.argmax(output, dim=1)
                validation_correct += (predictions == batch_y).sum().item()
                validation_total += batch_y.size(0)

        avg_validation_loss = total_validation_loss / len(validation_loader)
        validation_accuracy = validation_correct / validation_total
        print(f'Epoch: {epoch+1}/{epochs}, Train Accuracy: {train_accuracy*100:.2f}%, Validation Accuracy: {validation_accuracy*100:.2f}%')
        is_best = avg_validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = avg_validation_loss
        if is_best or epoch % 10 == 0:
            save_supernet(
                supernet,
                epoch,
                optimizer,
                validation_loss=avg_validation_loss,
                is_best=is_best,
                save_dir=save_dir
            )
    return supernet
