import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import time
import torch
import torch.optim as optim
from model.E3RelaxH2 import E3RelaxH2
from utils.lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from torch.utils.data import DataLoader
from utils.graph_utils import vector_norm
from utils.utils import *
from collections import defaultdict
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from model.ema import ModelEMA, save_ckpt, load_ckpt_for_resume
import warnings
warnings.filterwarnings("ignore")

# %%
def val(
    model,
    dataloader,
    device,
    use_cell_loss,
):
    model.eval()

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter() if use_cell_loss else None

    pred_quantity_dict = defaultdict(list) if use_cell_loss else None
    total_pos_abs = 0.0
    total_pos_count = 0

    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            if use_cell_loss:
                pos_pred_list, cell_pred_list = outputs
            else:
                pos_pred_list = outputs
                cell_pred_list = None
                
            pos_label = getattr(data, "pos_r", None)
            cell_label = getattr(data, "cell_r", None)
            if pos_label is None:
                pos_label = data.pos_relaxed
            if cell_label is None:
                cell_label = data.cell
            
            loss_pos = 0
            loss_cell = 0 if use_cell_loss else None
            for l, pos_pred in enumerate(pos_pred_list):
                loss_pos = loss_pos + compute_cart_mean_absolute_displacement_wrap_pred(pos_pred, pos_label, cell_label, data.batch)
                if use_cell_loss:
                    loss_cell = loss_cell + (cell_pred_list[l] - cell_label).abs().mean()
            loss = loss_pos + loss_cell if use_cell_loss else loss_pos

            if use_cell_loss:
                pred_quantity_dict['cell_label'].append(cell_label)
                pred_quantity_dict['cell_pred'].append(cell_pred_list[-1])
            batch_pos_mae = compute_cart_mean_absolute_displacement_wrap_pred(
                pos_pred_list[-1], pos_label, cell_label, data.batch
            )
            total_pos_abs += batch_pos_mae.item() * pos_label.numel()
            total_pos_count += pos_label.numel()

            running_loss.update(loss.item()) 
            running_loss_pos.update(loss_pos.item()) 
            if running_loss_cell is not None and loss_cell is not None:
                running_loss_cell.update(loss_cell.item())

    if use_cell_loss and pred_quantity_dict['cell_label']:
        cell_label = torch.cat(pred_quantity_dict['cell_label'], dim=0)
        cell_pred = torch.cat(pred_quantity_dict['cell_pred'], dim=0)
        valid_cell_mae = (cell_label - cell_pred).abs().mean().item()
    else:
        valid_cell_mae = None

    valid_pos_mae = total_pos_abs / (total_pos_count + 1e-12)
    
    valid_loss = running_loss.get_average()
    valid_loss_pos = running_loss_pos.get_average()
    valid_loss_cell = running_loss_cell.get_average() if running_loss_cell is not None else None

    model.train()

    return {
        "loss": valid_loss,
        "loss_pos": valid_loss_pos,
        "loss_cell": valid_loss_cell,
        "pos_mae": valid_pos_mae,
        "cell_mae": valid_cell_mae,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--dataset', type=str, default='xmno', choices=['xmno', 'mp', 'c2db', 'vdW', 'jarvis'], help='dataset to train on')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_norm', type=int, default=50, help='max_norm for clip_grad_norm')
    parser.add_argument('--epochs', type=int, default=800, help='epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=800, help='steps_per_epoch')
    parser.add_argument('--early_stop_epoch', type=int, default=10, help='early_stop_epoch')
    parser.add_argument('--save_model', action='store_true', help='Save the model after training')

    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers
    batch_size = args.batch_size
    max_norm = args.max_norm
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    early_stop_epoch = args.early_stop_epoch
    save_model = args.save_model
    dataset = args.dataset

    train_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'train')})
    valid_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, 'val')})
    collate_function = collate_fn
    use_cell_loss = True

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_function, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_function, num_workers=num_workers)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f'E3RelaxH2_{dataset}_{timestamp}'
    wandb.init(project="E3RelaxH2", 
            group=f"{dataset}",
            config={"train_len" : len(train_set), "valid_len" : len(valid_set)}, 
            name=log_name,
            id=log_name
            )

    device = torch.device('cuda:0')
    model = E3RelaxH2(
        hidden_channels=512,
        num_layers=4,
        num_rbf=128,
        cutoff=30.0,
        num_elements=118,
        return_cell=use_cell_loss,
    ).to(device)
    ema = ModelEMA(model, decay=0.999, warmup_steps=steps_per_epoch*5, device=device)
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 5, min_lr = 1.e-8)

    running_loss = AverageMeter()
    running_loss_pos = AverageMeter()
    running_loss_cell = AverageMeter() if use_cell_loss else None
    running_grad_norm = AverageMeter()
    running_best_loss = BestMeter("min")
    running_best_mae = BestMeter("min")

    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    global_step = 0
    global_epoch = 0

    break_flag = False
    model.train()

    for epoch in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1      

            data = data.to(device)
            outputs = model(data)
            if use_cell_loss:
                pos_pred_list, cell_pred_list = outputs
            else:
                pos_pred_list = outputs
                cell_pred_list = None

            pos_label = getattr(data, "pos_r", None)
            cell_label = getattr(data, "cell_r", None)
            if pos_label is None:
                pos_label = data.pos_relaxed
            if cell_label is None:
                cell_label = data.cell
            
            loss_pos = 0
            loss_cell = 0 if use_cell_loss else None
            for l, pos_pred in enumerate(pos_pred_list):
                loss_pos = loss_pos + compute_cart_mean_absolute_displacement_wrap_pred(pos_pred, pos_label, cell_label, data.batch)
                if use_cell_loss:
                    loss_cell = loss_cell + (cell_pred_list[l] - cell_label).abs().mean()
            loss = loss_pos + loss_cell if use_cell_loss else loss_pos
            
            if torch.isnan(loss):
                print(f"Skipping step {global_step} due to NaN loss")
                continue

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )
            optimizer.step()
            ema.update(model)

            running_loss.update(loss.item()) 
            running_loss_pos.update(loss_pos.item()) 
            if running_loss_cell is not None and loss_cell is not None:
                running_loss_cell.update(loss_cell.item())
            running_grad_norm.update(grad_norm.item())

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                train_loss = running_loss.get_average()
                train_loss_pos = running_loss_pos.get_average()
                train_loss_cell = running_loss_cell.get_average() if running_loss_cell is not None else None
                train_grad_norm = running_grad_norm.get_average()

                running_loss.reset()
                running_loss_pos.reset()
                if running_loss_cell is not None:
                    running_loss_cell.reset()
                running_grad_norm.reset()

                val_metrics = val(
                    ema,
                    valid_loader,
                    device,
                    use_cell_loss=use_cell_loss,
                )
                valid_pos_mae = val_metrics["pos_mae"]

                scheduler.step(valid_pos_mae)

                current_lr = optimizer.param_groups[0]['lr']

                log_dict = {
                    'train/epoch' : global_epoch,
                    'train/loss' : train_loss,
                    'train/loss_pos' : train_loss_pos,
                    'train/grad_norm' : train_grad_norm,
                    'train/lr' : current_lr,
                    'val/valid_loss' : val_metrics["loss"],
                    'val/valid_loss_pos' : val_metrics["loss_pos"],
                    'val/valid_pos_mae' : val_metrics["pos_mae"],
                }
                if train_loss_cell is not None:
                    log_dict['train/loss_cell'] = train_loss_cell
                if val_metrics["loss_cell"] is not None:
                    log_dict['val/valid_loss_cell'] = val_metrics["loss_cell"]
                if val_metrics["cell_mae"] is not None:
                    log_dict['val/valid_cell_mae'] = val_metrics["cell_mae"]
                wandb.log(log_dict)
                msg = "Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Pos MAE: {:.4f}".format(
                    global_epoch, epochs, train_loss, val_metrics["loss"], valid_pos_mae
                )
                if val_metrics["cell_mae"] is not None:
                    msg += ", Valid Cell MAE: {:.4f}".format(val_metrics["cell_mae"])
                print(msg)
                
                if valid_pos_mae < running_best_mae.get_best():
                    running_best_mae.update(valid_pos_mae)
                    if save_model:
                        save_ckpt(os.path.join(wandb.run.dir, "model.pt"), ema, model, optimizer, epoch=global_epoch, step=global_step)
                    
                else:
                    count = running_best_mae.counter()
                    if count > early_stop_epoch:
                        best_mae = running_best_mae.get_best()
                        print(f"early stop in epoch {global_epoch}")
                        print("best_mae: ", best_mae)
                        break_flag = True
                        break

    wandb.finish()
