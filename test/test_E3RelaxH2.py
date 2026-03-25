# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import time
import torch
import pandas as pd
from model.E3RelaxH2 import E3RelaxH2
from utils.lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from torch.utils.data import DataLoader
from model.ema import load_ckpt_for_resume
from collections import defaultdict
from pymatgen.analysis.structure_matcher import StructureMatcher
import argparse
from utils.utils import create_dir, compute_cart_mean_absolute_displacement_wrap_pred
from pymatgen.core import Structure

# %%

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='custom', help='dataset name for output filename')
    parser.add_argument('--data_root', type=str, required=True, help='dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--test_subdir', type=str, default='test', help='test split subdirectory under data_root')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='number of dataloader workers')
    parser.add_argument('--results_dir', type=str, default='./results', help='directory to save csv results')
    parser.add_argument('--device', type=str, default='cuda:0', help='torch device string, e.g. cuda:0 or cpu')
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    data_root = args.data_root
    model_path = args.model_path
    test_subdir = args.test_subdir
    batch_size = args.batch_size
    num_workers = args.num_workers
    results_dir = args.results_dir

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    test_set = TrajectoryLmdbDataset({"src": os.path.join(data_root, test_subdir)})
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    model = E3RelaxH2(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema = load_ckpt_for_resume(model_path, model, device)

    model = model.to(device)
    model.eval()

    performance_dict = defaultdict(list)
    matcher = StructureMatcher() # Initialize a StructureMatcher object
    for data in test_loader:
        with torch.no_grad():
            data = data.to(device)

            start_time = time.time() # Record the starting time

            pos_pred_list, cell_pred_list = ema(data)
            pos_pred, cell_pred = pos_pred_list[-1], cell_pred_list[-1]

            # keep batch dimension for single-graph batches
            if cell_pred.dim() == 2:
                cell_pred = cell_pred.unsqueeze(0)

            end_time = time.time()
            elapsed_time = end_time - start_time # Calculate the time elapsed

            pos_r = data.pos_r
            pos_u = data.pos_u
            cell_u = data.cell_u
            cell_r = data.cell_r

            if cell_u.dim() == 2:
                cell_u = cell_u.unsqueeze(0)
            if cell_r.dim() == 2:
                cell_r = cell_r.unsqueeze(0)

            batch_indices = data.batch
            unique_graphs = batch_indices.unique(sorted=True)

            for graph_idx in unique_graphs.cpu().tolist():
                mask = batch_indices == graph_idx
                if mask.sum().item() == 0:
                    continue

                # remap batch indices so compute_cart_* sees a single-graph batch
                single_batch = torch.zeros_like(batch_indices[mask])
                single_cell_r = cell_r[graph_idx].unsqueeze(0)

                mae_pos_Dummy = compute_cart_mean_absolute_displacement_wrap_pred(
                    pos_u[mask], pos_r[mask], single_cell_r, single_batch
                ).item()
                mae_pos_E3RelaxH2 = compute_cart_mean_absolute_displacement_wrap_pred(
                    pos_pred[mask], pos_r[mask], single_cell_r, single_batch
                ).item()

                cell_u_graph = cell_u[graph_idx]
                cell_r_graph = cell_r[graph_idx]
                cell_pred_graph = cell_pred[graph_idx]

                metric_tensor_unrelaxed = cell_u_graph @ cell_u_graph.T
                metric_tensor_relaxed = cell_r_graph @ cell_r_graph.T
                metric_tensor_predicted = cell_pred_graph @ cell_pred_graph.T

                mae_metric_tensor_Dummy = torch.norm(
                    metric_tensor_unrelaxed - metric_tensor_relaxed, p='fro'
                ).item()
                mae_metric_tensor_E3RelaxH2 = torch.norm(
                    metric_tensor_predicted - metric_tensor_relaxed, p='fro'
                ).item()

                mae_volume_Dummy = (torch.linalg.det(cell_u_graph) - torch.linalg.det(cell_r_graph)).abs().item()
                mae_volume_E3RelaxH2 = (torch.linalg.det(cell_pred_graph) - torch.linalg.det(cell_r_graph)).abs().item()

                atomic_numbers = getattr(data, 'atomic_numbers', None)
                if atomic_numbers is None:
                    atomic_numbers = getattr(data, 'x', None)

                structure_match_unrelaxed_gt = False
                structure_match_pred_gt = False
                if atomic_numbers is not None:
                    try:
                        species_graph = atomic_numbers[mask].detach().cpu().long().view(-1).tolist()

                        structure_gt = Structure(
                            lattice=cell_r_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_r[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_pred = Structure(
                            lattice=cell_pred_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_pred[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_unrelaxed = Structure(
                            lattice=cell_u_graph.detach().cpu().numpy(),
                            species=species_graph,
                            coords=pos_u[mask].detach().cpu().numpy(),
                            coords_are_cartesian=True,
                        )
                        structure_match_unrelaxed_gt = matcher.fit(structure_unrelaxed, structure_gt)
                        structure_match_pred_gt = matcher.fit(structure_pred, structure_gt)
                    except Exception:
                        structure_match_unrelaxed_gt = None
                        structure_match_pred_gt = None

                performance_dict['cif_id'].append(data.cif_id[graph_idx])
                performance_dict['mae_pos_Dummy'].append(mae_pos_Dummy)
                performance_dict['mae_pos_E3RelaxH2'].append(mae_pos_E3RelaxH2)
                performance_dict['mae_metric_tensor_Dummy'].append(mae_metric_tensor_Dummy)
                performance_dict['mae_metric_tensor_E3RelaxH2'].append(mae_metric_tensor_E3RelaxH2)
                performance_dict['mae_volume_Dummy'].append(mae_volume_Dummy)
                performance_dict['mae_volume_E3RelaxH2'].append(mae_volume_E3RelaxH2)
                performance_dict['structure_match_unrelaxed_gt'].append(structure_match_unrelaxed_gt)
                performance_dict['structure_match_pred_gt'].append(structure_match_pred_gt)
                performance_dict['elapsed_time'].append(elapsed_time)

    create_dir([results_dir])
    performance_df = pd.DataFrame(performance_dict)
    performance_df.to_csv(os.path.join(results_dir, f"{dataset}_E3RelaxH2.csv"), index=False)

    avg_metrics = {
        'avg_mae_pos_Dummy': performance_df['mae_pos_Dummy'].mean(),
        'avg_mae_pos_E3RelaxH2': performance_df['mae_pos_E3RelaxH2'].mean(),
        'avg_mae_metric_tensor_Dummy': performance_df['mae_metric_tensor_Dummy'].mean(),
        'avg_mae_metric_tensor_E3RelaxH2': performance_df['mae_metric_tensor_E3RelaxH2'].mean(),
        'avg_mae_volume_Dummy': performance_df['mae_volume_Dummy'].mean(),
        'avg_mae_volume_E3RelaxH2': performance_df['mae_volume_E3RelaxH2'].mean(),
        'avg_elapsed_time': performance_df['elapsed_time'].mean(),
    }

    unrelaxed_match = pd.to_numeric(performance_df['structure_match_unrelaxed_gt'], errors='coerce')
    pred_match = pd.to_numeric(performance_df['structure_match_pred_gt'], errors='coerce')
    avg_metrics['avg_structure_match_unrelaxed_gt'] = unrelaxed_match.mean()
    avg_metrics['avg_structure_match_pred_gt'] = pred_match.mean()

    print("\n===== Final Average Performance =====")
    for k, v in avg_metrics.items():
        if pd.isna(v):
            print(f"{k}: NaN")
        else:
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()


# %%
