import torch
import torch.nn.functional as F

def calculate_uncertainty(sem_seg_logits):
    top2_scores, _ = torch.topk(sem_seg_logits, k=2, dim=1)
    return (top2_scores[:, 1] - top2_scores[:, 0])

def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def sample_uncertain_points(coarse_logits, dmap, num_points, oversample_factor, importance_sample_ratio):
    assert oversample_factor >= 1
    assert 0 <= importance_sample_ratio and importance_sample_ratio <= 1

    num_batches = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_factor)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    oversampled_point_coords = torch.rand(num_batches, num_sampled, 2, device=coarse_logits.device)
    oversampled_point_logits = point_sample(coarse_logits, oversampled_point_coords, align_corners=False)
    point_uncertainties = calculate_uncertainty(oversampled_point_logits)
    _, top_indices = torch.topk(point_uncertainties, k=num_uncertain_points, dim=1)
    uncertain_point_coords = torch.stack([oversampled_point_coords[i, j] for i, j in enumerate(top_indices)])

    # importance_weights = point_sample(dmap, uncertain_point_coords, align_corners=False)
    # importance_weights = torch.squeeze(importance_weights)
    # resample_factors = torch.round(torch.exp(importance_weights)).long()

    # all_resampled_point_coords = []
    # for i, fac in enumerate(resample_factors):
    #     resampled_importance_weights = torch.repeat_interleave(importance_weights[i], repeats=fac, dim=0)
    #     resampled_point_coords = torch.repeat_interleave(uncertain_point_coords[i], repeats=fac, dim=0)
    #     _, top_indices = torch.topk(resampled_importance_weights, k=num_uncertain_points, dim=0)
    #     top_indices = top_indices[torch.randperm(top_indices.nelement())]
    #     resampled_point_coords = resampled_point_coords[top_indices]
    #     all_resampled_point_coords.append(resampled_point_coords)
    # all_resampled_point_coords = torch.stack(all_resampled_point_coords, dim=0)

    random_point_coords = torch.rand(num_batches, num_random_points, 2, device=coarse_logits.device)
    
    # point_coords = torch.cat([all_resampled_point_coords, random_point_coords], dim=1)
    point_coords = torch.cat([uncertain_point_coords, random_point_coords], dim=1)
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    B, H, W = uncertainty_map.shape
    total_n_pixs, n_dims = H * W, 2
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(total_n_pixs, num_points)
    _, point_indices = torch.topk(uncertainty_map.view(B, total_n_pixs), k=num_points, dim=1)
    point_coords = torch.zeros(B, num_points, n_dims, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords