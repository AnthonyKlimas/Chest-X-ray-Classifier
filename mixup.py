
"""
    Applies Mixup only to samples that belong to at least one minority class.
    Non-minority samples pass through unchanged.

    Args:
        imgs          : (B, C, H, W) tensor on device
        lbls          : (B, num_classes) float tensor on device
        minority_cols : list of class column indices considered minority
        alpha         : Beta distribution concentration parameter

    Returns:
        mixed_imgs, mixed_lbls  (same shape, same device)
"""
def mixup_minority(imgs, lbls, minority_cols, alpha=0.4):
    
    # Boolean mask: which items in the batch are minority-class samples
    is_minority = lbls[:, minority_cols].sum(dim=1) > 0   # (B,)

    if not is_minority.any():
        return imgs, lbls

    lam      = np.random.beta(alpha, alpha)
    rand_idx = torch.randperm(imgs.size(0), device=imgs.device)

    mixed_imgs = imgs.clone()
    mixed_lbls = lbls.clone()

    min_idx = is_minority.nonzero(as_tuple=True)[0]      # indices of minority rows
    pair_idx = rand_idx[min_idx]                          # their random partners

    mixed_imgs[min_idx] = lam * imgs[min_idx] + (1 - lam) * imgs[pair_idx]
    mixed_lbls[min_idx] = lam * lbls[min_idx] + (1 - lam) * lbls[pair_idx]

    return mixed_imgs, mixed_lbls