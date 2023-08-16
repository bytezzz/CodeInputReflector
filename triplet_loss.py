import torch

#Reimplemented the loss computing with PyTorch
#Reference: https://github.com/cs-sun/InputReflector/blob/main/triplet_loss.py

def _pairwise_distances(ebd, ebd_an, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        ebd: tensor of shape (batch_size, embed_dim)
        ebd_an: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(ebd, torch.transpose(ebd_an, 0, 1))

    square_norm_a = torch.diag(torch.matmul(ebd, torch.transpose(ebd, 0, 1)))
    square_norm_b = torch.diag(torch.matmul(ebd_an, torch.transpose(ebd_an, 0, 1)))
    distances = torch.unsqueeze(square_norm_a, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm_b, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.maximum(distances, torch.tensor(0.0))

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).to(torch.float)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: torch.int32 Tensor with shape [batch_size]
    Returns:
        mask: torch.bool Tensor with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool).to(labels.device)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: torch.int32 Tensor with shape [batch_size]
    Returns:
        mask: torch.bool Tensor with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

    mask = ~labels_equal

    return mask

def _get_negative_negative_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] != labels[j] and labels[i] != labels[k] and labels[j] != labels[k]
    Args:
        labels: torch.int32 Tensor with shape [batch_size]
    Returns:
        mask: torch.bool Tensor with shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool).to(labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    # Check if labels[i] != labels[j] and labels[i] != labels[k] and labels[j] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    j_equal_k = label_equal.unsqueeze(0)

    valid_labels = ~(i_equal_j | i_equal_k | j_equal_k)

    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: torch.int32 Tensor with shape [batch_size]
    Returns:
        mask: torch.bool Tensor with shape [batch_size, batch_size, batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool).to(labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = i_equal_j & ~i_equal_k

    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask

def batch_hard_triplet_loss(labels, ebd_anchor, ebd_positive, ebd_negative, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist_ap = _pairwise_distances(ebd_positive, ebd_anchor, squared=squared)
    pairwise_dist_an = _pairwise_distances(ebd_negative, ebd_anchor, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist_ap

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist_an, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist_an + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_c1c2_c1c(labels, ebd_anchor, ebd_positive, ebd_negative, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist_ap = _pairwise_distances(ebd_positive, ebd_anchor, squared=squared)
    pairwise_dist_an = _pairwise_distances(ebd_negative, ebd_anchor, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist_ap

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should be same images)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_anchor_negative = labels_equal.float()

    # We add the maximum value in each row to the invalid negatives (different images / label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist_an, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist_an + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_c1c_c1c2(labels, ebd_anchor, ebd_positive, ebd_negative, margin, squared=False):
    pairwise_dist_ap = _pairwise_distances(ebd_positive, ebd_anchor, squared=squared)
    pairwise_dist_an = _pairwise_distances(ebd_negative, ebd_anchor, squared=squared)

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_anchor_positive = labels_equal.float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist_ap

    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    mask_anchor_negative = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    max_anchor_negative_dist, _ = torch.max(pairwise_dist_an, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist_an + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist - margin, min=0.0)

    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_cde(labels, ebd_anchor, ebd_positive, ebd_negative, margin, squared=False):
    pairwise_dist_ap = _pairwise_distances(ebd_positive, ebd_anchor, squared=squared)
    pairwise_dist_an = _pairwise_distances(ebd_negative, ebd_anchor, squared=squared)

    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.float()

    anchor_positive_dist = mask_anchor_positive * pairwise_dist_ap

    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    mask_negative_negative = _get_negative_negative_triplet_mask(labels)
    mask_negative_negative = mask_negative_negative.float()

    max_anchor_negative_dist, _ = torch.max(pairwise_dist_an, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist_an + max_anchor_negative_dist * (1.0 - mask_negative_negative)

    min_values, _ = torch.min(anchor_negative_dist, dim=2, keepdim=True)
    hardest_negative_dist, _ = torch.min(min_values, dim=1)

    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss

def batch_hard_triplet_loss_new(labels, ebd_anchor, ebd_positive, ebd_negative, margin, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist_ap = _pairwise_distances(ebd_positive, ebd_anchor, squared=squared)
    pairwise_dist_an = _pairwise_distances(ebd_negative, ebd_anchor, squared=squared)

    # For each anchor, get the hardest positive
    # Check that i and j are the same
    indices_equal = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    mask_anchor_positive = indices_equal.float()

    # We put to 0 any element where (a, p) is not valid (valid if a == p)
    anchor_positive_dist = mask_anchor_positive * pairwise_dist_ap

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have the same labels)
    mask_anchor_negative = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.float()

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist_an, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist_an + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into the final triplet loss
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss
