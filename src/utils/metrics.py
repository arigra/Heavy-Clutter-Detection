def compute_metrics(pred_mask, rd_label, eps=1e-8):
    pred_mask = pred_mask.bool()
    rd_label = rd_label.bool()

    tp = (pred_mask & rd_label).sum().item()
    fp = (pred_mask & ~rd_label).sum().item()
    fn = (~pred_mask & rd_label).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    dice = (2 * tp) / (2*tp + fp + fn + eps)

    return precision, recall, f1, dice
