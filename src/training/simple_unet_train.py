def train_simple_unet(num_epoches,)
train_losses = []
val_losses = []
val_precisions = []
val_recalls = []
val_f1s = []
val_dices = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        S, C, IQ_map, clean_RD_map, RD_map, rd_label = batch

        # Prepare inputs for IQ map
        IQ_map_real = IQ_map.real.unsqueeze(1).cuda()  # [B, 1, N, K]
        IQ_map_imag = IQ_map.imag.unsqueeze(1).cuda()
        IQ_map_input = torch.cat([IQ_map_real, IQ_map_imag], dim=1) # [B, 2, N, K]

        rd_label = rd_label.unsqueeze(1).float().cuda() # [B, 1, N, K]

        # Forward pass
        y_pred_logits = model(IQ_map_input)  # [B, 1, N, K] logits
        B, C, H, W = y_pred_logits.shape
        y_pred_flat = y_pred_logits.view(B, H*W)
        y_true_flat = rd_label.view(B, H*W)

        loss = criterion(y_pred_flat, y_true_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * B

        with torch.no_grad():
            pred_probs = torch.sigmoid(y_pred_logits)
            pred_mask = (pred_probs > 0.5).float()
            correct = (pred_mask == rd_label).sum().item()
            total = rd_label.numel()
            train_correct += correct
            train_total += total

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total

    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_tp, val_fp, val_fn = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            S, C, IQ_map, clean_RD_map, RD_map, rd_label = batch
            IQ_map_real = IQ_map.real.unsqueeze(1).cuda()
            IQ_map_imag = IQ_map.imag.unsqueeze(1).cuda()
            IQ_map_input = torch.cat([IQ_map_real, IQ_map_imag], dim=1)

            rd_label = rd_label.unsqueeze(1).float().cuda()

            y_pred_logits = model(IQ_map_input)
            B, C, H, W = y_pred_logits.shape
            y_pred_flat = y_pred_logits.view(B, H*W)
            y_true_flat = rd_label.view(B, H*W)

            loss = criterion(y_pred_flat, y_true_flat)
            val_loss += loss.item() * B

            pred_probs = torch.sigmoid(y_pred_logits)
            pred_mask = (pred_probs > 0.5).float()
            correct = (pred_mask == rd_label).sum().item()
            total = rd_label.numel()
            val_correct += correct
            val_total += total

            # Compute TP, FP, FN for val
            pred_bool = pred_mask.bool()
            gt_bool = rd_label.bool()
            val_tp += (pred_bool & gt_bool).sum().item()
            val_fp += (pred_bool & ~gt_bool).sum().item()
            val_fn += (~pred_bool & gt_bool).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct / val_total
    val_precision, val_recall, val_f1, val_dice = compute_metrics(pred_mask, rd_label)

    # Append metrics to lists
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)
    val_dices.append(val_dice)

    wandb.log({
        "IQ_unet/train_loss": train_loss,
        "IQ_unet/val_loss": val_loss,
        "IQ_unet/val_precision": val_precision,
        "IQ_unet/val_recall": val_recall,
        "IQ_unet/val_f1": val_f1,
        "IQ_unet/val_dice": val_dice,
        "epoch": epoch + 1,
    })
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}")
    print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val Dice: {val_dice:.4f}")

    # Scheduler step - based on validation loss
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model so far
        torch.save(model.state_dict(), "IQ_unet.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            break

print("Training complete. Best validation loss: ", best_val_loss)

# Plotting the metrics
epochs = range(1, len(train_losses)+1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_losses, 'b', label='Train Loss')
plt.plot(epochs, val_losses, 'r', label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(epochs, val_precisions, 'g', label='Val Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(epochs, val_recalls, 'm', label='Val Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(epochs, val_f1s, 'c', label='Val F1')
plt.title('F1-Score')
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs, val_dices, 'y', label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice')
plt.legend()

plt.tight_layout()
plt.show()
