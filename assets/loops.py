import numpy as np
import torchmetrics
import torch
from torch import nn

# metric = torchmetrics.Accuracy(task="multiclass", num_classes=38)
batch_size = 32

def train_loop(device, dataloader, model, loss_fn, optimizer, batch_index: int, epochs: int, epoch: int, debug: bool=True):
    """Trains an epoch of the model

    Parameters:
        - `device`: destination device
        - `dataloader`: the dataloader of the dataset
        - `model`: the model used
        - `loss_fn`: the loss function of the model
        - `optimizer`: the optimizer
        - `batch_index`: the number of the currently processed batch
        - `epochs`: the number of epochs
        - `epoch`: the index of the epoch
        - `debug`: (default `True`): prints debug info
    """
    size = len(dataloader)
    predictions = torch.zeros((batch_size, 75, 38)).to(device)  #np.ndarray(shape=(batch_size, 75, 38))
    labels = torch.zeros((batch_size, 37)).to(device)  #np.ndarray(shape=(batch_size, 37))

    print(f"Test: {predictions[2]}")

    # Get the item from the dataset
    for item, (x, y) in enumerate(dataloader):
        #print(f"{x} -> {x.shape}")
        #for index, video in enumerate(x):
            # Move data to the device used
        video = x.to(device)
        label = y.to(device)

        # Compute the prediction and the loss
        pred = model(video)
        predictions[item] = pred
        labels[item] = label

            # if debug: print(video, video.shape, pred, pred.shape, label, label.shape, sep="\n\n========================================================\n\n")
            # total_acc = metric(pred, label)

        if debug: print(f"[DEBUG] Preds: {pred.shape}\n[DEBUG] Label: {label.shape}")

    loss = loss_fn(
        predictions,
        labels,
        torch.full(size=(batch_size, 75, 38), fill_value=75, dtype=torch.long),   # torch.Size([32])
        torch.full(size=(batch_size, 37), fill_value=37, dtype=torch.long)    # torch.Size([32])
    )

    # Adjust the weights
    # mean_loss = total_loss//batch_size
    # avg_acc=total_acc//batch_size
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if debug: print(f"→ Loss: {loss} [Batch {batch_index + 1}/{size}, Epoch {epoch + 1}/{epochs}]")

    """predictions = torch.stack(predictions)
    labels = torch.stack(labels)
    preds_shape = predictions.shape
    labels_shape = labels.shape
    predictions = torch.reshape(predictions, (preds_shape[1], preds_shape[0], preds_shape[2]))
    """

    """
    print(
    f"Predictions:\n{predictions}\n\nSize of predictions: {preds_shape}",
    f"Labels:\n{y}\n\nLabels shape: {y.shape}",
    f"Input size:\n{torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long)}",
    f"Labels size:\n{torch.full(size=(batch_size, ), fill_value=37, dtype=torch.long)}",
    sep="\n\n===============================================\n\n"
    )
    """


    # Print some information

    # if debug: print(f"Accuracy of item {item}/{size}: {GNLAccuracy(predictions, y)}")

    #accuracy = metric.compute()
    print(f"===     The batch {epoch + 1}/{epochs} has finished training     ===")
    #if debug: print(f"→ Final accuracy of the epoch: {accuracy}")
    #metric.reset()


def GNLAccuracy(preds, labels) -> float:
    alphabet = [x for x in "abcdefghijklmnopqrstuvwxyz0123456789 "]
    total = 0
    for index, video in enumerate(preds):
        correct = 0
        pred_label = []
        label = [i for i in labels[index] if i != " "]
        for frame in video:
            letter = alphabet[torch.argmax(frame)]
            if letter != " ": pred_label.append(letter)

        for i, c in enumerate(pred_label):
            if c == label[i]:
                correct += 1
        total += correct / len(pred_label)
    return total / batch_size


def test_loop(device, dataloader, model, loss_fn, debug=True):
    size = len(dataloader)

    # Disable the updating of the weights
    with torch.no_grad():
        for index, (x, y) in enumerate(dataloader):
            for index, video in enumerate(x):
            # Move the data to the device used for testing
                video = video.to(device)
                label = y[index].to(device)

                # Get the model prediction
                pred = model(video)

                # Get the accuracy score
                # acc = metric(pred, label)
                # if debug: print(f"→ Accuracy for image {index}: {acc}")
    # acc = metric.compute()
    print(f"===        The testing loop has finished        ===")
    # if debug: print(f"→ Final testing accuracy of the model: {acc}")
    # metric.reset()
