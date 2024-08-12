import torchmetrics
import torch
from torch import nn

metric = torchmetrics.Accuracy(task="multiclass", num_classes=37)
batch_size = 8

def train_loop(device, dataloader, model, loss_fn, optimizer, epochs, epoch=None, debug=True):
    """Trains an epoch of the model
    
    Parameters:
        - `device`: destination device
        - `dataloader`: the dataloader of the dataset
        - `model`: the model used
        - `loss_fn`: the loss function of the model
        - `optimizer`: the optimizer
        - `epoch`: the index of the epoch
    """
    size = len(dataloader)
    predictions = []
    
    # Get the item from the dataset
    for item, (x, y) in enumerate(dataloader):
        print(f"{x} -> {x.shape}")
        #for index, video in enumerate(x):
            # Move data to the device used
        video = x.to(device)
        label = y.to(device)

        # Compute the prediction and the loss
        pred = model(video)
        predictions.append(pred)

            # if debug: print(video, video.shape, pred, pred.shape, label, label.shape, sep="\n\n========================================================\n\n")
            # total_acc = metric(pred, label)

    predictions = torch.stack(predictions)
    preds_shape = predictions.shape
    predictions = torch.reshape(predictions, (preds_shape[1], preds_shape[0], preds_shape[2]))

    """print(
        f"Predictions:\n{predictions}\n\nSize of predictions: {preds_shape}",
        f"Labels:\n{y}\n\nLabels shape: {y.shape}",
        f"Input size:\n{torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long)}",
        f"Labels size:\n{torch.full(size=(batch_size, ), fill_value=37, dtype=torch.long)}",
        sep="\n\n===============================================\n\n"
    )"""

    loss = loss_fn(
        predictions,
        y,
        torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long),
        torch.full(size=(batch_size, ), fill_value=37, dtype=torch.long)
    )

    # Adjust the weights
    # mean_loss = total_loss//batch_size
    # avg_acc=total_acc//batch_size
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Print some information
    
    if debug: print(f"→ Loss: {loss} [Item {item + 1}/{size}, Epoch {epoch + 1}/{epochs}]")
    # if debug: print(f"Accuracy of item {item}/{size}: {GNLAccuracy(predictions, y)}")
        
    #accuracy = metric.compute()
    print(f"===     The epoch {epoch + 1}/{epochs} has finished training     ===")
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