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
    
    # Get the batch from the dataset
    for batch, (x,y) in enumerate(dataloader):
        target_lengths = [i for i in y]
        preds = []
        for index, video in enumerate(x):
            # Move data to the device used
            video = video.to(device)
            label = y[index].to(device)

            # Compute the prediction and the loss
            pred = model(video)

            preds.append(pred)
            tar_lens = torch.randint(low=21, high=33, size=(batch_size, ), dtype=torch.long)    # torch.tensor(size=(batch_size, ), dtype=torch.long)
            #loss = loss_fn(pred, label, (75), (33))

            print(video, video.shape, pred, pred.shape, label, label.shape, sep=2*"\n")
            total_acc = metric(pred, label)

        loss = loss_fn(torch.tensor(preds), torch.tensor(y), torch.full(size=(batch_size, ), fill_value=75, dtype=torch.long), torch.full(size=(batch_size, ), fill_value=33, dtype=torch.long)) # torch.tensor(size=(33, ), dtype=torch.long)
        # Adjust the weights
        # mean_loss = total_loss//batch_size
        # avg_acc=total_acc//batch_size
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print some information
        
        if debug: print(f"→ Loss: {loss} [Batch {batch}/{size}, Epoch {epoch}/{epochs}]")
        if debug: print(f"Accuracy of batch {batch}/{size}: {total_acc}")
        
    accuracy = metric.compute()
    print(f"=== The epoch {epoch}/{epochs} has finished training ===")
    if debug: print(f"→ Final accuracy of the epoch: {accuracy}")
    metric.reset()

def test_loop(device, dataloader, model, loss_fn, debug=True):
    size = len(dataloader)

    # Disable the updating of the weights
    with torch.no_grad():
        for index, (x, y) in enumerate(dataloader):
            # Move the data to the device used for testing
            x = x.to(device)
            y = y.to(device)

            # Get the model prediction
            pred = model(x)

            # Get the accuracy score
            acc = metric(pred, y)
            if debug: print(f"→ Accuracy for image {index}: {acc}")
    acc = metric.compute()
    print(f"===    The testing loop has finished    ===")
    if debug: print(f"→ Final testing accuracy of the model: {acc}")
    metric.reset()