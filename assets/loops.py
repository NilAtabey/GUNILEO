import torchmetrics
import torch

metric = torchmetrics.Accuracy(task="multiclass", num_classes=37)
batch_size = 16

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
        print(x)
        print(x.shape)
        print(y)
        print(y.shape)
        x = x.to(device)
        y = y.to(device)
        
        
        
        # Move data to the device used
           

            # Compute the prediction and the loss
        pred = model(x)
        loss = loss_fn(pred, y)
        total_acc = metric(pred, y)

        # Adjust the weights
        mean_loss = total_loss//batch_size
        avg_acc=total_acc//batch_size
        mean_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print some information
        
        if debug: print(f"→ Loss: {mean_loss} [Batch {batch}/{size}, Epoch {epoch}/{epochs}]")
        
        if debug: print(f"Accuracy of batch {batch}/{size}: {avg_acc}")
        
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