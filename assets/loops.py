import torchmetrics
metric = torchmetrics.Accuracy(task="multiclass", num_classes=37)

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
    for batch, (x, y) in enumerate(dataloader):
        # Move data to the device used
        x = x.to(device)
        y = y.to(device)

        # Compute the prediction and the loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # Adjust the weights
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print some information
        if batch % 32 == 0:
            loss_value, current_batch = loss.item(), (batch + 1) * len(x)
            if debug: print(f"→ Loss: {loss_value} [Batch {current_batch}/{size}, Epoch {epoch}/{epochs}]")
            accuracy = metric(pred, y)
            if debug: print(f"Accuracy of batch {current_batch}/{size}: {accuracy}")
        
    accuracy = metric.compute()
    print(f"=== The epoch {epoch}/{epochs} has finished training ===")
    if debug: print(f"→ Final accuracy of the epoch: {accuracy}")
    metric.reset()

def test_loop(dataloader, model, loss_fn, debug=True):
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