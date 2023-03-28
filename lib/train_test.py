import torch, numpy
from sklearn.metrics import accuracy_score

def train(model, train_data, val_data, num_epochs, best_model_path, device, optimizer_parameters,epoch_val):    

    best_acc = 0.0
    # Variables to store the results
    losses = []
    acc_train = []
    losses_val = []
    acc_val = []
    pred_label_train = torch.empty((0)).to(device)    # .to(device) to move the data/model on GPU or CPU (default)
    true_label_train = torch.empty((0)).to(device)
    pred_label_val = torch.empty((0)).to(device)
    true_label_val = torch.empty((0)).to(device)
    ### CREATE MODEL ###

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr = optimizer_parameters[0], momentum=optimizer_parameters[1], weight_decay=0.1)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)    # .to(device) to move the data/model on GPU or CPU (default)

    ### FIT MODEL ###
    for epoch in range(num_epochs):
        # Train step
        model.train()                                                   # tells to the model you are in training mode (batchnorm and dropout layers work)
        for data_tr in train_data:
            optim.zero_grad()

            x,y = data_tr                        # unlist the data from the train set
            x = x.float().to(device)     # change the size for the input data - convert to float type
            y = y.to(torch.int64).to(device)

            y_pred = numpy.squeeze(model(x))                                        # run the model()
            loss = criterion(y_pred,y)                               # compute loss
            _,pred = y_pred.max(1)                                      # get the index == class of the output along the rows (each sample)
            pred_label_train = torch.cat((pred_label_train,pred),dim=0)
            true_label_train = torch.cat((true_label_train,y),dim=0)
            loss.backward()                                             # compute backpropagation
            optim.step()                                                # parameter update

        losses.append(loss.cpu().detach().numpy())
        acc_t = accuracy_score(true_label_train.cpu(),pred_label_train.cpu())
        acc_train.append(acc_t)
        print(f'Epoch: {epoch+1}/{num_epochs}, loss = {loss:.4f} - acc = {acc_t:.4f}')
        
        if acc_t > best_acc:                                                            # save the best model (the highest accuracy in validation)
            torch.save(model.state_dict(), best_model_path)
            best_acc = acc_t

        if val_data is not None and epoch % epoch_val == 0:                                                             # compute the accuracy on validation set
            model.eval()
            with torch.no_grad():
                for data in val_data:
                    x_te, y_te = data
                    x_te = x_te.float().to(device)
                    y_te = y_te.to(torch.int64).to(device)

                    y_pred = numpy.squeeze(model(x_te))
                    _,pred = y_pred.max(1)
                    pred_label_val = torch.cat((pred_label_val,pred),dim=0)
                    true_label_val = torch.cat((true_label_val,y_te),dim=0)
                    loss = criterion(y_pred,y_te)
            losses_val.append(loss.cpu().detach().numpy())
            acc_val.append(accuracy_score(true_label_val.cpu(),pred_label_val.cpu()))
            print(f'Validation loss = {loss:.4f}')
        

        # Reinitialize the variables to compute accuracy
        pred_label_train = torch.empty((0)).to(device)
        true_label_train = torch.empty((0)).to(device)
        pred_label_val = torch.empty((0)).to(device)
        true_label_val = torch.empty((0)).to(device)
    
    return {'loss': losses, 'accuracy': acc_train, 'loss_val': losses_val, 'accuracy_val': acc_val}

def test(model, test_data, device): 

    ### TEST MODEL ###

    pred_label_test = torch.empty((0,)).to(device)
    true_label_test = torch.empty((0)).to(device)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for data in test_data:
            X_te, y_te = data
            X_te = X_te.float().to(device)
            y_te = y_te.to(device)

            output_test = model(X_te)
            pred_label_test = torch.cat((pred_label_test,output_test),dim=0)
            true_label_test = torch.cat((true_label_test,y_te),dim=0)

    return numpy.squeeze(true_label_test.cpu()), numpy.squeeze(pred_label_test.cpu())
