import torch
from tqdm import tqdm
from args import args


def validate_signal_data(val_loader, model, criterion, dataset_sizes):
    # switch to evaluate mode
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    signal_num = 0

    # Iterate over test data.
    pbar = tqdm(val_loader)
    for inputs, labels in pbar:
        inputs = inputs.cuda()
        labels = labels.cuda()

        with torch.no_grad():  # No need to track history during inference
            if args.arch in ['CNN1D_KD', 'CNN1D', 'SigNet50', 'SigNet50_KD']:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            else:
                l1_regularization = torch.tensor(0.).cuda()
                for param in model.parameters():
                    l1_regularization += torch.norm(param, 1)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels) + 0.0001 * l1_regularization

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        signal_num += inputs.size(0)

        # Display real-time dataset type, loss value, and accuracy on the right side of the progress bar
        epoch_loss = running_loss / signal_num
        epoch_acc = running_corrects.double() / signal_num
        pbar.set_postfix({'Set': 'test',
                          'Loss': '{:.4f}'.format(epoch_loss),
                          'Acc': '{:.4f}'.format(epoch_acc)})

    # Display loss and accuracy for the entire test set
    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc
