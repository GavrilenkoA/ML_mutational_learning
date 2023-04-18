
import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def training(model, criterion, optimizer, num_epochs, trainloader, testloader = None):
    loss_train_hist = []
    loss_valid_hist = []
    for e in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for batch in trainloader:
            optimizer.zero_grad()
            input_embeds, labels = batch
            #input_embeds = input_embeds.float().to(device)
            labels = labels.unsqueeze(1).to(device)
            labels = labels.float()
            prediction = model(input_embeds)
            loss = criterion(prediction, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        loss_train_hist.append(train_loss/len(trainloader))
        if testloader is not None:
            valid_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in testloader:
                    input_embeds, labels = batch
                    #input_embeds = input_embeds.float().to(device)
                    labels = labels.unsqueeze(1).to(device)
                    labels = labels.float()
                    prediction = model(input_embeds)
                    loss = criterion(prediction, labels)
                    valid_loss += loss.item()
                loss_valid_hist.append(valid_loss/len(testloader))
    return loss_train_hist, loss_valid_hist