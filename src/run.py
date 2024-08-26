def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        i = 0
        #print('iter_no:{}'.format(len(iter(tensor_loader))))
        for data in iter(tensor_loader):
            
            #print('Epoch:{},Image{}'.format(epoch,i))
            i+=1
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs,dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss/len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy/len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch+1, float(epoch_accuracy),float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device):
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        
        loss = criterion(outputs,labels)
        predict_y = torch.argmax(outputs,dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc),float(test_loss)))
    return

    
root = './Data/'
train_loader, test_loader, model, train_epoch = load_data_n_model("Widar", "LeNet", root)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

train(
    model=model,
    tensor_loader= train_loader,
    num_epochs= train_epoch,
    learning_rate=1e-4,
    criterion=criterion,
    device=device
     )
test(
    model=model,
    tensor_loader=test_loader,
    criterion=criterion,
    device= device
    )


