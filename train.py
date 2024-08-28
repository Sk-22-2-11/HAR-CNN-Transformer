import torch
import torch.nn as nn
import argparse
from src import load_data_n_model

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1)
            epoch_accuracy += (predict_y == labels).sum().item() / labels.size(0)
        
        epoch_loss /= len(tensor_loader.dataset)
        epoch_accuracy /= len(tensor_loader)
        print(f'Epoch {epoch+1}, Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.9f}')
        
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for HAR')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., LeNet, Cnn, etc.)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Widar)')
    parser.add_argument('--root', type=str, default='./Data/', help='Root directory of the dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, args.root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    model.to(device)
    
    train(
        model=model,
        tensor_loader=train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        criterion=criterion,
        device=device
    )
