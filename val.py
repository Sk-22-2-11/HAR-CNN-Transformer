import torch
import torch.nn as nn
import argparse
from src import load_data_n_model

def evaluate(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    
    with torch.no_grad():
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            outputs = model(inputs)
            outputs = outputs.type(torch.FloatTensor)
            
            loss = criterion(outputs, labels)
            predict_y = torch.argmax(outputs, dim=1)
            accuracy = (predict_y == labels).sum().item() / labels.size(0)
            
            test_acc += accuracy
            test_loss += loss.item() * inputs.size(0)
    
    test_acc /= len(tensor_loader)
    test_loss /= len(tensor_loader.dataset)
    
    print(f"Validation accuracy: {test_acc:.4f}, Loss: {test_loss:.5f}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model for HAR')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., LeNet, Cnn, etc.)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Widar)')
    parser.add_argument('--root', type=str, default='./Data/', help='Root directory of the dataset')
    
    args = parser.parse_args()
    
    train_loader, test_loader, model, _ = load_data_n_model(args.dataset, args.model, args.root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    model.to(device)
    
    evaluate(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )
