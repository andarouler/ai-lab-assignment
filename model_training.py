# model_training.py

import torch
import torch.nn as nn

from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from simple_cnn import SimpleCNN
from datasets import WeatherDataset, ArtStyleDataset

def get_dataloaders(classifier_type, train_dir, test_dir, batch_size=32):
    # Transformationen für die Bilder
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    if classifier_type == 'weather':
        # dataset for weather
        print("Loading weather training dataset")
        train_dataset = WeatherDataset(root_dir=train_dir, transform=transform)
        print("Loading weather test dataset")
        test_dataset = WeatherDataset(root_dir=test_dir, transform=transform)

    elif classifier_type == 'artstyle':
        # Dataset for art style ( realistic vs. artistic)
        print("Loading artstyle training dataset")
        train_dataset = ArtStyleDataset(root_dir=train_dir, transform=transform)
        print("Loading artstyle test datatset")
        test_dataset = ArtStyleDataset(root_dir=test_dir, transform=transform)

    # Erstellen des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.to(device))
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return test_loss / len(test_loader), accuracy


def train_and_evaluate(classifier_type, model_name, num_epochs=10, batch_size=32):
    print(f"Training classifier for: {classifier_type}")

    # Hauptlogik für das Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training")

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Lade die Datasets mit der Funktion aus dataset_dataloader.py
    train_loader, test_loader = get_dataloaders(classifier_type, '../dataset/train', '../dataset/test', batch_size)

    # num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Speichern des trainierten Modells
    print(f"Saving trained model: model/{model_name}.pth")
    torch.save(model.state_dict(), f'model/{model_name}.pth')
    print("Finished!\n")

train_and_evaluate(
    classifier_type='weather',
    model_name='rainy_sunny_classifier',
    num_epochs=10
)

train_and_evaluate(
    classifier_type='artstyle',
    model_name='artistic_realistic_classifier',
    num_epochs=10
)