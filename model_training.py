# model_training.py

import torch
import torch.nn as nn

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

from simple_cnn import SimpleCNN
from datasets import WeatherDataset, ArtStyleDataset

def get_dataloaders(classifier_type, train_dir, test_dir, batch_size=32):
    
    # Transformationen der Bilder und
    # Data Augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),  
    ])

    if classifier_type == 'weather':
        # Dataset für weather
        print("Loading weather training dataset")
        train_dataset = WeatherDataset(root_dir=train_dir, transform=transform)
        print("Loading weather test dataset")
        test_dataset = WeatherDataset(root_dir=test_dir, transform=transform)

    elif classifier_type == 'artstyle':
        # Dataset für art style ( realistic vs. artistic)
        print("Loading artstyle training dataset")
        train_dataset = ArtStyleDataset(root_dir=train_dir, transform=transform)
        print("Loading artstyle test datatset")
        test_dataset = ArtStyleDataset(root_dir=test_dir, transform=transform)

    # Erstellen des DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Hole die Klassenlabels aus dem Dataset
    class_labels = train_dataset.classes
    
    return train_loader, test_loader, class_labels

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

    # Prüfe ob cude (gpu) verfügbar
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training")

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Lade die Datasets mit der Funktion aus dataset_dataloader.py
    train_loader, test_loader, class_labels = get_dataloaders(classifier_type, '../dataset/train', '../dataset/test', batch_size)

    # Überprüfen, ob Klassenlabels korrekt geladen wurden
    if class_labels is not None:
        print(f"Class labels for {classifier_type}: {class_labels}")
    else:
        print(f"No class labels found for {classifier_type}")

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Speichern des Modells und der Labels
    model_info = {
        'model_state_dict': model.state_dict(),
        'class_labels': class_labels  # Speichere die Labels zusammen mit dem Modell
    }

    # Speichern des trainierten Modells und der Labels
    print(f"Saving trained model and labels: model/{model_name}.pth")
    torch.save(model_info, f'model/{model_name}.pth')
    print("Finished!\n")

# train_and_evaluate(
#     classifier_type='weather',
#     model_name='rainy_sunny_classifier',
#     num_epochs=9
# )

train_and_evaluate(
    classifier_type='artstyle',
    model_name='artistic_realistic_classifier',
    num_epochs=9
)