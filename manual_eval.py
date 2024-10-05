import torch
from PIL import Image
import torchvision.transforms as transforms

from simple_cnn import SimpleCNN

# Modell und Labels laden
def load_model_with_labels(model_path):
    model_info = torch.load(model_path)
    
    model = SimpleCNN()
    model.load_state_dict(model_info['model_state_dict'])

    class_labels = model_info['class_labels']  # Lade die gespeicherten Labels
    
    return model, class_labels

def evaluate_single_image(model, image_path, class_labels):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Batch-Dimension hinzufügen

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = class_labels[predicted.item()]
    print(f"Vorhersage: {predicted_class}\n")

def test_evaluate(model_name, image_path):
    print(f"testing {model_name}")
    model, class_labels = load_model_with_labels(f'model/{model_name}.pth')
    print(f"Geladene Labels: {class_labels}")
    print(f"Testing with image: {image_path}")
    evaluate_single_image(model, image_path, class_labels)



# Vorhersage für ein einzelnes Bild - weather
test_evaluate("rainy_sunny_classifier", "../dataset/test/artistic_sunny/image_08.png")

test_evaluate("rainy_sunny_classifier", "../dataset/test/artistic_rainy/image_08.png")

test_evaluate("rainy_sunny_classifier", "../dataset/test/realistic_sunny/image_41.png")

test_evaluate("rainy_sunny_classifier", "../dataset/test/realistic_rainy/image_41.png")

# Vorhersage für ein einzelnes Bild - artstyle
test_evaluate("artistic_realistic_classifier", "../dataset/test/realistic_sunny/image_08.png")

test_evaluate("artistic_realistic_classifier", "../dataset/test/realistic_rainy/image_08.png")

test_evaluate("artistic_realistic_classifier", "../dataset/test/artistic_sunny/image_41.png")

test_evaluate("artistic_realistic_classifier", "../dataset/test/artistic_rainy/image_41.png")
