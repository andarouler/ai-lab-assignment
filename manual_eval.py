import torch
from PIL import Image
import torchvision.transforms as transforms
import torch

from simple_cnn import SimpleCNN

# Modell initialisieren
model = SimpleCNN()
model.load_state_dict(torch.load('model/rainy_sunny_classifier.pth'))
model.eval()  # Setze das Modell in den Evaluierungsmodus


# Transformationen für das Bild (wie beim Training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Lade ein Bild
image_path = '../dataset/test/realistic_sunny/image_08.png'
image = Image.open(image_path)

# Wende die Transformationen an
image = transform(image)

# Bild ins richtige Format bringen (Batch-Dimension hinzufügen)
image = image.unsqueeze(0)  # Fügt eine Batch-Dimension hinzu: [1, Channels, Height, Width]

# Modell in den Evaluierungsmodus versetzen (falls noch nicht geschehen)
model.eval()

# Mache die Vorhersage
with torch.no_grad():  # Verhindert, dass Gradient-Informationen gesammelt werden (spart Speicher)
    output = model(image)
    print(f"output {output}")
    _, predicted = torch.max(output, 1)  # Nimmt die Klasse mit dem höchsten Wert
    print(f"predicted {predicted}")

# Die Vorhersage anzeigen
if predicted.item() == 0:
    print("Vorhersage: Rainy")
else:
    print("Vorhersage: Sunny")