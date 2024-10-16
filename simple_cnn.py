# simple_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # Eingabekanal 3 (RGB), Ausgabe 16 Filter
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Eingabe 16 Filter, Ausgabe 32 Filter
        
        # Dropout Schicht um overfitting vorzubeugen
        self.dropout = nn.Dropout(0.5) # 50% der Neuronen werden zufällig deaktiviert

        # Dummy input durchlaufen lassen, um die Größe der Ausgabedaten nach den Conv-Schichten zu berechnen
        self._calculate_conv_output()

        # Fully connected layers (aktualisierte Größe nach Convolution)
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc2 = nn.Linear(128, 2)  # Zwei Ausgabeklassen: regnerisch oder sonnig

    def _calculate_conv_output(self):
        # Wir nehmen eine Eingabebildgröße von 128x128 an (dies wird in den Transformationen gemacht)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)  # Beispielbild mit (Batch=1, Channels=3, Height=128, Width=128)
            x = F.relu(self.conv1(dummy_input))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            self.conv_output_size = x.numel()  # Anzahl der Elemente nach den Convolution Layers
            #print(f"Size of output after conv layers: {self.conv_output_size}")

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)  # Flachmachen der Tensoren für das Fully Connected Layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x