# datasets.py

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class WeatherDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['rainy', 'sunny']
        
        # Lade die Bilder aus den entsprechenden Verzeichnissen und setze Labels
        for label, subdir in enumerate(['rainy', 'sunny']):  # 0 für rainy, 1 für sunny
            # Pfade für rainy und sunny Bilder (artistic & realistic)
            for art_style in ['artistic', 'realistic']:
                full_dir = os.path.join(root_dir, f"{art_style}_{subdir}")
                for img_file in os.listdir(full_dir):
                    if img_file.endswith(('.png')):  # Füge die passenden Bildformate hinzu
                        self.image_paths.append(os.path.join(full_dir, img_file))
                        self.labels.append(label)  # 0 für rainy, 1 für sunny
                        #print(f"Image-file: {img_file}, label: {label}, path: {full_dir}")

        print(f"Image count: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ArtStyleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = ['artistic', 'realistic']  
        
        # Lade die Bilder aus den entsprechenden Verzeichnissen und setze Labels
        for label, subdir in enumerate(['artistic', 'realistic']): # 0 für artistic, 1 für realistic
            # Pfade für artistic und realistic Bilder (rainy & sunny)
            for weather_type in ['rainy', 'sunny']:
                full_dir = os.path.join(root_dir, f"{subdir}_{weather_type}")
                for img_file in os.listdir(full_dir):
                    if img_file.endswith(('.png')):  # Füge die passenden Bildformate hinzu
                        self.image_paths.append(os.path.join(full_dir, img_file))
                        self.labels.append(label)  # 0 für artistic, 1 für realistic
                        #print(f"Image-file: {img_file}, label: {label}, path: {full_dir}")

        print(f"Image count: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label