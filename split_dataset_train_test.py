# split_dataset_train_test.py

import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, train_dir, test_dir, test_size=0.2):
    classes = ['artistic_rainy', 'artistic_sunny', 'realistic_rainy', 'realistic_sunny']
    
    for cls in classes:
        # Erstelle Unterordner für jede Klasse im Train- und Testordner
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
        
        # Pfad zu den Bilddateien
        class_dir = os.path.join(source_dir, cls)
        # Liste nur die Dateien auf, keine Verzeichnisse
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        
        # Aufteilen in Trainings- und Testdaten
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        
        # Kopiere die Dateien in die jeweiligen Ordner
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, cls, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, cls, img))

source_dir = '../AI_lab_generated'
train_dir = '../dataset/train'
test_dir = '../dataset/test'

split_data(source_dir, train_dir, test_dir)
