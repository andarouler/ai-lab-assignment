import os
import shutil

def merge_classes(source_dirs, dest_dir, class_name):
    os.makedirs(dest_dir, exist_ok=True)
    for source_dir in source_dirs:
        for img in os.listdir(source_dir):
            shutil.copy(os.path.join(source_dir, img), os.path.join(dest_dir, img))

# Zusammenführen der "rainy" Bilder
rainy_dirs = ['../dataset/train/artistic_rainy', '../dataset/train/realistic_rainy']
merge_classes(rainy_dirs, '../dataset/train/rainy', 'rainy')

# Zusammenführen der "sunny" Bilder
sunny_dirs = ['../dataset/train/artistic_sunny', '../dataset/train/realistic_sunny']
merge_classes(sunny_dirs, '../dataset/train/sunny', 'sunny')

# Dasselbe für den Testdatensatz
rainy_dirs_test = ['../dataset/test/artistic_rainy', '../dataset/test/realistic_rainy']
merge_classes(rainy_dirs_test, '../dataset/test/rainy', 'rainy')

sunny_dirs_test = ['../dataset/test/artistic_sunny', '../dataset/test/realistic_sunny']
merge_classes(sunny_dirs_test, '../dataset/test/sunny', 'sunny')
