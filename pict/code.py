import cv2
import os
import albumentations as A

# Папки
input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Трансформации (случайные при каждом вызове)
transform = A.Compose([
    A.RandomCrop(width=224, height=224, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
])

# Обработка
image_counter = 0
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Генерируем 25 аугментаций
    for i in range(25):
        augmented = transform(image=image)["image"]
        aug_path = os.path.join(output_folder, f"img_{image_counter:03d}.jpg")
        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        image_counter += 1

print(f"Создано {image_counter} изображений.")