import os
from sahi.slicing import slice_image
from PIL import Image
import numpy as np

# Путь к основной папке с данными
data_dir = 'data/images'

# Путь к папке с оригинальными изображениями
train_images_dir = os.path.join(data_dir, 'original_test_data')

# Путь к папке для сохранения тайлов
images_tile_dir = os.path.join(data_dir, 'test')
def create_directory_if_not_exist(directory):
# Создание папки для сохранения тайлов, если она не существует
    os.makedirs(directory, exist_ok=True)



# Функция для проверки существования хотя бы одного тайла
def check_tiles_exist(image_name, output_image_dir):
    # Проверка наличия хотя бы одного файла, начинающегося с имени картинки и содержащего "_tile_"
    for filename in os.listdir(output_image_dir):
        if filename.startswith(image_name) and '_tile_' in filename:
            return True
    return False


# Функция для создания тайлов изображения
def create_tiles_from_image(image_path, output_image_dir):
    try:
        # Загрузка изображения
        image = Image.open(image_path)

        # Разделение изображения на тайлы
        sliced_images = slice_image(np.array(image), slice_height=256, slice_width=256, overlap_height_ratio=0.2,
                                    overlap_width_ratio=0.2)

        # Сохранение тайлов изображения в папку 'test'
        for i, tile in enumerate(sliced_images):
            tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_tile_{i}.jpg"  # Новое имя для тайла
            tile_path = os.path.join(output_image_dir, tile_name)

            # Преобразование массива numpy в изображение и сохранение
            tile_image = Image.fromarray(tile['image'])
            tile_image.save(tile_path)

            print(f"Saved image tile: {tile_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Функция для обработки изображений
def process_images_and_tiles(train_images_dir, images_tile_dir):
    if not os.path.exists(train_images_dir):
        print(f"Directory not found: {train_images_dir}")
        return

    # Получение списка изображений
    image_files = [f for f in os.listdir(train_images_dir) if
                   f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp')]

    # Проход по каждому изображению
    for image_file in image_files:
        image_path = os.path.join(train_images_dir, image_file)
        image_name = os.path.splitext(image_file)[0]

        # Проверка на существование хотя бы одного тайла
        if check_tiles_exist(image_name, images_tile_dir):
            print(f"Skipping image {image_file}, tiles already exist.")
            continue

        # Создание тайлов изображения и сохранение их в папку 'test'
        create_tiles_from_image(image_path, images_tile_dir)


# Вызов функции для обработки изображений
create_directory_if_not_exist(images_tile_dir)
process_images_and_tiles(train_images_dir, images_tile_dir)