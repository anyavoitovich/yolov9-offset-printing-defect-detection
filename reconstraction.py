import os
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
import re

# Размеры тайлов и перекрытие
tile_size = 256
overlap = 0.2

# Путь к папке с тайлами с дефектами (detected tiles) и папке с лэйблами
base_detect_dir = 'runs/detect'
output_dir = 'runs/restored'


def get_latest_detect_folder(base_dir):
    """Находит последнюю папку с детекцией."""
    folders = [folder for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    pattern = r'yolov9_m_c__detect(\d+)'
    folder_dict = {}

    for folder in folders:
        match = re.match(pattern, folder)
        if match:
            folder_number = int(match.group(1))  # Извлекаем цифру
            folder_dict[folder_number] = folder

    if folder_dict:
        latest_folder = folder_dict[max(folder_dict.keys())]
        return os.path.join(base_dir, latest_folder)
    else:
        raise FileNotFoundError("No detect folders found matching the pattern 'yolov9_m_c__detectX'.")


def check_directories(detect_dir):
    """Проверяет существование папки с тайлами."""
    if not os.path.exists(detect_dir):
        raise FileNotFoundError(f"Directory with detected tiles does not exist: {detect_dir}")


def merge_detected_tiles(tiles_dict, original_width, original_height):
    """Воссоздает изображение с найденными дефектами из тайлов."""
    reconstructed_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    tile_height_with_overlap = int(tile_size * (1 - overlap))
    tile_width_with_overlap = int(tile_size * (1 - overlap))

    for tile_index, tile_path in tiles_dict.items():
        if not os.path.exists(tile_path):
            print(f"Warning: Tile does not exist: {tile_path}")
            continue

        try:
            tile_image = np.array(Image.open(tile_path))
        except Exception as e:
            print(f"Error loading tile {tile_path}: {e}")
            continue

        row = tile_index // (original_width // tile_width_with_overlap)
        col = tile_index % (original_width // tile_width_with_overlap)
        y = row * tile_height_with_overlap
        x = col * tile_width_with_overlap

        actual_tile_height = min(tile_image.shape[0], original_height - y)
        actual_tile_width = min(tile_image.shape[1], original_width - x)

        reconstructed_image[y:y + actual_tile_height, x:x + actual_tile_width, :] = tile_image[:actual_tile_height,
                                                                                    :actual_tile_width, :]

    return Image.fromarray(reconstructed_image)


def draw_defects(image, label_file_path, original_width, original_height):
    """Рисует рамки вокруг дефектов на изображении."""
    draw = ImageDraw.Draw(image)

    if os.path.exists(label_file_path):
        with open(label_file_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if len(parts) != 6:
                    print(f"Warning: Invalid label format in file {label_file_path}: {line.strip()}")
                    continue

                cls, x_center, y_center, width, height, confidence = parts

                x_center_pixel = int(x_center * original_width)
                y_center_pixel = int(y_center * original_height)
                width_pixel = int(width * original_width)
                height_pixel = int(height * original_height)

                x1 = x_center_pixel - width_pixel // 2
                y1 = y_center_pixel - height_pixel // 2
                x2 = x_center_pixel + width_pixel // 2
                y2 = y_center_pixel + height_pixel // 2

                if x1 < 0 or y1 < 0 or x2 > original_width or y2 > original_height:
                    print(f"Warning: Invalid bounding box for {label_file_path}: ({x1}, {y1}), ({x2}, {y2})")
                    continue

                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                draw.text((x1, y1), f'{confidence:.2f}', fill='red')


def group_tiles_by_image(detect_dir):
    """Группирует тайлы по изображениям."""
    tiles_grouped_by_image = defaultdict(dict)
    tile_files = [f for f in os.listdir(detect_dir) if f.endswith('.jpg') or f.endswith('.tif')]

    for tile_file in tile_files:
        tile_path = os.path.join(detect_dir, tile_file)
        base_name = os.path.splitext(tile_file)[0]
        image_name, tile_index = base_name.rsplit('_tile_', 1)
        tile_index = int(tile_index)
        tiles_grouped_by_image[image_name][tile_index] = tile_path

    return tiles_grouped_by_image


def process_images(original_width, original_height):
    """Обрабатывает все изображения и воссоздает их с дефектами."""
    detect_dir = get_latest_detect_folder(base_detect_dir)
    check_directories(detect_dir)

    label_dir = os.path.join(detect_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)

    tiles_grouped_by_image = group_tiles_by_image(detect_dir)

    for image_name, tiles_dict in tiles_grouped_by_image.items():
        output_image_path = os.path.join(output_dir, f"{image_name}_reconstructed.jpg")
        reconstructed_image = merge_detected_tiles(tiles_dict, original_width, original_height)
        label_file_path = os.path.join(label_dir, f"{image_name}.txt")
        draw_defects(reconstructed_image, label_file_path, original_width, original_height)
        reconstructed_image.save(output_image_path)
        print(f"Reconstructed image with defects saved at: {output_image_path}")


# Пример использования
original_width = 1456  # Задайте ширину исходного изображения
original_height = 1088  # Задайте высоту исходного изображения
process_images(original_width, original_height)