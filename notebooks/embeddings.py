import os
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

# Путь к папке с YAML-файлами
DATA_FOLDER = "/Users/timskeip/Downloads/data/ILCD/processes_yaml"

# Папка для сохранения текстов и эмбеддингов
OUTPUT_FOLDER = "/Users/timskeip/Downloads/data/ILCD/embedding"

# Создаём папку, если её нет
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Загрузка модели для создания эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Функция для загрузки YAML-файла
def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Функция для подготовки текста для RAG
def prepare_text_for_rag(data):
    # Собираем текстовую информацию из YAML
    text_parts = [
        f"Name: {data.get('name', '')}",
        f"Year: {data.get('year', '')}",
        f"Description: {data.get('description', '')}",
        f"Geography: {data.get('geography', '')}",
        "Classes: " + ", ".join(data.get('classes', []) if isinstance(data.get('classes'), list) else [])
    ]
    return " ".join(text_parts)

# Основной процесс
def main():
    # Обрабатываем каждый YAML-файл
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.yaml'):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"Processing file: {filename}")

            # Загрузка данных из YAML
            data = load_yaml(file_path)

            # Подготовка текста для RAG
            text = prepare_text_for_rag(data)

            # Создание эмбеддинга
            embedding = model.encode(text)

            # Формируем пути сохранения
            base_filename = os.path.splitext(filename)[0]  # Убираем .yaml
            text_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.txt")
            embedding_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.npy")

            # Сохраняем текст
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Сохраняем эмбеддинг
            np.save(embedding_file_path, embedding)

            print(f"Saved: {text_file_path} and {embedding_file_path}")

if __name__ == "__main__":
    main()
