# import os
# import yaml
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# # Папки с данными
# DATA_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/processes_yaml_2"
# OUTPUT_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/embedding_hugface"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#
# # Загрузка модели эмбеддингов
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Используем новую модель
#
# # Функция загрузки YAML-файла
# def load_yaml(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return yaml.safe_load(file) or {}
#
# # Функция для обработки списков, строк и словарей в YAML
# def process_yaml_value(value):
#     """Преобразует любые типы данных в читаемый текст"""
#     if isinstance(value, list):
#         return " | ".join(process_yaml_value(v) for v in value)
#     elif isinstance(value, dict):
#         return " | ".join(f"{k}: {process_yaml_value(v)}" for k, v in value.items())
#     elif isinstance(value, str):
#         return value.strip()
#     return ""
#
# # Функция подготовки текста
# def prepare_text_for_rag(data):
#     text_parts = []
#
#     # Основные поля
#     text_parts.append(f"Name: {data.get('Name', 'Unknown')}")
#     text_parts.append(f"Year: {process_yaml_value(data.get('Year', []))}")
#     text_parts.append(f"Geography: {data.get('Geography', 'Unknown')}")
#     text_parts.append(f"Class: {process_yaml_value(data.get('Class', []))}")
#     text_parts.append(f"Technology: {process_yaml_value(data.get('Technology', []))}")
#     text_parts.append(f"Main Output: {process_yaml_value(data.get('Main Output', []))}")
#
#     return " ".join(text_parts)
#
# # Основной процесс
# def main():
#     for filename in os.listdir(DATA_FOLDER):
#         if filename.endswith('.yaml'):
#             file_path = os.path.join(DATA_FOLDER, filename)
#             print(f"Processing file: {filename}")
#
#             # Загружаем YAML
#             data = load_yaml(file_path)
#             if not data:
#                 print(f"⚠️ Warning: Empty or invalid YAML in {filename}")
#                 continue
#
#             # Подготовка текста
#             text = prepare_text_for_rag(data)
#             if not text.strip():
#                 print(f"⚠️ Warning: Empty text generated for {filename}")
#                 continue
#
#             # Создание эмбеддинга
#             embedding = model.encode(text).astype('float32')
#
#             # Проверка размерности эмбеддинга
#             if embedding.shape[0] != 768:  # Размерность для paraphrase-multilingual-mpnet-base-v2
#                 print(f"⚠️ Warning: Unexpected embedding dimension for {filename}: {embedding.shape[0]}")
#                 continue
#
#             # Пути сохранения
#             base_filename = os.path.splitext(filename)[0]
#             text_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.txt")
#             embedding_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.npy")
#
#             # Сохраняем текст
#             with open(text_file_path, "w", encoding="utf-8") as f:
#                 f.write(text)
#
#             # Сохраняем эмбеддинг
#             np.save(embedding_file_path, embedding)
#
#             print(f"✅ Saved: {text_file_path} and {embedding_file_path}")
#
# if __name__ == "__main__":
#     main()





import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

# Загружаем модель для создания эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Подключаемся к Qdrant локально
client = QdrantClient(":memory:")  # Используем in-memory Qdrant для локального запуска

# Название коллекции
COLLECTION_NAME = "processes"

# Проверяем, существует ли коллекция, и создаём её, если нет
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,  # Размерность эмбеддингов для paraphrase-multilingual-mpnet-base-v2
            distance=Distance.COSINE
        )
    )
    print("✅ Коллекция создана!")
else:
    print("⚡ Коллекция уже существует, пропускаем создание.")

# Папка с файлами
DATA_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/embedding_hugface"  # Укажите путь к вашей папке

# Список для точек Qdrant
points = []
texts = []  # Список для хранения текстов для BM25

# Обрабатываем файлы
for i, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith('.txt'):
        # Загружаем текст
        txt_path = os.path.join(DATA_FOLDER, filename)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        texts.append(text)  # Добавляем текст в список для BM25

        # Загружаем соответствующий эмбеддинг
        npy_path = os.path.join(DATA_FOLDER, filename.replace('.txt', '.npy'))
        if not os.path.exists(npy_path):
            print(f"⚠️ Эмбеддинг для файла {filename} не найден.")
            continue

        embedding = np.load(npy_path).tolist()  # Преобразуем в список

        # Создаём точку для Qdrant
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "filename": filename,
                "text": text,
            }
        )
        points.append(point)

# Загружаем данные в Qdrant
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Загружено {len(points)} документов в Qdrant!")
else:
    print("⚠️ Нет данных для загрузки в Qdrant.")

# Создаём BM25 индекс
tokenized_texts = [text.split() for text in texts]  # Токенизация текстов
bm25 = BM25Okapi(tokenized_texts)  # Создаём BM25 индекс

# Функция поиска с использованием Qdrant и BM25
def search_qdrant_and_bm25(query, top_k=10):
    # Поиск с использованием Qdrant (эмбеддинги)
    query_embedding = model.encode(query).tolist()
    qdrant_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    print("\n🔍 **Результаты поиска Qdrant:**")
    for i, result in enumerate(qdrant_results, 1):
        print(f"\n[{i}] 📄 **{result.payload['filename']}** (Score: {result.score:.4f})")
        print(f"📝 {result.payload['text'][:300]}...")
        print("-" * 50)

    # Поиск с использованием BM25
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]  # Индексы топ-K документов

    print("\n🔍 **Результаты поиска BM25:**")
    for i, idx in enumerate(top_bm25_indices, 1):
        print(f"\n[{i}] 📄 **{points[idx].payload['filename']}** (Score: {bm25_scores[idx]:.4f})")
        print(f"📝 {points[idx].payload['text'][:300]}...")
        print("-" * 50)


# Гибридный поиск
def hybrid_search(query, alpha=0.5, top_k=10):
    query_embedding = model.encode(query).tolist()

    # 1. Qdrant поиск
    qdrant_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    qdrant_scores = {res.payload['filename']: res.score for res in qdrant_results}

    # 2. BM25 поиск
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]

    bm25_scores_dict = {points[idx].payload['filename']: bm25_scores[idx] for idx in top_bm25_indices}

    # 3. Нормализация
    scaler = MinMaxScaler()
    all_scores = np.array(list(qdrant_scores.values()) + list(bm25_scores_dict.values())).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(all_scores).flatten()

    # Разделяем обратно
    norm_qdrant_scores = {k: normalized_scores[i] for i, k in enumerate(qdrant_scores.keys())}
    norm_bm25_scores = {k: normalized_scores[i + len(qdrant_scores)] for i, k in enumerate(bm25_scores_dict.keys())}

    # 4. Объединение результатов
    final_scores = {}
    for filename in set(norm_qdrant_scores.keys()).union(norm_bm25_scores.keys()):
        final_scores[filename] = alpha * norm_qdrant_scores.get(filename, 0) + (1 - alpha) * norm_bm25_scores.get(
            filename, 0)

    # 5. Сортировка
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # 6. Вывод результатов
    print("\n🔍 **Гибридный поиск:**")
    for i, (filename, score) in enumerate(sorted_results, 1):
        text = next(p.payload['text'] for p in points if p.payload['filename'] == filename)
        print(f"\n[{i}] 📄 **{filename}** (Score: {score:.4f})")
        print(f"📝 {text[:300]}...")
        print("-" * 50)




# Запускаем поиск
query = "Brown coal mining"
search_qdrant_and_bm25(query)
hybrid_search(query, alpha=0.7, top_k=5)
