import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Загружаем модель для создания эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Папка с файлами
DATA_FOLDER = "/Users/timskeip/Downloads/data/ILCD/processed"

# Создание индекса FAISS
dimension = 384  # Размерность эмбеддингов (MiniLM)
index = faiss.IndexFlatL2(dimension)  # Используем L2 расстояние

# Список для хранения текстов и метаданных
texts = []
metadata = []

# Обрабатываем файлы
for i, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith('.txt'):
        # Загружаем текст
        txt_path = os.path.join(DATA_FOLDER, filename)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Загружаем соответствующий эмбеддинг
        npy_path = os.path.join(DATA_FOLDER, filename.replace('.txt', '.npy'))
        if not os.path.exists(npy_path):
            print(f"⚠️ Эмбеддинг для файла {filename} не найден.")
            continue

        embedding = np.load(npy_path).astype('float32')  # Преобразуем в float32

        # Добавляем эмбеддинг в индекс FAISS
        index.add(np.array([embedding]))

        # Сохраняем текст и метаданные
        texts.append(text)
        metadata.append({
            "filename": filename,
            "id": i  # Уникальный ID для каждого документа
        })

print(f"✅ Загружено {len(texts)} документов в FAISS!")

# Функция поиска
def search_faiss(query, top_k=3):
    # Создаём эмбеддинг для запроса
    query_embedding = model.encode(query).astype('float32')

    # Поиск ближайших соседей
    distances, indices = index.search(np.array([query_embedding]), top_k)

    print("\n🔍 **Результаты поиска:**")
    for i, idx in enumerate(indices[0]):
        print(f"\n[{i + 1}] 📄 **{metadata[idx]['filename']}** (Расстояние: {distances[0][i]:.4f})")
        print(f"📝 {texts[idx][:300]}...")
        print("-" * 50)

# Запускаем поиск
query = "Gas DE?"
search_faiss(query)




# import os
# import numpy as np
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# import faiss
#
# # Подключаемся к Qdrant
# client = QdrantClient(
#     url="",
#     api_key="",
# )
#
# # Название коллекции
# COLLECTION_NAME = "processes"
#
# # Проверяем, существует ли коллекция, и создаём её, если нет
# collections = client.get_collections().collections
# if COLLECTION_NAME not in [c.name for c in collections]:
#     client.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(
#             size=384,  # Размерность эмбеддингов (MiniLM)
#             distance=Distance.COSINE  # Косинусное расстояние для поиска
#         )
#     )
#     print("✅ Коллекция создана!")
# else:
#     print("⚡ Коллекция уже существует, пропускаем создание.")
#
# # Папка с файлами
# DATA_FOLDER = ""
#
# # Список для точек Qdrant
# points = []
#
# # Обрабатываем файлы
# for i, filename in enumerate(os.listdir(DATA_FOLDER)):
#     if filename.endswith('.txt'):
#         # Загружаем текст
#         txt_path = os.path.join(DATA_FOLDER, filename)
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             text = f.read().strip()
#
#         # Загружаем соответствующий эмбеддинг
#         npy_path = os.path.join(DATA_FOLDER, filename.replace('.txt', '.npy'))
#         if not os.path.exists(npy_path):
#             print(f"⚠️ Эмбеддинг для файла {filename} не найден.")
#             continue
#
#         embedding = np.load(npy_path).tolist()  # Преобразуем в список
#
#         # Создаём точку для Qdrant
#         point = PointStruct(
#             id=i,
#             vector=embedding,
#             payload={
#                 "filename": filename,
#                 "text": text,
#             }
#         )
#         points.append(point)
#
# # Загружаем данные в Qdrant
# if points:
#     client.upsert(collection_name=COLLECTION_NAME, points=points)
#     print(f"✅ Загружено {len(points)} документов в Qdrant!")
# else:
#     print("⚠️ Нет данных для загрузки в Qdrant.")
#
# # Функция поиска
# def search_qdrant(query, top_k=3):
#     # Если у вас есть модель для создания эмбеддингов, используйте её
#     # query_embedding = model.encode(query).tolist()
#     # Или загрузите эмбеддинг запроса из файла, если он у вас есть
#
#     # Пример: предположим, что у вас есть эмбеддинг запроса
#     query_embedding = [...]  # Замените на ваш эмбеддинг запроса
#
#     search_results = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_embedding,
#         limit=top_k
#     )
#
#     print("\n🔍 **Результаты поиска:**")
#     for i, result in enumerate(search_results, 1):
#         print(f"\n[{i}] 📄 **{result.payload['filename']}** (Score: {result.score:.4f})")
#         print(f"📝 {result.payload['text'][:300]}...")
#         print("-" * 50)
#
# # Запускаем поиск
# query = "Gas DE?"
# search_qdrant(query)
