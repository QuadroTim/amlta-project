# EMBEDDING CREATION
import os
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

# Папки с данными
DATA_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/processes_yaml_2"
OUTPUT_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/embedding_hugface"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Загрузка модели эмбеддингов
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # Используем новую модель

# Функция загрузки YAML-файла
def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file) or {}

# Функция для обработки списков, строк и словарей в YAML
def process_yaml_value(value):
    """Преобразует любые типы данных в читаемый текст"""
    if isinstance(value, list):
        return " | ".join(process_yaml_value(v) for v in value)
    elif isinstance(value, dict):
        return " | ".join(f"{k}: {process_yaml_value(v)}" for k, v in value.items())
    elif isinstance(value, str):
        return value.strip()
    return ""

# Функция подготовки текста
def prepare_text_for_rag(data):
    text_parts = []

    # Основные поля
    text_parts.append(f"Name: {data.get('Name', 'Unknown')}")
    text_parts.append(f"Year: {process_yaml_value(data.get('Year', []))}")
    text_parts.append(f"Geography: {data.get('Geography', 'Unknown')}")
    text_parts.append(f"Class: {process_yaml_value(data.get('Class', []))}")
    text_parts.append(f"Technology: {process_yaml_value(data.get('Technology', []))}")
    text_parts.append(f"Main Output: {process_yaml_value(data.get('Main Output', []))}")

    return " ".join(text_parts)

# Основной процесс
def main():
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith('.yaml'):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"Processing file: {filename}")

            # Загружаем YAML
            data = load_yaml(file_path)
            if not data:
                print(f"⚠️ Warning: Empty or invalid YAML in {filename}")
                continue

            # Подготовка текста
            text = prepare_text_for_rag(data)
            if not text.strip():
                print(f"⚠️ Warning: Empty text generated for {filename}")
                continue

            # Создание эмбеддинга
            embedding = model.encode(text).astype('float32')

            # Проверка размерности эмбеддинга
            if embedding.shape[0] != 768:  # Размерность для paraphrase-multilingual-mpnet-base-v2
                print(f"⚠️ Warning: Unexpected embedding dimension for {filename}: {embedding.shape[0]}")
                continue

            # Пути сохранения
            base_filename = os.path.splitext(filename)[0]
            text_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.txt")
            embedding_file_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}.npy")

            # Сохраняем текст
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Сохраняем эмбеддинг
            np.save(embedding_file_path, embedding)

            print(f"✅ Saved: {text_file_path} and {embedding_file_path}")

if __name__ == "__main__":
    main()




# MODEL
import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

client = QdrantClient(":memory:")

COLLECTION_NAME = "processes"

if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )

# Папка с файлами
DATA_FOLDER = "/Users/timskeip/Downloads/data-2/ILCD/embedding_hugface"  # Укажите путь к вашей папке

points = []
texts = []

for i, filename in enumerate(os.listdir(DATA_FOLDER)):
    if filename.endswith('.txt'):
        txt_path = os.path.join(DATA_FOLDER, filename)
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        texts.append(text)

        # Загружаем соответствующий эмбеддинг
        npy_path = os.path.join(DATA_FOLDER, filename.replace('.txt', '.npy'))
        if not os.path.exists(npy_path):
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


def search_qdrant(query, top_k=10):
    query_embedding = model.encode(query).tolist()
    qdrant_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    print("\n🔍 **Results Qdrant:**")
    for i, result in enumerate(qdrant_results, 1):
        print(f"\n[{i}] 📄 **{result.payload['filename']}** (Score: {result.score:.4f})")
        print(f"📝 {result.payload['text'][:300]}...")
        print("-" * 50)

query = "frack-low (UBA) gas pipeline in Germany in 2010"
search_qdrant(query)





#EVALUATION
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score

# Пример тестовых данных с более точными запросами
test_data = [
    {
        "query": "frack-low (UBA) gas pipeline in Germany in 2010",
        "relevant_docs": ["0a1e4c77-9c20-42a2-8ba8-1ef32954f3dd.txt"]
    },
    {
        "query": "Gas-fired combined cycle power plant in Germany in 2010",
        "relevant_docs": ["0a2507db-7819-4de6-8970-c7242e85a430.txt"]
    },
    {
        "query": "Light oil refinery in Bosnia and Herzegovina, 2015",
        "relevant_docs": ["00aaa4aa-62d5-4b04-b0a7-cc0f6bd8d6b2.txt"]
    },
    {
        "query": "Syncrude oil boiler in Canada, 2005",
        "relevant_docs": ["0aaf8d5f-cc1c-481a-b88a-79f91a901046.txt"]
    },
    {
        "query": "Light oil central heating system in the Netherlands, 2005",
        "relevant_docs": ["0abc1ac7-fe4d-43d5-93ff-cece894d0a11.txt"]
    },
    {
        "query": "Light oil gas turbine power plant in Germany, 2000",
        "relevant_docs": ["0ac9bc16-7577-4524-aee9-441ccf1b74ae.txt"]
    },
    {
        "query": "Cement production in Germany, 2050, including clinker grinding, using 960 kg of clinker and 40 kg of gypsum per ton of Portland cement, electricity consumption 108 kWh/t, without accounting for dust emissions.",
        "relevant_docs": ["0b0d6f50-2c46-4392-a8fd-6187253bb698.txt"]
    },
    {
        "query": "Electricity transmission and distribution in Indonesia, 2005",
        "relevant_docs": ["25be9d6a-e67a-48d0-a88c-a92b93557166.txt"]
    },
    {
        "query": "Membrane electrolysis sodium hydroxide production in Germany",
        "relevant_docs": ["b7c07aef-3fc2-4f80-a954-d243790f8cdc.txt"]
    },
    {
        "query": "elementary flows from German oil refinery in 2005",
        "relevant_docs": ["8018e62b-f261-4ab5-a9a7-bbcb5aca078c.txt"]
    },
    {
        "query": "large-scale hydropower electricity generation in Australia 2000",
        "relevant_docs": ["06553e03-be4f-4142-996c-3cbcce968322.txt"]
    },
    {
        "query": "large-scale hydropower electricity generation in Australia 2000",
        "relevant_docs": ["06553e03-be4f-4142-996c-3cbcce968322.txt"]
    },
    {
        "query": "electric arc steel production in Germany 2050",
        "relevant_docs": ["bc775c74-de9f-41ad-8cc2-fc9d4d764724.txt"]
    },
    {
        "query": "straw-fired heating plant emissions 2030",
        "relevant_docs": ["aed59c08-b710-407c-936d-f862cf0ace2c.txt"]
    },
    {
        "query": "PtG heating technology for households in Germany 2050",
        "relevant_docs": ["9cc29d68-2e89-4c60-8033-3524e8f4c86c.txt"]
    },
    {
        "query": "hot-rolled steel production data 2005",
        "relevant_docs": ["3c92c885-f378-401d-acff-33b50327bbe3.txt"]
    },
]

# EVALUATION WITHOUT RERANKING
def evaluate_model(test_data, top_k=10):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    hit_rate_scores = []
    reciprocal_ranks = []

    for test_case in test_data:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        query_embedding = model.encode(f"query: {query}").tolist()
        qdrant_results = client.query_points(COLLECTION_NAME, query_embedding, limit=top_k)

        found_docs = [res.payload['filename'] for res in qdrant_results.points]

        y_true = [1 if doc in relevant_docs else 0 for doc in found_docs]
        y_pred = [1] * len(y_true)

        # Вычисляем метрики
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        ndcg = ndcg_score([y_true], [y_true], k=top_k)

        hit_rate = 1 if any(doc in relevant_docs for doc in found_docs) else 0

        reciprocal_rank = 0
        for rank, doc in enumerate(found_docs, start=1):
            if doc in relevant_docs:
                reciprocal_rank = 1 / rank
                break

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ndcg_scores.append(ndcg)
        hit_rate_scores.append(hit_rate)
        reciprocal_ranks.append(reciprocal_rank)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcg_scores)
    avg_hit_rate = np.mean(hit_rate_scores)
    avg_mrr = np.mean(reciprocal_ranks)

    print(f"Precision@{top_k}: {avg_precision:.4f}")
    print(f"Recall@{top_k}: {avg_recall:.4f}")
    print(f"F1-Score@{top_k}: {avg_f1:.4f}")
    print(f"NDCG@{top_k}: {avg_ndcg:.4f}")
    print(f"Hit Rate@{top_k}: {avg_hit_rate:.4f}")
    print(f"MRR@{top_k}: {avg_mrr:.4f}")

evaluate_model(test_data, top_k=10)


# RERANKING EVALUATION
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

def evaluate_model_with_reranker(test_data, top_k=10, rerank_limit=50):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    hit_rate_scores = []
    reciprocal_ranks = []

    for test_case in test_data:
        query = test_case["query"]
        relevant_docs = set(test_case["relevant_docs"])

        query_embedding = model.encode(f"query: {query}").tolist()
        qdrant_results = client.query_points(COLLECTION_NAME, query_embedding, limit=rerank_limit)

        found_docs = [res.payload['filename'] for res in qdrant_results.points]
        found_texts = [res.payload['text'] for res in qdrant_results.points]

        pairs = [(query, doc_text) for doc_text in found_texts]
        reranker_scores = reranker.predict(pairs)

        ranked_indices = np.argsort(reranker_scores)[::-1]
        ranked_docs = [found_docs[i] for i in ranked_indices]

        ranked_docs_top_k = ranked_docs[:top_k]

        y_true = [1 if doc in relevant_docs else 0 for doc in ranked_docs_top_k]
        y_pred = [1] * len(y_true)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        ndcg = ndcg_score([y_true], [y_true], k=top_k)

        hit_rate = 1 if any(doc in relevant_docs for doc in ranked_docs_top_k) else 0

        reciprocal_rank = 0
        for rank, doc in enumerate(ranked_docs_top_k, start=1):
            if doc in relevant_docs:
                reciprocal_rank = 1 / rank
                break

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ndcg_scores.append(ndcg)
        hit_rate_scores.append(hit_rate)
        reciprocal_ranks.append(reciprocal_rank)


    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcg_scores)
    avg_hit_rate = np.mean(hit_rate_scores)
    avg_mrr = np.mean(reciprocal_ranks)

    print(f"Precision@{top_k}: {avg_precision:.4f}")
    print(f"Recall@{top_k}: {avg_recall:.4f}")
    print(f"F1-Score@{top_k}: {avg_f1:.4f}")
    print(f"NDCG@{top_k}: {avg_ndcg:.4f}")
    print(f"Hit Rate@{top_k}: {avg_hit_rate:.4f}")
    print(f"MRR@{top_k}: {avg_mrr:.4f}")
    print("paraprhrase-multilingual-mpnet-base-v2")

evaluate_model_with_reranker(test_data, top_k=10, rerank_limit=50)
