import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. TEXT PREPROCESSING
# ============================================================

def normalize_text(text):
    """
    Chuẩn hóa văn bản:
    - chuyển về chữ thường
    - chuẩn hóa Unicode
    - loại bỏ ký tự thừa
    - giữ lại chữ cái, chữ số và dấu tiếng Việt
    """
    text = str(text or "")
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Tokenize đơn giản, phù hợp với phương pháp cổ điển."""
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def split_sentences(text):
    """
    Chia văn bản thành câu hoặc đoạn ngắn.
    Có xử lý cả xuống dòng và dấu câu.
    """
    text = str(text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Nếu một câu quá dài, tiếp tục chia nhỏ để tránh passage bị loãng.
    result = []
    for sentence in sentences:
        if len(sentence) <= 700:
            result.append(sentence)
        else:
            for start in range(0, len(sentence), 600):
                chunk = sentence[start:start + 700].strip()
                if chunk:
                    result.append(chunk)

    return result


# ============================================================
# 2. LOAD CORPUS AND CREATE PASSAGES
# ============================================================

def read_json_or_jsonl(file_path):
    """
    Đọc được cả:
    - JSON list: [{...}, {...}]
    - JSONL: mỗi dòng là một object JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return []

    try:
        data = json.loads(raw_text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["data", "documents", "items", "questions"]:
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]

    except json.JSONDecodeError:
        pass

    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return rows


def load_passages(corpus_file="dataset.json", window_size=3, stride=2):
    """
    Chia mỗi document thành passage gồm một cửa sổ 3 câu.

    Title được lặp lại hai lần để tăng trọng số nhẹ cho tiêu đề.
    Đây là một cách đơn giản để mô phỏng field weighting.
    """
    documents = read_json_or_jsonl(corpus_file)
    passages = []

    for doc_index, doc in enumerate(documents):
        title = str(doc.get("title", "")).strip()
        content = str(doc.get("content", "")).strip()

        sentences = split_sentences(content)

        if not sentences:
            sentences = [content or title]

        for start in range(0, len(sentences), stride):
            selected_sentences = sentences[start:start + window_size]

            if not selected_sentences:
                continue

            body = " ".join(selected_sentences).strip()

            if title:
                passage_text = f"{title}. {title}. {body}"
            else:
                passage_text = body

            if normalize_text(passage_text):
                passages.append({
                    "doc_index": doc_index,
                    "title": title,
                    "text": passage_text
                })

    return passages


# ============================================================
# 3. BM25 USING AN INVERTED INDEX
# ============================================================

class BM25InvertedIndex:
    """
    Okapi BM25 sử dụng inverted index.

    Với mỗi term, chỉ lưu danh sách document có chứa term đó.
    Khi truy vấn, không cần duyệt toàn bộ token trong toàn corpus.
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.tokenized_docs = [tokenize(text) for text in texts]
        self.num_docs = len(self.tokenized_docs)

        self.doc_lengths = np.array(
            [len(tokens) for tokens in self.tokenized_docs],
            dtype=np.float32
        )

        self.avg_doc_length = float(self.doc_lengths.mean()) \
            if self.num_docs > 0 else 0.0

        self.inverted_index = defaultdict(list)
        self.idf = {}

        self._build_index()

    def _build_index(self):
        document_frequency = Counter()

        for doc_id, tokens in enumerate(self.tokenized_docs):
            term_frequency = Counter(tokens)

            for term, frequency in term_frequency.items():
                self.inverted_index[term].append((doc_id, frequency))
                document_frequency[term] += 1

        for term, frequency in document_frequency.items():
            # Công thức BM25 IDF ổn định và luôn dương.
            self.idf[term] = math.log(
                1.0
                + (self.num_docs - frequency + 0.5)
                / (frequency + 0.5)
            )

    def score(self, query):
        query_tokens = tokenize(query)
        scores = np.zeros(self.num_docs, dtype=np.float32)

        if not query_tokens or self.avg_doc_length == 0:
            return scores

        query_term_frequency = Counter(query_tokens)

        for term, query_frequency in query_term_frequency.items():
            if term not in self.inverted_index:
                continue

            idf = self.idf.get(term, 0.0)

            for doc_id, term_frequency in self.inverted_index[term]:
                doc_length = self.doc_lengths[doc_id]

                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b * doc_length / self.avg_doc_length
                    )
                )

                scores[doc_id] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )

        return scores


# ============================================================
# 4. HYBRID RETRIEVER
# ============================================================

def min_max_normalize(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return values

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum - minimum < 1e-12:
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


class HybridRetriever:
    """
    Kết hợp:
    - BM25
    - TF-IDF word unigram + bigram
    - TF-IDF character n-gram

    Word n-gram bắt từ và cụm từ.
    Character n-gram hỗ trợ lỗi chính tả và biến thể cách viết.
    """

    def __init__(self, passages):
        self.passages = passages
        self.texts = [passage["text"] for passage in passages]

        if not self.texts:
            raise ValueError("Corpus không có passage hợp lệ.")

        print("Đang xây dựng BM25 inverted index...")
        self.bm25 = BM25InvertedIndex(self.texts)

        print("Đang xây dựng TF-IDF word n-gram...")
        self.word_vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            tokenizer=None,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            max_features=250_000
        )

        self.word_matrix = self.word_vectorizer.fit_transform(self.texts)

        print("Đang xây dựng TF-IDF character n-gram...")
        self.char_vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            max_features=180_000
        )

        self.char_matrix = self.char_vectorizer.fit_transform(self.texts)

    def get_raw_scores(self, query):
        """Tính toàn bộ điểm retrieval trước khi trộn."""
        bm25_scores = self.bm25.score(query)

        word_query_vector = self.word_vectorizer.transform([query])
        word_scores = (self.word_matrix @ word_query_vector.T).toarray().ravel()

        char_query_vector = self.char_vectorizer.transform([query])
        char_scores = (self.char_matrix @ char_query_vector.T).toarray().ravel()

        return {
            "bm25": bm25_scores,
            "word": word_scores,
            "char": char_scores
        }

    def search(self, query, top_k=20):
        """
        Hybrid retrieval.
        Chuẩn hóa từng nhóm điểm trước khi kết hợp.
        """
        scores = self.get_raw_scores(query)

        hybrid_scores = (
            0.50 * min_max_normalize(scores["bm25"])
            + 0.35 * min_max_normalize(scores["word"])
            + 0.15 * min_max_normalize(scores["char"])
        )

        top_k = min(top_k, len(self.texts))

        if top_k <= 0:
            return [], scores, hybrid_scores

        if top_k == len(self.texts):
            indices = np.argsort(-hybrid_scores)
        else:
            indices = np.argpartition(-hybrid_scores, top_k - 1)[:top_k]
            indices = indices[np.argsort(-hybrid_scores[indices])]

        return indices.tolist(), scores, hybrid_scores


# ============================================================
# 5. ANSWER SCORING FEATURES
# ============================================================

NEGATIVE_PATTERNS = [
    r"\bkhông đúng\b",
    r"\bkhông chính xác\b",
    r"\bkhông phải\b",
    r"\bkhông thuộc\b",
    r"\bkhông bao gồm\b",
    r"\bngoại trừ\b",
    r"\bphát biểu sai\b",
    r"\bnhận định sai\b",
    r"\bsai là\b",
    r"\bchưa đúng\b"
]


def is_negative_question(question):
    normalized_question = normalize_text(question)

    return any(
        re.search(pattern, normalized_question)
        for pattern in NEGATIVE_PATTERNS
    )


def top_mean(values, top_n=3):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0

    selected = np.sort(values)[-top_n:]
    return float(selected.mean())


def get_choice_similarity(retriever, choice, passage_indices):
    """
    Tính cosine similarity giữa đáp án và các passage đã lấy ra.
    """
    if not passage_indices or not normalize_text(choice):
        return 0.0, 0.0

    word_vector = retriever.word_vectorizer.transform([choice])
    word_scores = (
        retriever.word_matrix[passage_indices] @ word_vector.T
    ).toarray().ravel()

    char_vector = retriever.char_vectorizer.transform([choice])
    char_scores = (
        retriever.char_matrix[passage_indices] @ char_vector.T
    ).toarray().ravel()

    return top_mean(word_scores), top_mean(char_scores)


def calculate_idf_overlap(retriever, choice, passage_indices):
    """
    Tỷ lệ token của đáp án được passage hỗ trợ.
    Token hiếm có trọng số cao hơn token phổ biến.
    """
    choice_tokens = set(tokenize(choice))

    if not choice_tokens or not passage_indices:
        return 0.0

    denominator = sum(
        retriever.bm25.idf.get(token, 0.1)
        for token in choice_tokens
    )

    if denominator <= 0:
        return 0.0

    best_score = 0.0

    for passage_index in passage_indices[:8]:
        passage_tokens = set(
            retriever.bm25.tokenized_docs[passage_index]
        )

        numerator = sum(
            retriever.bm25.idf.get(token, 0.1)
            for token in choice_tokens
            if token in passage_tokens
        )

        best_score = max(best_score, numerator / denominator)

    return float(best_score)


def calculate_exact_phrase_score(retriever, choice, passage_indices):
    """
    Kiểm tra đáp án có xuất hiện nguyên cụm trong passage hay không.
    Chỉ dùng khi đáp án có ít nhất hai token.
    """
    normalized_choice = normalize_text(choice)

    if len(tokenize(normalized_choice)) < 2:
        return 0.0

    for passage_index in passage_indices[:8]:
        normalized_passage = normalize_text(
            retriever.texts[passage_index]
        )

        if normalized_choice in normalized_passage:
            return 1.0

    return 0.0


def fuzzy_similarity(text_a, text_b):
    """
    Fuzzy matching bằng SequenceMatcher trong thư viện chuẩn.
    So sánh với từng câu để tránh passage dài làm giảm điểm.
    """
    text_a = normalize_text(text_a)
    text_b = normalize_text(text_b)

    if not text_a or not text_b:
        return 0.0

    candidates = split_sentences(text_b)

    if not candidates:
        candidates = [text_b]

    scores = [
        SequenceMatcher(
            None,
            text_a,
            normalize_text(candidate)
        ).ratio()
        for candidate in candidates
    ]

    return max(scores, default=0.0)


def calculate_fuzzy_score(retriever, choice, passage_indices):
    """
    Điểm fuzzy tốt nhất trong các passage liên quan nhất.
    """
    if not normalize_text(choice):
        return 0.0

    return max(
        (
            fuzzy_similarity(
                choice,
                retriever.texts[passage_index]
            )
            for passage_index in passage_indices[:5]
        ),
        default=0.0
    )


def extract_choice_features(
    retriever,
    question,
    choice,
    question_passage_indices,
    answer_top_k=15
):
    """
    Chấm một đáp án bằng nhiều tín hiệu độc lập.
    """
    answer_aware_query = f"{question} {choice}"

    answer_indices, raw_scores, _ = retriever.search(
        answer_aware_query,
        top_k=answer_top_k
    )

    # Mức độ passage phù hợp với cả câu hỏi và đáp án.
    word_retrieval_score = top_mean(
        raw_scores["word"][answer_indices]
    )

    char_retrieval_score = top_mean(
        raw_scores["char"][answer_indices]
    )

    unique_query_token_count = max(
        len(set(tokenize(answer_aware_query))),
        1
    )

    normalized_bm25_score = math.log1p(
        float(np.max(raw_scores["bm25"][answer_indices]))
        / unique_query_token_count
    ) if answer_indices else 0.0

    # Mức độ đáp án được hỗ trợ bởi passage lấy theo question + choice.
    local_word_support, local_char_support = get_choice_similarity(
        retriever,
        choice,
        answer_indices
    )

    # Mức độ đáp án được hỗ trợ bởi passage lấy từ câu hỏi gốc.
    global_word_support, global_char_support = get_choice_similarity(
        retriever,
        choice,
        question_passage_indices
    )

    overlap_score = calculate_idf_overlap(
        retriever,
        choice,
        answer_indices
    )

    exact_phrase_score = calculate_exact_phrase_score(
        retriever,
        choice,
        answer_indices
    )

    fuzzy_score = calculate_fuzzy_score(
        retriever,
        choice,
        answer_indices
    )

    return {
        "word_retrieval": word_retrieval_score,
        "char_retrieval": char_retrieval_score,
        "bm25_retrieval": normalized_bm25_score,
        "local_word_support": local_word_support,
        "local_char_support": local_char_support,
        "global_word_support": global_word_support,
        "global_char_support": global_char_support,
        "idf_overlap": overlap_score,
        "exact_phrase": exact_phrase_score,
        "fuzzy": fuzzy_score
    }


# ============================================================
# 6. RERANK FOUR ANSWERS
# ============================================================

FEATURE_WEIGHTS = {
    "word_retrieval": 0.12,
    "char_retrieval": 0.05,
    "bm25_retrieval": 0.12,
    "local_word_support": 0.15,
    "local_char_support": 0.06,
    "global_word_support": 0.13,
    "global_char_support": 0.05,
    "idf_overlap": 0.16,
    "exact_phrase": 0.10,
    "fuzzy": 0.06
}


def normalize_features_between_choices(choice_features):
    """
    Chuẩn hóa từng feature giữa A/B/C/D.

    Ví dụ:
    - A có BM25 cao nhất sẽ gần 1
    - D có BM25 thấp nhất sẽ gần 0

    Nhờ đó các feature có thang điểm khác nhau vẫn kết hợp được.
    """
    normalized = {
        key: {}
        for key in choice_features
    }

    feature_names = next(iter(choice_features.values())).keys()

    for feature_name in feature_names:
        values = np.array(
            [
                choice_features[key][feature_name]
                for key in choice_features
            ],
            dtype=np.float32
        )

        scaled_values = min_max_normalize(values)

        for index, choice_key in enumerate(choice_features):
            normalized[choice_key][feature_name] = float(
                scaled_values[index]
            )

    return normalized


def predict_answer(retriever, item, question_top_k=25):
    question = str(item.get("question", ""))
    valid_choices = ["A", "B", "C", "D"]

    question_indices, _, _ = retriever.search(
        question,
        top_k=question_top_k
    )

    raw_choice_features = {}

    for choice_key in valid_choices:
        choice_text = str(item.get(choice_key, ""))

        raw_choice_features[choice_key] = extract_choice_features(
            retriever=retriever,
            question=question,
            choice=choice_text,
            question_passage_indices=question_indices
        )

    normalized_choice_features = normalize_features_between_choices(
        raw_choice_features
    )

    final_scores = {}

    for choice_key in valid_choices:
        final_scores[choice_key] = sum(
            FEATURE_WEIGHTS[feature_name]
            * normalized_choice_features[choice_key][feature_name]
            for feature_name in FEATURE_WEIGHTS
        )

    negative_question = is_negative_question(question)

    if negative_question:
        predicted_answer = min(final_scores, key=final_scores.get)
    else:
        predicted_answer = max(final_scores, key=final_scores.get)

    debug_info = {
        "negative_question": negative_question,
        "scores": final_scores,
        "raw_features": raw_choice_features,
        "normalized_features": normalized_choice_features
    }

    return predicted_answer, debug_info


# ============================================================
# 7. CREATE SUBMISSION
# ============================================================

def make_submission(
    test_file="/content/de_thi.json",
    corpus_file="/content/dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
    debug_file="debug_predictions.json"
):
    print("Đang đọc corpus và chia passage...")
    passages = load_passages(
        corpus_file=corpus_file,
        window_size=3,
        stride=2
    )

    if not passages:
        print("Lỗi: corpus không có passage hợp lệ.")
        return

    print(f"Đã tạo {len(passages)} passage.")

    retriever = HybridRetriever(passages)

    test_data = read_json_or_jsonl(test_file)

    if not test_data:
        print("Lỗi: không đọc được dữ liệu đề thi.")
        return

    submissions = []
    debug_predictions = []

    print("Bắt đầu truy xuất và dự đoán đáp án...")

    for index, item in enumerate(test_data, start=1):
        question_id = item.get("id")

        answer, debug_info = predict_answer(
            retriever=retriever,
            item=item
        )

        submissions.append({
            "id": question_id,
            "answer": answer
        })

        debug_predictions.append({
            "id": question_id,
            "question": item.get("question", ""),
            "prediction": answer,
            **debug_info
        })

        if index % 20 == 0 or index == len(test_data):
            print(f"Đã xử lý {index}/{len(test_data)} câu.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            submissions,
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(
            debug_predictions,
            f,
            ensure_ascii=False,
            indent=2
        )

    with zipfile.ZipFile(
        zip_file,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zipf:
        zipf.write(output_file)

    print(f"Đã xử lý xong {len(submissions)} câu hỏi.")
    print(f"File đáp án: {output_file}")
    print(f"File debug: {debug_file}")
    print(f"File nộp bài: {zip_file}")


if __name__ == "__main__":
    make_submission()
