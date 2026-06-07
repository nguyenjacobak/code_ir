import heapq
import json
import math
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pyvi import ViTokenizer


# ============================================================
# CẤU HÌNH
# ============================================================

CORPUS_FILE = "/content/dataset.json"
TEST_FILE = "/content/de_thi.json"

OUTPUT_FILE = "submission.json"
ZIP_FILE = "submission.zip"

# Mỗi passage gồm 3 câu, cửa sổ trượt 2 câu.
PASSAGE_SIZE = 3
PASSAGE_STRIDE = 2

# Số passage lấy từ câu hỏi gốc.
QUESTION_TOP_K = 25

# Số passage lấy từ question + từng đáp án.
CHOICE_TOP_K = 12

# Tham số BM25.
BM25_K1 = 1.5
BM25_B = 0.75

# Tiêu đề quan trọng hơn content.
# Lặp title là cách đơn giản để tăng trọng số mà không cần BM25F phức tạp.
TITLE_REPEAT = 2


# ============================================================
# 1. CHUẨN HÓA VÀ TÁCH TỪ TIẾNG VIỆT
# ============================================================

def normalize_text(text):
    """
    Chuẩn hóa văn bản:
    - Unicode NFC
    - chữ thường
    - bỏ dấu câu không cần thiết
    - giữ dấu gạch dưới do PyVi sinh ra
    """
    text = str(text or "")
    text = unicodedata.normalize("NFC", text.lower())

    # Giữ lại chữ, số và underscore.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    """
    Tách từ tiếng Việt bằng PyVi.

    Ví dụ:
    "công nghệ thông tin"
    → ["công_nghệ", "thông_tin"]
    """
    text = normalize_text(text)

    if not text:
        return []

    segmented_text = ViTokenizer.tokenize(text)

    return segmented_text.split()


def split_sentences(text):
    """
    Chia content thành các câu ngắn.
    Tránh passage quá dài làm loãng kết quả retrieval.
    """
    text = str(text or "").strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    result = []

    for sentence in sentences:
        # Cắt tiếp nếu một câu quá dài.
        if len(sentence) <= 700:
            result.append(sentence)
        else:
            for start in range(0, len(sentence), 600):
                chunk = sentence[start:start + 700].strip()

                if chunk:
                    result.append(chunk)

    return result


# ============================================================
# 2. ĐỌC JSON HOẶC JSONL
# ============================================================

def read_json_or_jsonl(file_path):
    """
    Hỗ trợ cả hai định dạng:

    JSON list:
    [
        {"id": 1, ...},
        {"id": 2, ...}
    ]

    JSONL:
    {"id": 1, ...}
    {"id": 2, ...}
    """
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read().strip()

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


# ============================================================
# 3. CHIA CORPUS THÀNH PASSAGE
# ============================================================

def load_passages(corpus_file):
    """
    Chia document thành passage 3 câu.

    Mỗi passage giữ lại title.
    Title được lặp lại để tăng trọng số.
    """
    documents = read_json_or_jsonl(corpus_file)

    passages = []

    for doc_index, document in enumerate(documents):
        title = str(document.get("title", "")).strip()
        content = str(document.get("content", "")).strip()

        sentences = split_sentences(content)

        if not sentences:
            sentences = [content or title]

        for start in range(0, len(sentences), PASSAGE_STRIDE):
            selected_sentences = sentences[start:start + PASSAGE_SIZE]

            if not selected_sentences:
                continue

            body = " ".join(selected_sentences).strip()

            if title:
                weighted_title = " ".join([title] * TITLE_REPEAT)
                passage_text = f"{weighted_title} {body}"
            else:
                passage_text = body

            if normalize_text(passage_text):
                passages.append({
                    "doc_index": doc_index,
                    "title": title,
                    "text": passage_text,
                    "normalized_text": normalize_text(passage_text)
                })

    return passages


# ============================================================
# 4. BM25 INVERTED INDEX
# ============================================================

class BM25InvertedIndex:
    """
    BM25 sử dụng inverted index.

    Mỗi term lưu danh sách:
        (passage_id, term_frequency)

    Khi tìm kiếm, chỉ duyệt các passage chứa từ trong query.
    Không cần duyệt toàn bộ corpus cho từng câu hỏi.
    """

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.tokenized_documents = []
        self.document_lengths = []

        self.inverted_index = defaultdict(list)
        self.idf = {}

        self._build(texts)

    def _build(self, texts):
        document_frequency = Counter()

        print("Đang tách từ corpus và xây dựng inverted index...")

        for document_id, text in enumerate(texts):
            tokens = tokenize(text)

            self.tokenized_documents.append(tokens)
            self.document_lengths.append(len(tokens))

            term_frequency = Counter(tokens)

            for term, frequency in term_frequency.items():
                self.inverted_index[term].append((document_id, frequency))
                document_frequency[term] += 1

        self.num_documents = len(self.tokenized_documents)

        if self.num_documents == 0:
            self.average_document_length = 0.0
            return

        self.average_document_length = (
            sum(self.document_lengths) / self.num_documents
        )

        for term, frequency in document_frequency.items():
            self.idf[term] = math.log(
                1.0
                + (
                    self.num_documents
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )

    def search(self, query, top_k=20):
        """
        Trả về danh sách:
            [(passage_id, bm25_score), ...]

        Chỉ giữ top-k passage.
        """
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        query_term_frequency = Counter(query_tokens)
        scores = defaultdict(float)

        for term, query_frequency in query_term_frequency.items():
            postings = self.inverted_index.get(term)

            if not postings:
                continue

            idf = self.idf.get(term, 0.0)

            for document_id, term_frequency in postings:
                document_length = self.document_lengths[document_id]

                denominator = (
                    term_frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b
                        * document_length
                        / self.average_document_length
                    )
                )

                score = (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )

                scores[document_id] += score

        if not scores:
            return []

        return heapq.nlargest(
            top_k,
            scores.items(),
            key=lambda item: item[1]
        )

    def score_document(self, query_tokens, document_id):
        """
        Tính BM25 của query đối với một passage cụ thể.
        Chỉ dùng trong bước rerank top-k nên rất nhanh.
        """
        if not query_tokens:
            return 0.0

        document_tokens = self.tokenized_documents[document_id]
        document_term_frequency = Counter(document_tokens)

        document_length = self.document_lengths[document_id]
        score = 0.0

        for term, query_frequency in Counter(query_tokens).items():
            term_frequency = document_term_frequency.get(term, 0)

            if term_frequency == 0:
                continue

            idf = self.idf.get(term, 0.0)

            denominator = (
                term_frequency
                + self.k1
                * (
                    1.0
                    - self.b
                    + self.b
                    * document_length
                    / self.average_document_length
                )
            )

            score += (
                idf
                * term_frequency
                * (self.k1 + 1.0)
                / denominator
                * query_frequency
            )

        return score


# ============================================================
# 5. FEATURE CHỌN ĐÁP ÁN
# ============================================================

NEGATIVE_PATTERNS = [
    r"\bkhông đúng\b",
    r"\bkhông chính xác\b",
    r"\bkhông phải\b",
    r"\bkhông thuộc\b",
    r"\bkhông bao gồm\b",
    r"\bngoại trừ\b",
    r"\bsai là\b",
    r"\bphát biểu sai\b",
    r"\bnhận định sai\b",
    r"\bchưa đúng\b"
]


def is_negative_question(question):
    """
    Nhận biết câu hỏi yêu cầu tìm đáp án sai hoặc ngoại lệ.
    """
    normalized_question = normalize_text(question)

    return any(
        re.search(pattern, normalized_question)
        for pattern in NEGATIVE_PATTERNS
    )


def min_max_normalize(values):
    """
    Chuẩn hóa danh sách điểm A/B/C/D về khoảng [0, 1].
    """
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum - minimum < 1e-12:
        return [0.0] * len(values)

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def calculate_idf_overlap(index, choice_tokens, passage_ids):
    """
    Tính tỷ lệ token quan trọng của đáp án được corpus hỗ trợ.

    Từ hiếm có trọng số cao hơn từ phổ biến.
    """
    unique_choice_tokens = set(choice_tokens)

    if not unique_choice_tokens or not passage_ids:
        return 0.0

    denominator = sum(
        index.idf.get(token, 0.1)
        for token in unique_choice_tokens
    )

    if denominator <= 0:
        return 0.0

    best_score = 0.0

    for passage_id in passage_ids:
        passage_tokens = set(index.tokenized_documents[passage_id])

        numerator = sum(
            index.idf.get(token, 0.1)
            for token in unique_choice_tokens
            if token in passage_tokens
        )

        best_score = max(best_score, numerator / denominator)

    return best_score


def calculate_exact_phrase(choice, passages, passage_ids):
    """
    Kiểm tra đáp án có xuất hiện nguyên cụm trong passage không.

    Không dùng với đáp án chỉ có một token vì dễ gây nhiễu.
    """
    choice = normalize_text(choice)

    if len(tokenize(choice)) < 2:
        return 0.0

    for passage_id in passage_ids:
        passage_text = passages[passage_id]["normalized_text"]

        if choice in passage_text:
            return 1.0

    return 0.0


def calculate_choice_features(
    index,
    passages,
    question,
    choice,
    question_passage_ids
):
    """
    Tính feature cho một đáp án.

    Không dùng char n-gram.
    Không dùng fuzzy matching.
    Không quét toàn corpus ngoài inverted index.
    """
    choice_tokens = tokenize(choice)
    combined_query = f"{question} {choice}"

    # Truy xuất bổ sung bằng câu hỏi + đáp án.
    combined_results = index.search(
        combined_query,
        top_k=CHOICE_TOP_K
    )

    combined_passage_ids = [
        passage_id
        for passage_id, _ in combined_results
    ]

    # Gộp passage từ câu hỏi gốc và question + choice.
    candidate_passage_ids = list(
        dict.fromkeys(
            question_passage_ids
            + combined_passage_ids
        )
    )

    # Giữ số lượng ứng viên nhỏ để rerank nhanh.
    candidate_passage_ids = candidate_passage_ids[:40]

    if combined_results:
        best_combined_bm25 = combined_results[0][1]
        mean_top3_combined_bm25 = (
            sum(score for _, score in combined_results[:3])
            / min(3, len(combined_results))
        )
    else:
        best_combined_bm25 = 0.0
        mean_top3_combined_bm25 = 0.0

    # Đáp án riêng lẻ có được passage hỗ trợ hay không?
    choice_support_scores = [
        index.score_document(
            query_tokens=choice_tokens,
            document_id=passage_id
        )
        for passage_id in candidate_passage_ids
    ]

    best_choice_support = max(
        choice_support_scores,
        default=0.0
    )

    idf_overlap = calculate_idf_overlap(
        index=index,
        choice_tokens=choice_tokens,
        passage_ids=candidate_passage_ids[:15]
    )

    exact_phrase = calculate_exact_phrase(
        choice=choice,
        passages=passages,
        passage_ids=candidate_passage_ids[:15]
    )

    return {
        "best_combined_bm25": best_combined_bm25,
        "mean_top3_combined_bm25": mean_top3_combined_bm25,
        "best_choice_support": best_choice_support,
        "idf_overlap": idf_overlap,
        "exact_phrase": exact_phrase
    }


# ============================================================
# 6. RERANK A/B/C/D
# ============================================================

FEATURE_WEIGHTS = {
    "best_combined_bm25": 0.30,
    "mean_top3_combined_bm25": 0.20,
    "best_choice_support": 0.20,
    "idf_overlap": 0.20,
    "exact_phrase": 0.10
}


def predict_answer(index, passages, item):
    """
    Truy xuất passage và chọn đáp án A/B/C/D.
    """
    question = str(item.get("question", ""))
    choice_keys = ["A", "B", "C", "D"]

    # Bước 1: retrieval bằng câu hỏi gốc.
    question_results = index.search(
        query=question,
        top_k=QUESTION_TOP_K
    )

    question_passage_ids = [
        passage_id
        for passage_id, _ in question_results
    ]

    # Bước 2: tính feature riêng cho từng đáp án.
    all_features = {}

    for choice_key in choice_keys:
        choice_text = str(item.get(choice_key, ""))

        all_features[choice_key] = calculate_choice_features(
            index=index,
            passages=passages,
            question=question,
            choice=choice_text,
            question_passage_ids=question_passage_ids
        )

    # Bước 3: chuẩn hóa từng feature giữa bốn đáp án.
    normalized_features = {
        choice_key: {}
        for choice_key in choice_keys
    }

    for feature_name in FEATURE_WEIGHTS:
        feature_values = [
            all_features[choice_key][feature_name]
            for choice_key in choice_keys
        ]

        normalized_values = min_max_normalize(feature_values)

        for index_choice, choice_key in enumerate(choice_keys):
            normalized_features[choice_key][feature_name] = (
                normalized_values[index_choice]
            )

    # Bước 4: tính tổng điểm.
    final_scores = {}

    for choice_key in choice_keys:
        final_scores[choice_key] = sum(
            FEATURE_WEIGHTS[feature_name]
            * normalized_features[choice_key][feature_name]
            for feature_name in FEATURE_WEIGHTS
        )

    # Bước 5: nếu câu hỏi phủ định, chọn đáp án ít được hỗ trợ nhất.
    if is_negative_question(question):
        predicted_answer = min(
            final_scores,
            key=final_scores.get
        )
    else:
        predicted_answer = max(
            final_scores,
            key=final_scores.get
        )

    debug_info = {
        "negative_question": is_negative_question(question),
        "scores": final_scores,
        "features": all_features
    }

    return predicted_answer, debug_info


# ============================================================
# 7. SINH SUBMISSION
# ============================================================

def create_submission():
    print("Đang đọc corpus...")
    passages = load_passages(CORPUS_FILE)

    if not passages:
        raise ValueError("Corpus không có passage hợp lệ.")

    print(f"Đã tạo {len(passages)} passage.")

    index = BM25InvertedIndex(
        texts=[
            passage["text"]
            for passage in passages
        ],
        k1=BM25_K1,
        b=BM25_B
    )

    print("Đang đọc câu hỏi...")
    test_data = read_json_or_jsonl(TEST_FILE)

    if not test_data:
        raise ValueError("Không đọc được dữ liệu đề thi.")

    submissions = []
    debug_predictions = []

    print(f"Bắt đầu dự đoán {len(test_data)} câu hỏi...")

    for position, item in enumerate(test_data, start=1):
        question_id = item.get("id")

        predicted_answer, debug_info = predict_answer(
            index=index,
            passages=passages,
            item=item
        )

        submissions.append({
            "id": question_id,
            "answer": predicted_answer
        })

        debug_predictions.append({
            "id": question_id,
            "answer": predicted_answer,
            **debug_info
        })

        if position % 25 == 0 or position == len(test_data):
            print(
                f"Đã xử lý {position}/{len(test_data)} câu."
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            submissions,
            file,
            ensure_ascii=False,
            indent=2
        )

    # File debug giúp kiểm tra nếu kết quả chưa tốt.
    with open("debug_predictions.json", "w", encoding="utf-8") as file:
        json.dump(
            debug_predictions,
            file,
            ensure_ascii=False,
            indent=2
        )

    # Zip cả code và file đáp án để tránh hệ thống chấm 0 điểm.
    with zipfile.ZipFile(
        ZIP_FILE,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:
        zip_file.write(OUTPUT_FILE)

        current_code_file = Path(sys.argv[0])

        if current_code_file.exists() and current_code_file.suffix == ".py":
            zip_file.write(
                current_code_file,
                arcname="/content/IR_cuoiki.py"
            )

    print()
    print("Đã hoàn thành.")
    print(f"File đáp án: {OUTPUT_FILE}")
    print(f"File debug: debug_predictions.json")
    print(f"File nộp bài: {ZIP_FILE}")


if __name__ == "__main__":
    create_submission()
