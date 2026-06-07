import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# CẤU HÌNH
# =========================

TOP_K = 90
TOP_SUPPORT_DOCS = 3

# Trọng số chấm điểm
W_COSINE = 0.55
W_QUESTION_OVERLAP = 0.15
W_CHOICE_COVERAGE = 0.20
W_SUBSTRING = 0.10

# Kết hợp passage tốt nhất với trung bình top passage
W_BEST_DOC = 0.85
W_TOP_DOC_MEAN = 0.15

# Fallback về đáp án mặc định
DEFAULT_ANSWER = "B"

# Nếu tất cả đáp án đều có điểm thấp hơn mức này thì chọn mặc định.
# Tăng giá trị này nếu muốn fallback về B nhiều hơn.
MIN_EVIDENCE_SCORE = 0.24

# Nếu đáp án tốt nhất và đáp án thứ hai quá sát nhau thì chọn mặc định.
# Tăng giá trị này nếu muốn fallback về B nhiều hơn.
FALLBACK_MARGIN = 0.018

# Nếu chưa đủ mơ hồ để fallback nhưng vẫn khá sát điểm,
# ưu tiên coverage và substring để tie-break.
TIE_THRESHOLD = 0.04

MIN_CORPUS_DOCS = 50000
EXPECTED_QUESTIONS = 765

# Mở rộng thêm một số mẫu câu phủ định thường gặp
NEGATION_RE = re.compile(
    r"không thuộc"
    r"|không bao gồm"
    r"|không đúng"
    r"|không chính xác"
    r"|không phù hợp"
    r"|không được"
    r"|không phải"
    r"|ngoại trừ"
    r"|sai là"
    r"|nhận định sai",
    re.IGNORECASE,
)


def is_negation_question(question_norm):
    return bool(NEGATION_RE.search(question_norm))


def normalize_text(text):
    """Chuẩn hóa khoảng trắng và chuyển về chữ thường."""
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def tokenize(text):
    """Tokenizer đơn giản, nhanh và đủ dùng cho TF-IDF."""
    return re.findall(r"\w+", text, flags=re.UNICODE)


def retrieve_top_k_tfidf(retrieval_text, vectorizer, doc_vectors, top_k):
    """
    Truy xuất top-k passage bằng cosine TF-IDF.

    TfidfVectorizer mặc định đã chuẩn hóa L2.
    Vì vậy, tích vô hướng tương đương cosine similarity nhưng nhanh hơn
    việc gọi cosine_similarity riêng.
    """
    query_vector = vectorizer.transform([retrieval_text])
    similarities = (doc_vectors @ query_vector.T).toarray().ravel()

    n = len(similarities)
    k = min(top_k, n)

    if k == n:
        return np.argsort(similarities)[::-1]

    indices = np.argpartition(similarities, -k)[-k:]
    return indices[np.argsort(similarities[indices])][::-1]


def aggregate_top_scores(scores, top_n=3):
    """
    Tránh phụ thuộc hoàn toàn vào đúng một passage.
    Kết hợp passage tốt nhất và trung bình top-n passage.
    """
    if len(scores) == 0:
        return 0.0

    n = min(top_n, len(scores))
    if n == len(scores):
        top_scores = scores
    else:
        top_scores = np.partition(scores, -n)[-n:]

    best_score = float(np.max(top_scores))
    mean_score = float(np.mean(top_scores))

    return W_BEST_DOC * best_score + W_TOP_DOC_MEAN * mean_score


def pick_answer(
    choice_scores,
    choice_coverage,
    choice_substring,
    pick_lowest,
):
    """
    Chọn đáp án cuối cùng.

    - Câu bình thường: chọn điểm cao.
    - Câu phủ định: chọn điểm thấp.
    - Nếu bằng chứng yếu hoặc hai đáp án quá sát nhau: fallback về B.
    """
    if not choice_scores:
        return DEFAULT_ANSWER

    scores = list(choice_scores.values())

    # Không passage nào chứa đủ bằng chứng đáng tin cậy
    if max(scores) < MIN_EVIDENCE_SCORE:
        return DEFAULT_ANSWER

    ranked = sorted(
        choice_scores.items(),
        key=lambda x: x[1],
        reverse=not pick_lowest,
    )

    best_choice, best_score = ranked[0]

    if len(ranked) == 1:
        return best_choice

    second_choice, second_score = ranked[1]
    margin = abs(best_score - second_score)

    # Hai đáp án quá sát nhau: không đoán liều
    if margin < FALLBACK_MARGIN:
        return DEFAULT_ANSWER

    # Điểm tương đối sát nhau: dùng coverage và substring để tie-break
    if margin < TIE_THRESHOLD:
        candidates = [best_choice, second_choice]

        def tie_rank(choice):
            coverage = choice_coverage.get(choice, 0.0)
            substring = choice_substring.get(choice, 0.0)

            if pick_lowest:
                return coverage, substring

            return -coverage, -substring

        return min(candidates, key=tie_rank)

    return best_choice


def load_corpus(corpus_file="/content/dataset.json"):
    """Đọc dataset.json dạng JSON Lines."""
    documents = []

    try:
        with open(corpus_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                title = normalize_text(doc.get("title", ""))
                demuc = normalize_text(doc.get("demuc_name", ""))
                chude = normalize_text(doc.get("chude_name", ""))
                content = normalize_text(doc.get("content", ""))

                # Giữ title hai lần để tăng nhẹ trọng số tiêu đề
                text = f"{title} {title} {demuc} {chude} {content}".strip()

                if text:
                    documents.append(text)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {corpus_file}.")

    return documents


def make_submission(
    test_file="/content/de_thi.json",
    corpus_file="/content/dataset.json",
    output_file="submission.json",
):
    # =========================
    # 1. ĐỌC CORPUS
    # =========================

    documents = load_corpus(corpus_file)

    if not documents:
        print("Lỗi: Corpus rỗng hoặc không đọc được dữ liệu.")
        return

    if len(documents) < MIN_CORPUS_DOCS:
        print(
            f"Cảnh báo: chỉ tải được {len(documents)} điều luật "
            f"(dự kiến khoảng 70.000). Có thể bạn đang dùng sai file dataset."
        )
        return

    print(f"Đã tải {len(documents)} tài liệu từ corpus.")

    # =========================
    # 2. XÂY TF-IDF
    # =========================

    print("Đang xây dựng TF-IDF...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        dtype=np.float32,
    )

    doc_vectors = vectorizer.fit_transform(documents)

    # Tạo trước word set để không tokenize lặp lại trong từng câu hỏi
    doc_word_sets = [set(tokenize(doc)) for doc in documents]

    print("Đã xây dựng xong TF-IDF.")

    # =========================
    # 3. ĐỌC BỘ ĐỀ
    # =========================

    try:
        with open(test_file, "r", encoding="utf-8") as file:
            test_data = json.load(file)

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {test_file}.")
        return

    if len(test_data) != EXPECTED_QUESTIONS:
        print(
            f"Lỗi: de_thi.json có {len(test_data)} câu, "
            f"nhưng cần đúng {EXPECTED_QUESTIONS} câu."
        )
        return

    submissions = []
    valid_choices = ["A", "B", "C", "D"]
    total = len(test_data)

    print("Bắt đầu truy xuất và dự đoán đáp án...")

    # =========================
    # 4. XỬ LÝ TỪNG CÂU HỎI
    # =========================

    for idx, item in enumerate(test_data, start=1):
        question_id = item.get("id")
        question_norm = normalize_text(item.get("question", ""))

        choices = {
            key: normalize_text(item.get(key, ""))
            for key in valid_choices
        }

        # ---------------------------------
        # 4.1. RETRIEVAL
        # ---------------------------------
        #
        # Lặp câu hỏi hai lần để câu hỏi vẫn có trọng số chính.
        # Thêm cả bốn đáp án để truy xuất đúng passage hơn.
        #
        retrieval_text = " ".join(
            [
                question_norm,
                question_norm,
                choices["A"],
                choices["B"],
                choices["C"],
                choices["D"],
            ]
        ).strip()

        top_doc_indices = retrieve_top_k_tfidf(
            retrieval_text,
            vectorizer,
            doc_vectors,
            TOP_K,
        )

        top_doc_vectors = doc_vectors[top_doc_indices]

        top_pool = [
            (doc_word_sets[index], documents[index])
            for index in top_doc_indices
        ]

        # ---------------------------------
        # 4.2. TẠO BATCH BỐN GIẢ THUYẾT
        # ---------------------------------
        #
        # Lặp lại choice hai lần để tăng trọng số phần khác biệt
        # giữa các đáp án.
        #
        hypotheses = [
            f"{question_norm} {choices[key]} {choices[key]}".strip()
            for key in valid_choices
        ]

        hypothesis_vectors = vectorizer.transform(hypotheses)

        # Shape: (4, TOP_K)
        cosine_matrix = (
            top_doc_vectors @ hypothesis_vectors.T
        ).toarray().T

        question_words = set(tokenize(question_norm))
        pick_lowest = is_negation_question(question_norm)

        choice_scores = {}
        choice_coverage = {}
        choice_substring = {}

        # ---------------------------------
        # 4.3. CHẤM ĐIỂM BỐN ĐÁP ÁN
        # ---------------------------------

        for choice_index, choice_key in enumerate(valid_choices):
            choice_text = choices[choice_key]

            if not choice_text:
                continue

            hypothesis = hypotheses[choice_index]
            hypo_words = set(tokenize(hypothesis))
            choice_words = set(tokenize(choice_text))

            passage_scores = []
            max_coverage = 0.0
            max_substring = 0.0

            for pool_index, (doc_words, doc_text) in enumerate(top_pool):
                cosine = float(cosine_matrix[choice_index, pool_index])

                question_overlap = (
                    len(question_words & doc_words) / len(question_words)
                    if question_words
                    else 0.0
                )

                choice_coverage_value = (
                    len(choice_words & doc_words) / len(choice_words)
                    if choice_words
                    else 0.0
                )

                substring = (
                    1.0
                    if choice_text and choice_text in doc_text
                    else 0.0
                )

                score = (
                    W_COSINE * cosine
                    + W_QUESTION_OVERLAP * question_overlap
                    + W_CHOICE_COVERAGE * choice_coverage_value
                    + W_SUBSTRING * substring
                )

                passage_scores.append(score)

                max_coverage = max(max_coverage, choice_coverage_value)
                max_substring = max(max_substring, substring)

            choice_scores[choice_key] = aggregate_top_scores(
                np.asarray(passage_scores, dtype=np.float32),
                TOP_SUPPORT_DOCS,
            )

            choice_coverage[choice_key] = max_coverage
            choice_substring[choice_key] = max_substring

        # ---------------------------------
        # 4.4. CHỌN ĐÁP ÁN HOẶC FALLBACK
        # ---------------------------------

        best_choice = pick_answer(
            choice_scores,
            choice_coverage,
            choice_substring,
            pick_lowest,
        )

        submissions.append(
            {
                "id": question_id,
                "answer": best_choice,
            }
        )

        if idx % 20 == 0 or idx == total:
            print(f"Đã xử lý {idx}/{total} câu.")

    # =========================
    # 5. GHI KẾT QUẢ JSON
    # =========================

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            submissions,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Đã xử lý xong {len(submissions)} câu.")
    print(f"File kết quả: {output_file}")

    if len(submissions) != EXPECTED_QUESTIONS:
        print(
            f"Lỗi: submission chỉ có {len(submissions)} câu, "
            f"cần {EXPECTED_QUESTIONS} câu."
        )
    else:
        print("OK: submission.json có đúng 765 câu.")


if __name__ == "__main__":
    make_submission()
