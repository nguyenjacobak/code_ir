"""
baseline_v2.py — TF-IDF retrieval + scoring (~61.7%).
Chạy: cd Final && python baseline_v2.py
Cần: dataset.json (~70k dòng), de_thi.json (765 câu) trong cùng thư mục Final/
"""
import json
import os
import re
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [V2] Thêm cấu hình — root không có
TOP_K = 90
W_COSINE = 0.50
W_OVERLAP = 0.20
W_COVERAGE = 0.20
W_SUBSTRING = 0.10
TIE_THRESHOLD = 0.01  # hai đáp án sát điểm hơn ngưỡng này thì tie-break

MIN_CORPUS_DOCS = 50000  # Final/dataset.json có ~70k điều — phát hiện chạy sai thư mục
EXPECTED_QUESTIONS = 765  # Final/de_thi.json — KHÔNG dùng de_thi.json 1000 câu ở thư mục gốc

# Câu hỏi dạng "không thuộc / không đúng" → chọn đáp án ít khớp luật nhất
NEGATION_RE = re.compile(
    r"không thuộc|không bao gồm|điểm không đúng|không đúng là gì",
    re.IGNORECASE,
)


def is_negation_question(question_norm):
    return bool(NEGATION_RE.search(question_norm))


def tokenize(text):
    return re.findall(r"\w+", text)


def retrieve_top_k_tfidf(question_norm, vectorizer, doc_vectors, top_k):
    """Retrieval bằng cosine TF-IDF — phù hợp bộ Final hơn BM25."""
    query_vector = vectorizer.transform([question_norm])
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    n = len(similarities)
    k = min(top_k, n)
    if k == n:
        return np.argsort(similarities)[-k:][::-1]
    idx = np.argpartition(similarities, -k)[-k:]
    return idx[np.argsort(similarities[idx])][::-1]


def pick_answer(choice_scores, choice_coverage, choice_substring, pick_lowest):
    """Chọn đáp án; nếu top-2 sát điểm thì ưu tiên coverage rồi substring."""
    if not choice_scores:
        return "B"  # mặc định nếu không tính được điểm nào (câu rất khó hoặc lỗi dữ liệu)

    ranked = sorted(choice_scores.items(), key=lambda x: x[1], reverse=not pick_lowest)
    best_choice, best_score = ranked[0]
    if len(ranked) == 1:
        return best_choice

    second_score = ranked[1][1]
    if abs(best_score - second_score) >= TIE_THRESHOLD:
        return best_choice

    candidates = [ranked[0][0], ranked[1][0]]

    def tie_rank(choice):
        cov = choice_coverage.get(choice, 0.0)
        sub = choice_substring.get(choice, 0.0)
        if pick_lowest:
            return (cov, sub)  # phủ định: coverage thấp hơn tốt hơn
        return (-cov, -sub)  # bình thường: coverage/substring cao hơn tốt hơn

    return min(candidates, key=tie_rank) if pick_lowest else max(candidates, key=tie_rank)


def load_corpus(corpus_file="dataset.json"):
    """Đọc dataset.json — mỗi dòng 1 điều luật (JSON Lines)."""
    documents = []
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # [V2] Sửa từ root: root chỉ ghép title + content
                title = str(doc.get("title", "")).lower().strip()
                demuc = str(doc.get("demuc_name", "")).lower().strip()
                chude = str(doc.get("chude_name", "")).lower().strip()
                content = str(doc.get("content", "")).lower().strip()
                text = f"{title} {title} {demuc} {chude} {content}".strip()

                if text:
                    documents.append(text)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {corpus_file}.")
    return documents


def make_submission(
    test_file="de_thi.json",
    corpus_file="dataset.json",
    output_file="submission.json",
    zip_file="submission.zip",
):
    # 1. Tải corpus (giống root)
    documents = load_corpus(corpus_file)
    if not documents:
        print("Lỗi: Corpus rỗng. Hãy chạy trong thư mục Final/ có dataset.json dạng JSON Lines.")
        return
    if len(documents) < MIN_CORPUS_DOCS:
        print(
            f"Cảnh báo: chỉ tải {len(documents)} điều luật "
            f"(cần ~70k). Có thể đang chạy sai thư mục hoặc sai file dataset."
        )
        return
    print(f"Đã tải {len(documents)} tài liệu từ tập corpus...")

    # 2. Xây TF-IDF
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    doc_vectors = vectorizer.fit_transform(documents)
    doc_word_sets = [set(tokenize(doc)) for doc in documents]

    # 3. Đọc đề thi (giống root)
    try:
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {test_file}.")
        return

    if len(test_data) != EXPECTED_QUESTIONS:
        print(
            f"Lỗi: de_thi.json có {len(test_data)} câu, cần {EXPECTED_QUESTIONS} câu.\n"
            "Bạn đang dùng nhầm bộ đề (thường là de_thi.json 1000 câu ở thư mục gốc).\n"
            "Hãy chạy: cd d:\\IR_Final\\Final && python baseline_v2.py"
        )
        return

    submissions = []
    valid_choices = ["A", "B", "C", "D"]
    total = len(test_data)

    print("Bắt đầu truy xuất và dự đoán đáp án...")
    # 4. Xử lý từng câu hỏi
    for idx, item in enumerate(test_data, start=1):
        question_id = item.get("id")
        question_text = item.get("question", "")

        # --- BƯỚC 4.1: RETRIEVAL (TF-IDF cosine) ---
        question_norm = re.sub(r"\s+", " ", str(question_text or "").lower().strip())
        top_doc_indices = retrieve_top_k_tfidf(
            question_norm, vectorizer, doc_vectors, TOP_K
        )
        top_doc_vectors = doc_vectors[top_doc_indices]
        top_pool = [(doc_word_sets[i], documents[i]) for i in top_doc_indices]

        # --- BƯỚC 4.2: CHỌN ĐÁP ÁN A/B/C/D ---
        pick_lowest = is_negation_question(question_norm)
        choice_scores = {}
        choice_coverage = {}
        choice_substring = {}

        for choice_key in valid_choices:
            choice_text = re.sub(
                r"\s+", " ", str(item.get(choice_key, "")).lower().strip()
            )

            hypothesis = f"{question_norm} {choice_text}".strip()
            if not hypothesis:
                continue

            hypo_vector = vectorizer.transform([hypothesis])
            hypo_words = set(tokenize(hypothesis))
            choice_words = set(tokenize(choice_text))
            cosine_scores = cosine_similarity(hypo_vector, top_doc_vectors).flatten()

            choice_best = -1.0
            max_coverage = 0.0
            max_substring = 0.0
            for i, (doc_words, doc_text) in enumerate(top_pool):
                cosine = float(cosine_scores[i])
                overlap = len(hypo_words & doc_words) / len(hypo_words) if hypo_words else 0.0
                coverage = len(choice_words & doc_words) / len(choice_words) if choice_words else 0.0
                substring = 1.0 if choice_text and choice_text in doc_text else 0.0

                score = (
                    W_COSINE * cosine
                    + W_OVERLAP * overlap
                    + W_COVERAGE * coverage
                    + W_SUBSTRING * substring
                )
                choice_best = max(choice_best, score)
                max_coverage = max(max_coverage, coverage)
                max_substring = max(max_substring, substring)

            if choice_best > -1.0:
                choice_scores[choice_key] = choice_best
                choice_coverage[choice_key] = max_coverage
                choice_substring[choice_key] = max_substring

        best_choice = pick_answer(
            choice_scores, choice_coverage, choice_substring, pick_lowest
        )

        submissions.append({"id": question_id, "answer": best_choice})

        if idx % 20 == 0 or idx == total:
            print(f"Đã xử lý {idx}/{total} câu")

    # 5. Ghi submission.json (giống root)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submissions, f, ensure_ascii=False, indent=2)
    print(f"File kết quả: {output_file}")

    # 6. Zip — [V2] thêm file .py vào zip để nộp bài
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_file, arcname="submission.json")
        zipf.write(__file__, arcname=os.path.basename(__file__))
    print(f"File nộp: {zip_file}")
    print(f"Đã xử lý xong {len(submissions)} câu.")
    if len(submissions) != EXPECTED_QUESTIONS:
        print(f"Lỗi: submission có {len(submissions)} câu, cần {EXPECTED_QUESTIONS}.")
    else:
        print("OK: submission.json đúng 765 câu — nộp file submission.zip trong thư mục Final/.")


if __name__ == "__main__":
    make_submission()
