# ============================================================
# CÀI ĐẶT THƯ VIỆN
# ============================================================
!pip -q install transformers sentencepiece accelerate safetensors

# ============================================================
# IMPORT
# ============================================================
import json
import re
import time
import zipfile
import unicodedata
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. CẤU HÌNH
# ============================================================
CORPUS_FILE = "/content/dataset.json"
TEST_FILE = "/content/de_thi.json"

OUTPUT_FILE = "/content/submission.json"
ZIP_FILE = "/content/submission.zip"
DEBUG_FILE = "/content/submission_debug.json"

MODEL_NAME = "google/flan-t5-large"

VALID_CHOICES = ["A", "B", "C", "D"]

# Số khoản pháp luật được đưa vào prompt.
# Có thể thử TOP_K = 3, 5 hoặc 7.
TOP_K = 5

# FLAN-T5-large khá nặng.
# Mỗi câu hỏi được nhân thành 4 mẫu để chấm A, B, C, D.
# BATCH_SIZE = 2 tương ứng batch thực tế là 8.
BATCH_SIZE = 2

# Giới hạn độ dài đầu vào để inference nhanh và tránh tràn VRAM.
MAX_INPUT_TOKENS = 512

# In tiến độ sau mỗi số câu này.
PRINT_EVERY = 20


# ============================================================
# 2. CHUẨN HÓA VĂN BẢN
# ============================================================
def normalize_text(text):
    """
    Chuẩn hóa nhẹ:
    - lowercase
    - chuẩn hóa Unicode
    - đưa ký tự đặc biệt về khoảng trắng
    - chuẩn hóa số có số 0 ở đầu: 03 -> 3

    Không loại bỏ dấu tiếng Việt.
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # 03 tháng -> 3 tháng
    text = re.sub(r"\b0+(\d+)\b", r"\1", text)

    # Chỉ giữ chữ, số và khoảng trắng.
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)

    return " ".join(tokens)


# ============================================================
# 3. ĐỌC FILE JSON HOẶC JSONL
# ============================================================
def load_json_or_jsonl(file_path):
    """
    Hỗ trợ:
    - JSON list
    - JSON object
    - JSON object có key "data" hoặc "questions"
    - JSONL: mỗi dòng là một JSON object
    """
    with open(file_path, "r", encoding="utf-8") as file:
        raw_content = file.read().strip()

    if not raw_content:
        return []

    try:
        parsed_data = json.loads(raw_content)

        if isinstance(parsed_data, list):
            return parsed_data

        if isinstance(parsed_data, dict):
            if isinstance(parsed_data.get("data"), list):
                return parsed_data["data"]

            if isinstance(parsed_data.get("questions"), list):
                return parsed_data["questions"]

            return [parsed_data]

        return []

    except json.JSONDecodeError:
        records = []

        for line_number, line in enumerate(
            raw_content.splitlines(),
            start=1
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                print(
                    f"Bỏ qua dòng JSON lỗi: "
                    f"{line_number} trong {file_path}"
                )

        return records


# ============================================================
# 4. TÁCH MỖI ĐIỀU LUẬT THÀNH CÁC KHOẢN
# ============================================================
def split_content_into_sections(content):
    """
    Ví dụ:

        1. Nội dung khoản 1.
        2. Nội dung khoản 2.
        3. Nội dung khoản 3.

    sẽ được tách thành ba chunk.

    Nếu không tìm thấy cấu trúc đánh số, giữ nguyên toàn bộ content.
    """
    if content is None:
        return []

    content = str(content).strip()

    if not content:
        return []

    # Match các khoản dạng "1. ", "2. ", "3. "
    # Không match ngày tháng hoặc số hiệu như 24/2016/NĐ-CP.
    pattern = re.compile(
        r"(?<!\S)(\d{1,3})\.\s+",
        flags=re.UNICODE
    )

    matches = list(pattern.finditer(content))

    if not matches:
        return [{
            "section_number": None,
            "section_text": content
        }]

    sections = []

    # Giữ phần mở đầu nếu có.
    prefix = content[:matches[0].start()].strip()

    if prefix:
        sections.append({
            "section_number": None,
            "section_text": prefix
        })

    for index, match in enumerate(matches):
        section_number = match.group(1)
        start_position = match.end()

        if index + 1 < len(matches):
            end_position = matches[index + 1].start()
        else:
            end_position = len(content)

        section_text = content[start_position:end_position].strip()

        if section_text:
            sections.append({
                "section_number": section_number,
                "section_text": section_text
            })

    return sections


# ============================================================
# 5. TẠO CÁC CHUNK DÙNG CHO RETRIEVAL
# ============================================================
def load_corpus_as_chunks(corpus_file):
    raw_documents = load_json_or_jsonl(corpus_file)

    chunks = []

    for document_index, doc in enumerate(raw_documents):
        if not isinstance(doc, dict):
            continue

        document_id = str(doc.get("id", document_index))
        title = str(doc.get("title", ""))
        content = str(doc.get("content", ""))
        chude_name = str(doc.get("chude_name", ""))
        demuc_name = str(doc.get("demuc_name", ""))

        sections = split_content_into_sections(content)

        for section_index, section in enumerate(sections):
            section_number = section["section_number"]
            section_text = section["section_text"]

            # Text dùng để search.
            retrieval_text = " ".join([
                chude_name,
                demuc_name,
                title,
                section_text
            ]).strip()

            normalized_text = normalize_text(retrieval_text)

            if not normalized_text:
                continue

            chunk_id = (
                f"{document_id}__section_{section_number}"
                if section_number is not None
                else f"{document_id}__section_{section_index}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": title,
                "section_number": section_number,
                "section_text": section_text,
                "retrieval_text": retrieval_text,
                "normalized_text": normalized_text
            })

    return chunks


# ============================================================
# 6. XÂY TF-IDF INDEX CHẠY NHANH
# ============================================================
def build_tfidf_index(chunks):
    """
    TF-IDF unigram + bigram.

    Matrix được L2-normalize mặc định.
    Vì vậy:
        query_vector @ chunk_matrix.T
    chính là cosine similarity.
    """
    texts = [
        chunk["normalized_text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2"
    )

    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix


# ============================================================
# 7. TRUY XUẤT TOP-K CHUNK
# ============================================================
def retrieve_top_k_chunks(
    question,
    chunks,
    vectorizer,
    tfidf_matrix,
    top_k=TOP_K
):
    normalized_question = normalize_text(question)

    query_vector = vectorizer.transform([
        normalized_question
    ])

    # Sparse matrix multiplication.
    scores = (
        query_vector @ tfidf_matrix.T
    ).toarray().reshape(-1)

    actual_k = min(top_k, len(chunks))

    if actual_k <= 0:
        return []

    if actual_k == len(chunks):
        top_indices = np.argsort(scores)[::-1]
    else:
        candidate_indices = np.argpartition(
            scores,
            -actual_k
        )[-actual_k:]

        top_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]

    results = []

    for index in top_indices:
        results.append({
            "chunk_index": int(index),
            "retrieval_score": float(scores[index]),
            **chunks[index]
        })

    return results


# ============================================================
# 8. TẠO PROMPT CHO FLAN-T5
# ============================================================
def build_prompt(item, retrieved_chunks):
    """
    Prompt tiếng Anh vì FLAN-T5 được instruction-tune.
    Câu hỏi, đáp án và nội dung pháp luật vẫn giữ tiếng Việt.

    Mô hình chỉ cần chọn A, B, C hoặc D.
    """
    context_parts = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        context_parts.append(
            f"[Đoạn {rank}] "
            f"{chunk['title']}. "
            f"{chunk['section_text']}"
        )

    context = "\n".join(context_parts)

    prompt = f"""
Read the Vietnamese legal context and answer the multiple-choice question.
Choose the most accurate answer based only on the context.
Return only one letter: A, B, C, or D.

Context:
{context}

Question:
{item.get("question", "")}

A. {item.get("A", "")}
B. {item.get("B", "")}
C. {item.get("C", "")}
D. {item.get("D", "")}

Answer:
""".strip()

    return prompt


# ============================================================
# 9. LOAD FLAN-T5-LARGE
# ============================================================
def load_model():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print(f"Thiết bị: {device}")
    print(f"Kiểu dữ liệu: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 10. CHẤM XÁC SUẤT CHO A, B, C, D
# ============================================================
def score_answer_letters(
    prompts,
    tokenizer,
    model,
    device
):
    """
    Với mỗi prompt:
        - tạo 4 bản sao
        - lần lượt ép output mục tiêu là A, B, C, D
        - tính negative log-likelihood
        - chữ cái có loss nhỏ nhất được chọn

    Cách này ổn định hơn gọi model.generate() rồi xử lý
    trường hợp mô hình trả lời dài dòng.
    """
    expanded_prompts = []
    target_letters = []

    for prompt in prompts:
        for letter in VALID_CHOICES:
            expanded_prompts.append(prompt)
            target_letters.append(letter)

    inputs = tokenizer(
        expanded_prompts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt"
    ).to(device)

    label_tokens = tokenizer(
        target_letters,
        padding=True,
        return_tensors="pt"
    )["input_ids"]

    # Padding không tham gia loss.
    label_tokens[
        label_tokens == tokenizer.pad_token_id
    ] = -100

    label_tokens = label_tokens.to(device)

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode():
        with autocast_context:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=label_tokens
            )

            logits = outputs.logits

    # Cross entropy cho từng token output.
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        label_tokens.reshape(-1),
        reduction="none",
        ignore_index=-100
    )

    token_losses = token_losses.reshape(
        label_tokens.shape
    )

    valid_mask = (
        label_tokens != -100
    ).float()

    sequence_losses = (
        token_losses * valid_mask
    ).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

    # Mỗi câu hỏi có 4 loss tương ứng A, B, C, D.
    sequence_losses = (
        sequence_losses
        .reshape(len(prompts), 4)
        .detach()
        .cpu()
        .numpy()
    )

    predictions = []
    score_records = []

    for losses in sequence_losses:
        best_index = int(np.argmin(losses))
        predictions.append(
            VALID_CHOICES[best_index]
        )

        score_records.append({
            letter: float(loss)
            for letter, loss in zip(
                VALID_CHOICES,
                losses
            )
        })

    return predictions, score_records


# ============================================================
# 11. XỬ LÝ TOÀN BỘ ĐỀ THI
# ============================================================
def make_submission():
    total_start_time = time.time()

    # --------------------------------------------------------
    # Đọc và chia corpus
    # --------------------------------------------------------
    print("Đang đọc corpus và tách điều luật theo khoản...")

    start_time = time.time()

    chunks = load_corpus_as_chunks(
        CORPUS_FILE
    )

    if not chunks:
        raise ValueError(
            "Không tìm thấy chunk hợp lệ trong corpus."
        )

    print(
        f"Số chunk: {len(chunks):,}"
    )

    print(
        f"Thời gian tách chunk: "
        f"{time.time() - start_time:.2f} giây"
    )

    # --------------------------------------------------------
    # Xây TF-IDF index
    # --------------------------------------------------------
    print("\nĐang xây TF-IDF index...")

    start_time = time.time()

    vectorizer, tfidf_matrix = build_tfidf_index(
        chunks
    )

    print(
        f"Kích thước matrix: {tfidf_matrix.shape}"
    )

    print(
        f"Thời gian xây TF-IDF index: "
        f"{time.time() - start_time:.2f} giây"
    )

    # --------------------------------------------------------
    # Đọc đề thi
    # --------------------------------------------------------
    test_data = load_json_or_jsonl(
        TEST_FILE
    )

    print(
        f"\nSố câu hỏi: {len(test_data):,}"
    )

    # --------------------------------------------------------
    # Load FLAN-T5-large
    # --------------------------------------------------------
    print("\nĐang tải google/flan-t5-large...")

    tokenizer, model, device = load_model()

    print("Đã tải mô hình.")

    # --------------------------------------------------------
    # Truy xuất context cho tất cả câu hỏi
    # --------------------------------------------------------
    print("\nĐang truy xuất top-k chunk...")

    prompts = []
    retrieval_records = []

    retrieval_start_time = time.time()

    for question_index, item in enumerate(
        test_data,
        start=1
    ):
        retrieved_chunks = retrieve_top_k_chunks(
            question=item.get("question", ""),
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=TOP_K
        )

        prompt = build_prompt(
            item=item,
            retrieved_chunks=retrieved_chunks
        )

        prompts.append(prompt)

        retrieval_records.append(
            retrieved_chunks
        )

        if (
            question_index % 100 == 0
            or question_index == len(test_data)
        ):
            print(
                f"Đã retrieval: "
                f"{question_index}/{len(test_data)}"
            )

    print(
        f"Thời gian retrieval: "
        f"{time.time() - retrieval_start_time:.2f} giây"
    )

    # --------------------------------------------------------
    # Inference theo batch
    # --------------------------------------------------------
    print("\nBắt đầu inference bằng FLAN-T5-large...\n")

    submissions = []
    debug_records = []

    inference_start_time = time.time()

    for batch_start in range(
        0,
        len(test_data),
        BATCH_SIZE
    ):
        batch_end = min(
            batch_start + BATCH_SIZE,
            len(test_data)
        )

        batch_items = test_data[
            batch_start:batch_end
        ]

        batch_prompts = prompts[
            batch_start:batch_end
        ]

        predictions, score_records = (
            score_answer_letters(
                prompts=batch_prompts,
                tokenizer=tokenizer,
                model=model,
                device=device
            )
        )

        for local_index, (
            item,
            prediction,
            scores
        ) in enumerate(
            zip(
                batch_items,
                predictions,
                score_records
            )
        ):
            global_index = (
                batch_start + local_index
            )

            question_id = item.get(
                "id",
                global_index + 1
            )

            submissions.append({
                "id": question_id,
                "answer": prediction
            })

            debug_records.append({
                "id": question_id,
                "question": item.get(
                    "question",
                    ""
                ),
                "prediction": prediction,
                "letter_losses": scores,
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "retrieval_score": chunk[
                            "retrieval_score"
                        ],
                        "title": chunk["title"],
                        "section_number": chunk[
                            "section_number"
                        ],
                        "section_text": chunk[
                            "section_text"
                        ]
                    }
                    for chunk in retrieval_records[
                        global_index
                    ]
                ]
            })

        processed_count = batch_end

        if (
            processed_count % PRINT_EVERY == 0
            or processed_count == len(test_data)
        ):
            elapsed = (
                time.time()
                - inference_start_time
            )

            print(
                f"Đã xử lý: "
                f"{processed_count}/{len(test_data)} câu | "
                f"{elapsed:.2f} giây | "
                f"{elapsed / processed_count:.3f} giây/câu"
            )

    # --------------------------------------------------------
    # Lưu submission JSON
    # --------------------------------------------------------
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            submissions,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Lưu debug JSON
    # --------------------------------------------------------
    with open(
        DEBUG_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            debug_records,
            file,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Nén file submission
    # --------------------------------------------------------
    with zipfile.ZipFile(
        ZIP_FILE,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_object:
        zip_object.write(
            OUTPUT_FILE,
            arcname="submission.json"
        )

    # --------------------------------------------------------
    # Hoàn thành
    # --------------------------------------------------------
    print("\n================ HOÀN THÀNH ================")

    print(
        f"Tổng thời gian: "
        f"{time.time() - total_start_time:.2f} giây"
    )

    print(
        f"File đáp án JSON: {OUTPUT_FILE}"
    )

    print(
        f"File ZIP để nộp: {ZIP_FILE}"
    )

    print(
        f"File debug: {DEBUG_FILE}"
    )

    print("============================================")


# ============================================================
# 12. CHẠY
# ============================================================
make_submission()
