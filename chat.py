import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from fuzzywuzzy import process
import re

# --- Load PhoBERT ---
@st.cache_resource
def load_phobert():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", cache_dir="./phobert_cache")
    model = AutoModel.from_pretrained("vinai/phobert-base", cache_dir="./phobert_cache")
    return tokenizer, model

# --- Embedding ---
def get_embedding(text, _tokenizer, _model):
    if text is None: return None
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# --- Chuẩn hóa ---
def normalize_column(series):
    mapping = {
        "đại số": "Đại số", "đai số": "Đại số",
        "hình học": "Hình học", "hÌnh học": "Hình học",
        "bài tập": "Bài tập", "bài tập ": "Bài tập",
        "bài tập trắc nghiệm": "Bài tập trắc nghiệm",
        "lý thuyết": "Lý thuyết"
    }
    return series.fillna("").apply(lambda x: mapping.get(str(x).strip().lower(), str(x).strip()))

# --- Load dữ liệu ---
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Câu hỏi"] = df["Câu hỏi"].astype(str).str.strip()
    df["Câu trả lời"] = df["Câu trả lời"].astype(str).str.strip()
    df = df.dropna(subset=["Câu hỏi", "Câu trả lời"])
    df["Chủ đề"] = normalize_column(df["Chủ đề"])
    df["Thể loại"] = normalize_column(df["Thể loại"])
    df["Lớp"] = df["Lớp"].astype(str).str.strip()
    return df

# --- FAISS Index ---
@st.cache_resource
def build_faiss_index(df, _tokenizer, _model):
    embeddings, valid_idx = [], []
    for i, row in df.iterrows():
        emb = get_embedding(row["Câu hỏi"], _tokenizer, _model)
        if emb is not None:
            embeddings.append(emb)
            valid_idx.append(i)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    return index, valid_idx

# --- Fuzzy Matching ---
def fuzzy_match(query, df):
    best = process.extractOne(query, df["Câu hỏi"].tolist())
    if best:
        matched = df[df["Câu hỏi"] == best[0]].iloc[0]
        return matched["Câu trả lời"], matched.get("Hướng dẫn giải", ""), best[1]
    return "Không tìm thấy", "", 0

# --- Tính toán biểu thức toán học ---
def evaluate_expression(query):
    try:
        result = eval(query, {"__builtins__": {}})
        return result
    except Exception:
        return None

# --- Trạng thái hội thoại ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.stage = "start"
    st.session_state.grade = ""
    st.session_state.topic = ""
    st.session_state.type = ""
    st.session_state.quiz_row = None

# --- Load mô hình & dữ liệu ---
tokenizer, model = load_phobert()
df = load_data("Toan.csv")
index, valid_idx = build_faiss_index(df, tokenizer, model)

# --- Giao diện ---
st.title("📚 Chatbot Toán học")
st.markdown("Nhập câu hỏi lý thuyết, biểu thức toán học hoặc gõ **`Hãy cho tôi bài tập`** để bắt đầu.")
if st.button("🧹 Xóa hội thoại"):
    st.session_state.messages = []
    st.session_state.stage = "start"
    st.session_state.grade = ""
    st.session_state.topic = ""
    st.session_state.type = ""
    st.session_state.quiz_row = None
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Nhập câu hỏi hoặc yêu cầu bài tập...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # === Luồng bài tập ===
    if st.session_state.stage == "start" and "bài tập" in query.lower():
        st.session_state.stage = "ask_grade"
        msg = "📌 Bạn muốn chọn **lớp mấy**?"

    elif st.session_state.stage == "ask_grade":
        st.session_state.grade = query.strip()
        st.session_state.stage = "ask_topic"
        msg = "📘 Bạn muốn chọn **chủ đề** nào? _(Đại số / Hình học)_"

    elif st.session_state.stage == "ask_topic":
        st.session_state.topic = query.strip()
        st.session_state.stage = "ask_type"
        msg = "📗 Bạn muốn chọn **thể loại** nào? _(Bài tập / Bài tập trắc nghiệm / Lý thuyết)_"

    elif st.session_state.stage == "ask_type":
        st.session_state.type = query.strip()
        df_filtered = df[
            (df["Lớp"] == st.session_state.grade) &
            (df["Chủ đề"].str.lower() == st.session_state.topic.lower()) &
            (df["Thể loại"].str.lower() == st.session_state.type.lower())
        ]
        if not df_filtered.empty:
            row = df_filtered.sample(1).iloc[0]
            st.session_state.quiz_row = row
            st.session_state.stage = "answer_quiz"
            msg = f"📝 **Câu hỏi**: {row['Câu hỏi']}"
        else:
            st.session_state.stage = "start"
            msg = "❌ Không tìm thấy bài tập phù hợp. Vui lòng thử lại."

    elif st.session_state.stage == "answer_quiz":
        user_ans = query.strip().lower()
        correct = st.session_state.quiz_row["Câu trả lời"].strip().lower()
        if user_ans == correct:
            msg = "✅ Chính xác!"
        else:
            msg = f"❌ Chưa đúng. Đáp án đúng là: **{st.session_state.quiz_row['Câu trả lời']}**"
            if pd.notna(st.session_state.quiz_row.get("Hướng dẫn giải", "")):
                msg += f"\n📖 **Hướng dẫn giải**: {st.session_state.quiz_row['Hướng dẫn giải']}"
        st.session_state.stage = "start"

    # === Luồng lý thuyết ===
    elif st.session_state.stage == "start":
        if re.fullmatch(r"[0-9\s\+\-\*/().]+", query.strip()):
            result = evaluate_expression(query)
            if result is not None:
                msg = f"🧮 Kết quả của biểu thức `{query}` là: **{result}**"
            else:
                msg = f"⚠️ Không thể tính toán biểu thức: `{query}`"
        else:
            emb = get_embedding(query, tokenizer, model)
            D, I = index.search(np.array([emb]).astype("float32"), 1)
            if D[0][0] > 1.0:
                ans, hint, score = fuzzy_match(query, df)
                msg = f"**Fuzzy Matching ({score}%)**\n\n🧠 {ans}"
                if pd.notna(hint) and hint.strip():
                    msg += f"\n📖 **Hướng dẫn giải**: {hint}"
            else:
                row = df.iloc[valid_idx[I[0][0]]]
                msg = f"🧠 **{row['Câu trả lời']}**"
                if pd.notna(row.get("Hướng dẫn giải", "")):
                    msg += f"\n📖 **Hướng dẫn giải**: {row['Hướng dẫn giải']}"

    with st.chat_message("assistant"):
        st.markdown(msg)
    st.session_state.messages.append({"role": "assistant", "content": msg})
