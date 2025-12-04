# app.py — Streamlit app with robust HF loader + full preprocessing for inputs (text & YouTube comments)
import os
import re
import html
import string
import time
import traceback
from typing import List, Optional

import streamlit as st
import pandas as pd
import torch
import requests
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from huggingface_hub import list_repo_files, hf_hub_download

# --- optional text libraries (Sastrawi) with graceful fallback ---
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _SASTRAWI_AVAILABLE = True
except Exception:
    _SASTRAWI_AVAILABLE = False

# ---------------- Config ----------------
DEFAULT_REPO = "yossss90/indobert_imbalance_1"  # ganti sesuai repo HF Anda
st.set_page_config(page_title="IndoBERT Classifier", layout="centered", initial_sidebar_state="expanded")

# ---------------- Preprocessing utilities (copied & adapted) ----------------
# FULL_UNICODE_NORMALIZATION_MAP (trimmed here for brevity — include full map from your script)
FULL_UNICODE_NORMALIZATION_MAP = {
    'Ａ':'A','Ｂ':'B','Ｃ':'C','Ｄ':'D','Ｅ':'E','Ｆ':'F','Ｇ':'G','Ｈ':'H','Ｉ':'I','Ｊ':'J','Ｋ':'K','Ｌ':'L','Ｍ':'M','Ｎ':'N','Ｏ':'O','Ｐ':'P','Ｑ':'Q','Ｒ':'R','Ｓ':'S','Ｔ':'T','Ｕ':'U','Ｖ':'V','Ｗ':'W','Ｘ':'X','Ｙ':'Y','Ｚ':'Z',
    'ａ':'a','ｂ':'b','ｃ':'c','ｄ':'d','ｅ':'e','ｆ':'f','ｇ':'g','ｈ':'h','ｉ':'i','ｊ':'j','ｋ':'k','ｌ':'l','ｍ':'m','ｎ':'n','ｏ':'o','ｐ':'p','𝟎':'0','𝟏':'1','𝟐':'2',
    # (NOTE) -- paste the complete mapping you had here in production
}

MULTI_CHAR_NORMALIZATION_MAP = {
    '0️⃣': '0', '1️⃣': '1', '2️⃣': '2', '3️⃣': '3', '4️⃣': '4', '5️⃣': '5', '6️⃣': '6', '7️⃣': '7', '8️⃣': '8', '9️⃣': '9',
    '❶': '1', '❷': '2', '❸': '3', '❹': '4', '❺': '5', '❻': '6', '❼': '7', '❽': '8', '❾': '9',
}

def normalize_and_clean_styles(text: str) -> str:
    # multi-char mapping
    for old, new in MULTI_CHAR_NORMALIZATION_MAP.items():
        text = text.replace(old, new)

    # strip combining diacritics / zero-width / variation selectors
    diacritic_stripper = re.compile(r"[\u0300-\u036f\u0483-\u0489\u200b-\u200f\u20d0-\u20ff\ufe0e\ufe0f]")
    text = diacritic_stripper.sub('', text)

    # map characters via translation table
    trans_table = str.maketrans(FULL_UNICODE_NORMALIZATION_MAP)
    text = text.translate(trans_table)
    return text

def clean_text_modified(text: str) -> str:
    text = str(text)

    # remove anchor tags content
    text = re.sub(r'<a[^>]*>.*?</a>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    # remove any html tag
    text = re.sub(r'<[^>]+>', ' ', text)
    # remove urls (including something.tld/...)
    url_pattern = re.compile(r'(?:https?://|www\.)\S+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/\S*)?')
    text = url_pattern.sub(' ', text)

    # normalize fancy unicode characters
    text = normalize_and_clean_styles(text)

    # unescape HTML entities
    text = html.unescape(text)

    # remove punctuation except keep hyphen '-'
    punc_to_remove = string.punctuation.replace('-', '')
    pattern = r'[' + re.escape(punc_to_remove) + r']'
    text = re.sub(pattern, ' ', text)

    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_stemmer(text: str, stemmer) -> str:
    new_tokens = []
    for token in text.split():
        if token.isalpha():
            try:
                new_tokens.append(stemmer.stem(token))
            except Exception:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    return " ".join(new_tokens)

# create sastrawi objects if available
if _SASTRAWI_AVAILABLE:
    try:
        stop_factory = StopWordRemoverFactory()
        stop_remover = stop_factory.create_stop_word_remover()
        stem_factory = StemmerFactory()
        stemmer = stem_factory.create_stemmer()
    except Exception:
        stop_remover = None
        stemmer = None
else:
    stop_remover = None
    stemmer = None

def preprocess_text_full(text: str) -> str:
    # 1) clean & normalize unicode / html / urls / punctuation
    t = clean_text_modified(text)
    # 2) lowercase
    t = t.lower()
    # 3) stopword remove (if available)
    if stop_remover is not None:
        try:
            t = stop_remover.remove(t)
        except Exception:
            pass
    # 4) safe stem
    if stemmer is not None:
        try:
            t = safe_stemmer(t, stemmer)
        except Exception:
            pass
    # final whitespace collapse
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ---------------- Robust HF loader (same as before) ----------------
def find_model_subfolders(repo_id: str, token: Optional[str] = None) -> List[str]:
    files = list_repo_files(repo_id, token=token)
    folders = set()
    for f in files:
        if f.endswith("config.json"):
            if "/" in f:
                folders.add(f.rsplit("/", 1)[0])
            else:
                folders.add("")  # root
    nonroot = [f for f in folders if f]
    return nonroot + ([""] if "" in folders else [])

def try_from_pretrained(repo_id: str, subfolder: Optional[str], device: int, token: Optional[str] = None):
    kwargs = {}
    if token:
        kwargs["use_auth_token"] = token
    tok = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder, local_files_only=False, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=subfolder, local_files_only=False, **kwargs)
    pipe = pipeline("text-classification", model=model, tokenizer=tok, return_all_scores=True, device=device)
    return pipe

def download_and_load_local(repo_id: str, subfolder: Optional[str], token: Optional[str], device: int):
    cache_root = os.path.join("model_cache", repo_id.replace("/", "_"))
    if subfolder:
        cache_dir = os.path.join(cache_root, subfolder.replace("/", "_"))
    else:
        cache_dir = os.path.join(cache_root, "root")
    os.makedirs(cache_dir, exist_ok=True)

    files = list_repo_files(repo_id, token=token)

    candidates = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt")]
    model_file = None
    if subfolder:
        for c in candidates:
            if c.startswith(subfolder + "/"):
                model_file = c
                break
    else:
        for c in candidates:
            if "/" not in c:
                model_file = c
                break
    if model_file is None and candidates:
        model_file = candidates[0]

    def join(folder, name):
        return f"{folder}/{name}" if folder else name

    needed = set()
    if model_file:
        needed.add(model_file)
    needed.add(join(subfolder or "", "config.json"))
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "merges.txt", "tokenizer.model"]:
        path = join(subfolder or "", name)
        if path in files:
            needed.add(path)

    for f in files:
        if subfolder:
            if f.startswith(subfolder + "/") and (f.endswith(".json") or f.endswith(".txt") or f.endswith(".model")):
                needed.add(f)
        else:
            if "/" not in f and (f.endswith(".json") or f.endswith(".txt") or f.endswith(".model")):
                needed.add(f)

    for fn in sorted(needed):
        if not fn or fn not in files:
            continue
        try:
            hf_hub_download(repo_id=repo_id, filename=fn, local_dir=cache_dir, token=token, local_dir_use_symlinks=False)
        except Exception as e:
            print(f"Warning: failed to download {fn}: {e}")

    tok = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(cache_dir, local_files_only=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tok, return_all_scores=True, device=device)
    return pipe

@st.cache_resource
def load_pipeline_hf(repo_id: str, device_choice: str = "auto"):
    hf_token = None
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    if device_choice == "cpu":
        device = -1
    elif device_choice == "gpu":
        device = 0
    else:
        device = 0 if torch.cuda.is_available() else -1

    try:
        return try_from_pretrained(repo_id, subfolder=None, device=device, token=hf_token), device
    except Exception as e_root:
        root_err = e_root
        print("Direct failed:", e_root)

    try:
        candidates = find_model_subfolders(repo_id, token=hf_token)
    except Exception as e_list:
        candidates = []
        print("list_repo_files failed:", e_list)

    for folder in candidates:
        try:
            pipe = try_from_pretrained(repo_id, subfolder=folder if folder else None, device=device, token=hf_token)
            return pipe, device
        except Exception as e:
            print(f"from_pretrained with subfolder='{folder}' failed:", e)

    fallback_folder = candidates[0] if candidates else ""
    try:
        pipe = download_and_load_local(repo_id, fallback_folder if fallback_folder else None, token=hf_token, device=device)
        return pipe, device
    except Exception as e_dl:
        print("Download-and-load failed:", e_dl)
        raise RuntimeError(
            "Failed to load model from Hugging Face repo. "
            f"Root error: {root_err}\nDownload fallback error: {e_dl}"
        )

# ---------------- Utility helpers ----------------
def get_top_prediction(scores_list):
    best = max(scores_list, key=lambda x: x["score"])
    return best["label"], float(best["score"])

def normalize_label(lbl: str):
    if lbl.startswith("LABEL_"):
        return lbl.replace("LABEL_", "")
    return lbl

# ---------------- YouTube helpers ----------------
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"

def extract_video_id(url: str) -> Optional[str]:
    regexes = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/v/([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?.*v=([A-Za-z0-9_-]{11})"
    ]
    for r in regexes:
        m = re.search(r, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    return None

@st.cache_data(ttl=60*60)
def fetch_youtube_comments(video_id: str, api_key: str, max_comments: int = 200) -> List[str]:
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "textFormat": "plainText",
        "key": api_key,
    }
    nextPageToken = None
    while True:
        if nextPageToken:
            params["pageToken"] = nextPageToken
        resp = requests.get(YOUTUBE_API_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"YouTube API error {resp.status_code}: {resp.text}")
        data = resp.json()
        items = data.get("items", [])
        for it in items:
            try:
                text = it["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(text)
                if len(comments) >= max_comments:
                    return comments[:max_comments]
            except Exception:
                continue
        nextPageToken = data.get("nextPageToken")
        if not nextPageToken:
            break
        time.sleep(0.1)
    return comments

# ---------------- Sidebar UI ----------------
st.sidebar.header("Settings")
repo_input = st.sidebar.text_input("Model repo / folder", value=DEFAULT_REPO,
                                  help="Hugging Face repo (username/repo) or local path")
device_opt = st.sidebar.selectbox("Device", options=["auto", "cpu", "gpu"], index=0,
                                  help="auto -> use GPU if available")
show_raw = st.sidebar.checkbox("Show raw scores (for debugging)", value=False)
example_btn = st.sidebar.button("Use example text")
st.sidebar.markdown("---")
st.sidebar.write("YouTube API key (optional for comments): set Streamlit secret `YOUTUBE_API_KEY` or env var `YOUTUBE_API_KEY`.")
st.sidebar.caption("If missing, YouTube comment feature won't work.")

# ---------------- Main UI ----------------
st.title("🧪 IndoBERT — Comment Classification")
st.subheader("Single Text or YouTube link")

# Load model (cached)
with st.spinner("Memuat model ..."):
    try:
        nlp, device_used = load_pipeline_hf(repo_input, device_choice=device_opt)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"Gagal memuat model dari `{repo_input}`:\n{e}")
        st.code(tb)
        st.stop()

# choose input mode
mode = st.radio("Pilih mode input:", ["Single Text", "YouTube URL (comments)"])

if example_btn:
    default_text = "Produk ini sangat memuaskan. Pengiriman cepat dan kualitasnya bagus."
else:
    default_text = ""

# ---------------- Text single mode (with preprocessing) ----------------
if mode == "Text single":
    text = st.text_area("Masukkan teks untuk diklasifikasi", value=default_text, height=140)
    if st.button("Analyze Text"):
        if not text or not text.strip():
            st.warning("Input tidak boleh kosong.")
        else:
            with st.spinner("Melakukan preprocessing & inference..."):
                pre = preprocess_text_full(text)
                out = nlp(pre)
                scores = out[0]
                top_label, top_score = get_top_prediction(scores)
                display_label = normalize_label(top_label)

            st.markdown("### 🔎 Prediksi Akhir")
            st.write("**Original:**", text)
            st.write("**Preprocessed:**", pre)
            st.metric(label="Predicted class", value=f"{display_label}", delta=f"{top_score:.4f}")
            st.caption("Probabilitas di metric adalah probabilitas kelas terpilih.")

            df = pd.DataFrame([{ "label": normalize_label(x["label"]), "score": x["score"] } for x in scores])
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            st.markdown("indobert-online-gambling-detection")
            st.bar_chart(df.set_index("label"))

            if show_raw:
                st.markdown("#### Raw scores")
                st.json(scores)

# ---------------- YouTube comments mode (with preprocessing) ----------------
else:
    youtube_url = st.text_input("Enter YouTube link or directly video id:", value="")
    max_comments = st.slider("Maximum number of comments", min_value=10, max_value=1000, value=200, step=10)
    analyze_btn = st.button("Analyze comments")

    if analyze_btn:
        vid = extract_video_id(youtube_url)
        if not vid:
            st.error("Tidak dapat mengekstrak video id. Pastikan URL benar.")
        else:
            api_key = "AIzaSyDfe_-OqnsyKqdTUskaK9aotvyNL3cs2w0"
            try:
                api_key = st.secrets["YOUTUBE_API_KEY"]
            except Exception:
                api_key = os.environ.get("YOUTUBE_API_KEY")
            if not api_key:
                st.error("YouTube API key tidak ditemukan. Set `YOUTUBE_API_KEY` di Streamlit secrets atau env var.")
            else:
                with st.spinner("Mengambil komentar dari YouTube..."):
                    try:
                        comments = fetch_youtube_comments(vid, api_key, max_comments=max_comments)
                    except Exception as e:
                        st.error(f"Gagal mengambil komentar: {e}")
                        comments = []

                if not comments:
                    st.warning("Tidak ada komentar yang berhasil diambil (atau komentar dinonaktifkan).")
                else:
                    st.success(f"Fetched  {len(comments)} comments — running preprocessing & inference...")
                    batch_size = 32
                    preds = []
                    confidences = []
                    texts = []
                    preprocessed_texts = []
                    progress_bar = st.progress(0)
                    total = len(comments)
                    for i in range(0, total, batch_size):
                        batch = comments[i:i+batch_size]
                        # preprocess batch first
                        pre_batch = [preprocess_text_full(c) for c in batch]
                        try:
                            outs = nlp(pre_batch)
                        except Exception:
                            # fallback single
                            outs = [nlp(pb)[0] for pb in pre_batch]
                        for out in outs:
                            scores = out
                            label, conf = get_top_prediction(scores)
                            preds.append(normalize_label(label))
                            confidences.append(conf)
                        texts.extend(batch)
                        preprocessed_texts.extend(pre_batch)
                        progress_bar.progress(min(1.0, (i+batch_size)/total))
                    progress_bar.empty()

                    df_res = pd.DataFrame({
                        "comment": texts,
                        "preprocessed": preprocessed_texts,
                        "predicted_label": preds,
                        "confidence": confidences
                    })

    
                    label_mapping = {
                        0: "Netral",
                        1: "Toxic",
                        2: "Judol",
                        "LABEL_0": "Netral",
                        "LABEL_1": "Toxic",
                        "LABEL_2": "Judol",
                        "0": "Netral", 
                        "1": "Toxic", 
                        "2": "Judol"
                    }

                    counts = df_res["predicted_label"].value_counts().sort_index()

 
                    mapped_labels = [label_mapping.get(x, f"Label {x}") for x in counts.index]

                    st.markdown("### 📊 Distribusi Kelas (Komentar)")
                    fig, ax = plt.subplots()

                    ax.pie(
                        counts.values, 
                        labels=mapped_labels, 
                        autopct='%1.1f%%', 
                        startangle=90
                    )
                    ax.axis('equal')
                    st.pyplot(fig)

                    st.markdown("### 🔎 Results Table")
                    st.dataframe(df_res.head(200))

                    csv = df_res.to_csv(index=False)
                    st.download_button("Download hasil (CSV)", csv, file_name=f"yt_comments_pred_{vid}.csv", mime="text/csv")

                    if show_raw:
                        st.markdown("#### All predictions")
                        st.write(df_res)

# Footer / notes
st.markdown("---")
st.write("Notes:")
st.write("- Preprocessing applied: unicode normalization, HTML/url removal, punctuation cleanup, lowercasing, optional stopword removal & stemming (Sastrawi if installed).")
st.caption("Developed with ❤️ by Group 4")