# app.py — Streamlit app using Hugging Face Inference API + full preprocessing for inputs (text & YouTube comments)
import os
import re
import html
import string
import time
import traceback
from typing import List, Optional, Iterable

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except Exception:
    plt = None
    _MPL_AVAILABLE = False

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _SASTRAWI_AVAILABLE = True
except Exception:
    _SASTRAWI_AVAILABLE = False

# ---------------- Config ----------------
DEFAULT_REPO = "yossss90/indobert_imbalance_1"  # ganti sesuai repo HF Anda
st.set_page_config(page_title="IndoBERT Classifier (HF API)", layout="centered", initial_sidebar_state="expanded")

# ---------------- Preprocessing utilities (same as your original) ----------------
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

# ---------------- HF Inference API wrapper (replaces local model/pipeline) ----------------
def call_hf_inference_batch(repo_id: str, inputs: Iterable[str], token: Optional[str], timeout: int = 60):
    """
    Call Hugging Face Inference API with a batch (list) of input strings.
    Returns: list of outputs per input (each output is list of {label, score})
    Raises RuntimeError on failure.
    """
    api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    # HF accepts a list of inputs for batch inference
    try:
        resp = requests.post(api_url, headers=headers, json={"inputs": list(inputs)}, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")
    if resp.status_code != 200:
        # try to extract message
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Hugging Face API error {resp.status_code}: {body}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse HF response: {e}")

def call_hf_inference(repo_id: str, text: str, token: Optional[str], timeout: int = 60):
    out = call_hf_inference_batch(repo_id, [text], token, timeout=timeout)
    # out expected: [ [ {label, score}, ... ] ]
    if isinstance(out, list) and len(out) > 0:
        return out[0]
    return out

# ---------------- Utility helpers ----------------
def get_top_prediction(scores_list):
    best = max(scores_list, key=lambda x: x["score"])
    return best["label"], float(best["score"])

def normalize_label(lbl: str):
    if isinstance(lbl, str) and lbl.startswith("LABEL_"):
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
                                  help="Hugging Face repo (username/repo)")
device_opt = st.sidebar.selectbox("Device (ignored for HF API)", options=["auto", "cpu", "gpu"], index=0,
                                  help="Device selection is ignored when using HF Inference API")
show_raw = st.sidebar.checkbox("Show raw scores (for debugging)", value=False)
example_btn = st.sidebar.button("Use example text")
st.sidebar.markdown("---")
st.sidebar.write("YouTube API key (optional for comments): set Streamlit secret `YOUTUBE_API_KEY` or env var `YOUTUBE_API_KEY`.")
st.sidebar.caption("If missing, YouTube comment feature won't work.")

# ---------------- Main UI ----------------
st.title("🧪 IndoBERT — Comment Classification (via Hugging Face Inference API)")
st.subheader("Single Text or YouTube link")

# Retrieve HF token from secrets or env
hf_token = None
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# Quick check: ensure we have repo and optionally token
with st.spinner("Memeriksa akses ke Hugging Face..."):
    try:
        # We do not perform heavy verification; just attempt a small ping inference with a short input
        test_out = call_hf_inference(repo_input, "test", hf_token, timeout=10)
        # If no exception, OK
        st.success("Hugging Face model reachable (via Inference API).")
    except Exception as e:
        st.warning(f"Warning: Tidak dapat memverifikasi repo/token lewat API. Error singkat: {e}. Aplikasi akan tetap mencoba saat inference.")
        # not fatal here — will show runtime errors later if calls fail

# choose input mode
if example_btn:
    default_text = "Produk ini sangat memuaskan. Pengiriman cepat dan kualitasnya bagus."
else:
    default_text = ""

mode = st.radio("Pilih mode input:", ["Single Text", "YouTube URL (comments)"])

# ---------------- Text single mode (with preprocessing) ----------------
if mode == "Single Text":
    text = st.text_area("Masukkan teks untuk diklasifikasi", value=default_text, height=140)
    if st.button("Analyze Text"):
        if not text or not text.strip():
            st.warning("Input tidak boleh kosong.")
        else:
            with st.spinner("Melakukan preprocessing & inference via HF Inference API..."):
                pre = preprocess_text_full(text)
                try:
                    out = call_hf_inference(repo_input, pre, hf_token, timeout=60)
                except Exception as e:
                    tb = traceback.format_exc()
                    st.error(f"Gagal memanggil Hugging Face Inference API:\n{e}")
                    st.code(tb)
                    st.stop()

                scores = out  # list of dicts
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
            api_key = None
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
                    st.success(f"Fetched  {len(comments)} comments — running preprocessing & inference via HF API...")
                    batch_size = 32  # chunk size for HF API calls
                    preds = []
                    confidences = []
                    texts = []
                    preprocessed_texts = []
                    progress_bar = st.progress(0)
                    total = len(comments)

                    for i in range(0, total, batch_size):
                        batch = comments[i:i+batch_size]
                        pre_batch = [preprocess_text_full(c) for c in batch]

                        # call HF in batches (to avoid huge payloads)
                        try:
                            outs = call_hf_inference_batch(repo_input, pre_batch, hf_token, timeout=120)
                        except Exception as e:
                            # fallback: try single-call per item (slower) to get partial results
                            outs = []
                            for pb in pre_batch:
                                try:
                                    single = call_hf_inference(repo_input, pb, hf_token, timeout=60)
                                    outs.append(single)
                                except Exception:
                                    outs.append([{"label":"ERROR","score":0.0}])

                        # outs is list-per-input
                        for out in outs:
                            if isinstance(out, list) and out:
                                label, conf = get_top_prediction(out)
                            else:
                                label, conf = ("ERROR", 0.0)
                            preds.append(normalize_label(label))
                            confidences.append(conf)

                        texts.extend(batch)
                        preprocessed_texts.extend(pre_batch)
                        progress_bar.progress(min(1.0, (i + batch_size) / total))
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

# debug-info (sementara - hapus setelah selesai)
import sys, pkgutil
import streamlit as st
st.sidebar.markdown("*Debug info (temp)*")
st.sidebar.write("Python:", sys.version)
installed = sorted([m.name for m in pkgutil.iter_modules()])
# tampilkan apakah matplotlib ada
st.sidebar.write("matplotlib present:", "matplotlib" in installed)
# optional: show top 30 installed packages
st.sidebar.write("Installed pkgs sample:", installed[:30])