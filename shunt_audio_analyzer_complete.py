# -*- coding: utf-8 -*-
"""
シャント音 解析ビューア 完全版（HLPR FFT視覚化付き + 2ファイル比較 + STFT比較 + CSV拡張）
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import streamlit as st
from datetime import datetime
from scipy.signal import (
    butter, filtfilt, iirnotch, welch, hilbert,
    get_window, stft as sp_stft, resample_poly
)

st.set_page_config(page_title="Shunt Sound Analyzer - 完全版", layout="wide")

# ---- UI小道具 ----
def explain_button(title: str, body_md: str):
    with st.expander(f"ℹ️ {title} の説明"):
        st.markdown(body_md)

# ---- DSP utils ----
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.0001, lowcut / nyq)
    high = min(0.9999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass(x, fs, low, high, order=4):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)

def apply_notch(x, fs, freq=50.0, q=30.0):
    b, a = iirnotch(freq/(fs/2), q)
    return filtfilt(b, a, x)

def compute_psd_welch(x, fs, nperseg=2048, noverlap=1024):
    f, pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, pxx

def compute_stft(x, fs, n_fft=2048, hop=256, win="hann"):
    f, t, Zxx = sp_stft(
        x, fs=fs, nperseg=n_fft,
        noverlap=n_fft-hop,
        window=get_window(win, n_fft)
    )
    S = np.abs(Zxx)
    return f, t, S

def band_envelope(x, fs, band, order=4):
    y = apply_bandpass(x, fs, band[0], band[1], order=order)
    env = np.abs(hilbert(y))
    return y, env

def calculate_hlpr(x, fs, high_band=(500, 700), low_band=(100, 250), order=4):
    _, high_env = band_envelope(x, fs, high_band, order=order)
    _, low_env = band_envelope(x, fs, low_band, order=order)
    high_peak = np.max(high_env)
    low_peak = np.max(low_env)
    hlpr = high_peak / (low_peak + 1e-9)
    return hlpr, high_peak, low_peak

def calculate_hlpr_fft(x, fs, low_band=(100, 250), high_band=(500, 700)):
    freqs = np.fft.rfftfreq(len(x), d=1/fs)
    fft_spectrum = np.abs(np.fft.rfft(x))
    idx_low = np.where((freqs >= low_band[0]) & (freqs <= low_band[1]))
    idx_high = np.where((freqs >= high_band[0]) & (freqs <= high_band[1]))
    low_peak = np.max(fft_spectrum[idx_low])
    high_peak = np.max(fft_spectrum[idx_high])
    hlpr_fft = high_peak / (low_peak + 1e-9)
    return hlpr_fft, freqs, fft_spectrum, low_peak, high_peak

# ---- サイドバー ----
with st.sidebar:
    st.header("2) 前処理")
    target_sr = st.selectbox("解析サンプリング周波数", [2000, 4000, 8000, 16000], index=2)
    use_notch = st.checkbox("ノッチ除去（商用電源）", value=True)
    notch_freq = st.selectbox("ノッチ周波数", [50, 60], index=0)
    notch_q = st.slider("ノッチQ（鋭さ）", 10, 60, 30)
    bp_low = st.number_input("バンドパス下限 [Hz]", 0.0, 5000.0, 20.0, 10.0)
    bp_high = st.number_input("バンドパス上限 [Hz]", 50.0, 20000.0, 1200.0, 50.0)
    bp_order = st.slider("バンドパス次数", 2, 8, 4)

    st.header("3) 出力")
    export_csv = st.checkbox("CSV出力（全解析結果）", value=True)

# ---- メイン ----
st.title("シャント音 解析ビューア（STFT/PSD/HLPR/FFT + 2ファイル比較）")

# === 基本情報入力 ===
st.subheader("基本情報の入力")
col1, col2 = st.columns(2)
with col1:
    shunt_type = st.radio("シャントの種類", ["AVG", "AVF"], horizontal=True)
    sex = st.radio("性別", ["男性", "女性"], horizontal=True)
with col2:
    site = st.radio("測定場所", ["吻合部", "その他"], horizontal=True)
    site_comment = "" if site != "その他" else st.text_input("その他の測定場所を入力してください")

# ===音声ファイル ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("音声ファイル1")
    up1 = st.file_uploader("音声ファイル1 (WAV/MP3/FLAC/OGG/M4A)", type=["wav", "mp3", "flac", "ogg", "m4a"])

with col2:
    st.subheader("音声ファイル2（比較用・任意）")
    up2 = st.file_uploader("音声ファイル2 (比較用)", type=["wav", "mp3", "flac", "ogg", "m4a"], key="second")

if up1 is None:
    st.warning("音声ファイル1をアップロードしてください")
    st.stop()

# ---- 音声読込関数 ----
def load_audio_file(uploaded_file, label="1"):
    TMP_DIR = Path(tempfile.gettempdir())
    tmp_path = TMP_DIR / f"_input_{label}.wav"
    tmp_path.write_bytes(uploaded_file.read())
    y_raw, sr_raw = librosa.load(str(tmp_path), sr=None, mono=True)

    if sr_raw != target_sr:
        from math import gcd
        g = gcd(sr_raw, target_sr)
        y = resample_poly(y_raw, target_sr // g, sr_raw // g)
        sr = target_sr
    else:
        y = y_raw
        sr = sr_raw
    return y.astype(float), sr

# ---- 前処理関数 ----
def preprocess_audio(y, sr):
    t = np.arange(len(y)) / sr
    x_proc = y.copy()
    if use_notch:
        x_proc = apply_notch(x_proc, sr, freq=float(notch_freq), q=float(notch_q))
    x_proc = apply_bandpass(x_proc, sr, bp_low, bp_high, order=bp_order)
    return x_proc, t

# ---- 共通解析関数 ----
def analyze_audio(x_proc, sr, label="ファイル1"):
    results = {}
    st.markdown(f"## 🔍 {label} の解析結果")

    # ---------- 時間波形 ----------
    fig, ax = plt.subplots(figsize=(11,3))
    t = np.arange(len(x_proc)) / sr
    ax.plot(t, x_proc, lw=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig); plt.close(fig)

    # ---------- HLPR ----------
    hlpr, high_peak, low_peak = calculate_hlpr(x_proc, sr)
    st.metric(f"{label} HLPR値（Hilbert包絡）", f"{hlpr:.3f}")
    results["HLPR_hilbert"] = hlpr

    # ---------- FFT HLPR ----------
    hlpr_fft, freqs, spectrum, lpk, hpk = calculate_hlpr_fft(x_proc, sr)
    fig_fft, ax_fft = plt.subplots(figsize=(10, 4))
    ax_fft.plot(freqs, spectrum, lw=0.8)
    ax_fft.set_xlim(0, 1000)
    ax_fft.axvspan(100, 250, color='blue', alpha=0.1, label="Low Band")
    ax_fft.axvspan(500, 700, color='red', alpha=0.1, label="High Band")
    ax_fft.set_title(f"{label} - FFT HLPR = {hlpr_fft:.3f} (H / L = {hpk:.3f} / {lpk:.3f})")
    ax_fft.legend()
    st.pyplot(fig_fft)
    plt.close(fig_fft)
    results["HLPR_fft"] = hlpr_fft
    results["hilbert_high_peak"] = high_peak
    results["hilbert_low_peak"] = low_peak
    results["fft_high_peak"] = hpk
    results["fft_low_peak"] = lpk

    # ---------- STFT Linear ----------
    st.subheader(f"{label} - STFTスペクトログラム（Linear）")
    F_stft, TT_stft, S_stft = compute_stft(x_proc, sr)
    fig_stft, ax_stft = plt.subplots(figsize=(11, 3.6))
    pcm = ax_stft.pcolormesh(
        TT_stft, F_stft, S_stft,
        shading="auto", cmap="plasma",
        vmin=0.0005, vmax=0.015
    )
    ax_stft.set_ylim(0, min(600, sr//2))
    ax_stft.set_xlabel("Time [s]")
    ax_stft.set_ylabel("Frequency [Hz]")
    ax_stft.set_title(f"{label} - STFT Spectrogram (Linear)")
    cb = fig_stft.colorbar(pcm, ax=ax_stft)
    cb.set_label("Amplitude")
    st.pyplot(fig_stft)
    plt.close(fig_stft)

    # ---------- STFT Log ----------
    st.subheader(f"{label} - STFTスペクトログラム（Logスケール）")
    explain_button("Logスケールとは？", "周波数軸を対数表示することで広範囲の特性を見やすくし、高周波の異常も検出しやすくなります。")
    S_db = 10 * np.log10(S_stft + 1e-6)
    fig_log, ax_log = plt.subplots(figsize=(11, 3.8))
    pcm2 = ax_log.pcolormesh(TT_stft, F_stft, S_db, shading="auto", cmap="jet")
    ax_log.set_yscale("log")
    ax_log.set_ylim(20, sr//2)
    ax_log.set_xlabel("Time [s]")
    ax_log.set_ylabel("Frequency [Hz] (log scale)")
    ax_log.set_title(f"{label} - STFT Spectrogram (Log Power)")
    cb2 = fig_log.colorbar(pcm2, ax=ax_log)
    cb2.set_label("Power [dB]")
    st.pyplot(fig_log)
    plt.close(fig_log)

    results["label"] = label
    results["x_proc"] = x_proc
    results["sr"] = sr
    
    return results

# ==== 解析実行と比較結果表示 ====
results = []

# 音声ファイル1を読み込み＆前処理
y1, sr1 = load_audio_file(up1, label="1")
x1, _ = preprocess_audio(y1, sr1)

# 音声ファイル2があるかどうかで処理分岐
if up2:
    # 音声ファイル2も読み込み＆前処理
    y2, sr2 = load_audio_file(up2, label="2")
    x2, _ = preprocess_audio(y2, sr2)

    # 横並びレイアウト（2カラム）
    col1, col2 = st.columns(2)

    with col1:
        res1 = analyze_audio(x1, sr1, label="音声ファイル1")
        results.append(res1)

    with col2:
        res2 = analyze_audio(x2, sr2, label="音声ファイル2")
        results.append(res2)

    # 差分表示（中央に）
    st.subheader("HLPR 差分比較")
    st.metric("HLPR Hilbert 差", f"{res1['HLPR_hilbert'] - res2['HLPR_hilbert']:.3f}")
    st.metric("HLPR FFT 差", f"{res1['HLPR_fft'] - res2['HLPR_fft']:.3f}")

else:
    # ファイル1だけ → 通常表示
    res1 = analyze_audio(x1, sr1, label="音声ファイル1")
    results.append(res1)

# ---- シャント機能評価 入力フォーム ----
st.subheader("シャント評価パラメータの入力")

col1, col2 = st.columns(2)
with col1:
    fv = st.number_input("FV（血流量 mL/min）", min_value=0.0, value=0.0)
    psv = st.number_input("PSV（収縮期最大流速 cm/s）", min_value=0.0, value=0.0)
    ri = st.number_input("RI（抵抗指数）", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
with col2:
    tav = st.number_input("TAV（平均流速 cm/s）", min_value=0.0, value=0.0)
    edv = st.number_input("EDV（拡張期最小流速 cm/s）", min_value=0.0, value=0.0)
    pi = st.number_input("PI（脈波指数）", min_value=0.0, value=0.0)

st.subheader("備考欄（狭窄の有無）")

stenosis_flag = st.radio("狭窄の有無を教えてください", ["いいえ", "はい"], horizontal=True)

if stenosis_flag == "はい":
    stenosis_location = st.text_input("狭窄部位を入力してください")
else:
    stenosis_location = ""
    
# ------- 特徴量抽出はこの位置で -------
for res in results:
    label = res["label"]
    x_proc = res["x_proc"]
    sr = res["sr"]
    hlpr = res["HLPR_hilbert"]
    hlpr_fft = res["HLPR_fft"]

    spec_cent = librosa.feature.spectral_centroid(y=x_proc, sr=sr)[0]
    spec_bw = librosa.feature.spectral_bandwidth(y=x_proc, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=x_proc, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=x_proc)[0]
    rms = librosa.feature.rms(y=x_proc)[0]
    sflat = librosa.feature.spectral_flatness(y=x_proc)[0]

    feat = {
        "label": label,
        "shunt_type": shunt_type,
        "sex": sex,
        "site": site,
        "site_comment": site_comment,
        "mean_centroid_Hz": float(np.mean(spec_cent)),
        "mean_bandwidth_Hz": float(np.mean(spec_bw)),
        "median_rolloff_Hz": float(np.median(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "rms_energy": float(np.mean(rms)),
        "spectral_flatness": float(np.mean(sflat)),
        "HLPR_hilbert": float(hlpr),
        "HLPR_fft": float(hlpr_fft),
        "FV_mL_min": fv,
        "TAV_cm_s": tav,
        "PSV_cm_s": psv,
        "EDV_cm_s": edv,
        "RI": ri,
        "PI": pi,
        "stenosis_flag": stenosis_flag,
        "stenosis_location": stenosis_location
    }

    res.update(feat)  # 結果に追加

if export_csv and results:
    df_csv = pd.DataFrame(results)

    st.subheader("簡易スペクトル特徴量（+HLPR + 血流指標）")
    explain_button("各特徴量とは？（シャント評価）", 
    """
- **mean_centroid_Hz**（スペクトル重心）  
　音の平均的な周波数位置。高周波が多くなると高値に。血流の乱れ（渦流）で上昇する傾向。

- **mean_bandwidth_Hz**（スペクトル帯域幅）  
　音の広がりを表す。異常血流では広帯域化し、値が増加する可能性あり。

- **median_rolloff_Hz**（スペクトルロールオフ）  
　全エネルギーの85%を含む周波数。シャントが狭窄すると高周波成分が増え、ロールオフも上がりやすい。

- **zcr_mean**（ゼロクロッシング率）  
　音の変化の激しさ（波形が0を通過する回数）。高ZCRは乱流ノイズを示唆する。

- **rms_energy**（音圧エネルギー）  
　音の大きさを表す。流速の増加や血流障害でエネルギーが変化。

- **spectral_flatness**（スペクトル平坦度）  
　ノイズ的（フラット）かトーン的（鋭いピーク）か。血流の乱れでノイズ傾向が増し、値が高くなる。

- **HLPR_hilbert**（包絡線ピーク比）  
　500–700Hz（高周波）と100–250Hz（低周波）のエネルギー比。**0.35以上で血流の渦や狭窄の可能性あり**。

- **HLPR_fft**（スペクトルピーク比）  
　FFTスペクトル上の高低ピークの比率。論文で用いられたHLPR定義に準拠。**高値は高OSI・異常血流の可能性を示す**。
    """)

    # 表示
    st.dataframe(df_csv, use_container_width=True)

    # ファイル名の自動生成
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"shunt_comparison_results_{timestamp}.csv"

    # CSVダウンロードボタン
    st.download_button(
        label="CSVをダウンロード",
        data=df_csv.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv"
    )


