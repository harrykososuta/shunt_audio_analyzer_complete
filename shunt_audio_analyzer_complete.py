# -*- coding: utf-8 -*-
"""
シャント音 解析ビューア 完全版（Cloud安定版）
 - 入力: MP4/WAV など（MP4は音声抽出→解析）
 - 前処理: ノッチ(50/60Hz), バンドパス, リサンプリング
 - 可視化: 時間波形, STFTスペクトログラム(Linear/Log)
 - 解析: 帯域包絡(Hilbert), Welch PSD, 簡易特徴量, HLPR比
 - UI: 各解析に「説明」ボタン（expander表示）
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import streamlit as st
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
    f, t, Zxx = sp_stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, window=get_window(win, n_fft))
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

# ---- サイドバー ----
with st.sidebar:
    st.header("1) 音声の読み込み")
    up = st.file_uploader("WAV/MP3/FLAC/OGG/M4A", type=["wav","mp3","flac","ogg","m4a"])

    st.header("2) 前処理")
    target_sr = st.selectbox("解析サンプリング周波数", [2000, 4000, 8000, 16000], index=2)
    use_notch = st.checkbox("ノッチ除去（商用電源）", value=True)
    notch_freq = st.selectbox("ノッチ周波数", [50, 60], index=0)
    notch_q = st.slider("ノッチQ（鋭さ）", 10, 60, 30)
    bp_low = st.number_input("バンドパス下限 [Hz]", 0.0, 5000.0, 20.0, 10.0)
    bp_high = st.number_input("バンドパス上限 [Hz]", 50.0, 20000.0, 1200.0, 50.0)
    bp_order = st.slider("バンドパス次数", 2, 8, 4)

    st.header("3) 出力")
    export_csv = st.checkbox("CSV出力（スペクトル特徴量）", value=True)

# ---- メイン ----
st.title("シャント音 解析ビューア（STFT/PSD/包絡/HLPR）")
if up is None:
    st.info("左のサイドバーから音声ファイルをアップロードしてください。")
    st.stop()

TMP_DIR = Path(tempfile.gettempdir())
tmp_input = TMP_DIR / ("_input_" + Path(up.name).name)
tmp_input.write_bytes(up.read())

def load_audio(p: Path):
    y, sr = librosa.load(str(p), sr=None, mono=True)
    return y.astype(float), int(sr)

y_raw, sr_raw = load_audio(tmp_input)
if sr_raw != target_sr:
    from math import gcd
    g = gcd(sr_raw, target_sr)
    y = resample_poly(y_raw, target_sr//g, sr_raw//g)
    sr = target_sr
else:
    y = y_raw.copy()
    sr = sr_raw

t = np.arange(len(y))/sr
x_proc = y.copy()
if use_notch:
    x_proc = apply_notch(x_proc, sr, freq=float(notch_freq), q=float(notch_q))
x_proc = apply_bandpass(x_proc, sr, bp_low, bp_high, order=bp_order)

# ---- 時間波形 ----
st.subheader("時間波形")
fig, ax = plt.subplots(figsize=(11,3))
ax.plot(t, x_proc, lw=0.6)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
st.pyplot(fig); plt.close(fig)

# ---- HLPR ----
st.subheader("HLPR（高低周波ピーク比）")
explain_button("HLPRとは？", "高周波（例: 500–700Hz）と低周波（例: 100–250Hz）の包絡線ピーク比率を取り、シャント異常を検出する指標です。")
hlpr, high_peak, low_peak = calculate_hlpr(x_proc, sr)
st.metric("HLPR値", f"{hlpr:.3f}")
st.caption(f"High peak: {high_peak:.3f}, Low peak: {low_peak:.3f}")
if hlpr >= 0.35:
    st.error("⚠️ HLPRが0.35以上 → シャントトラブルの可能性あり")
else:
    st.success("HLPRは正常範囲内です")

# ---- Welch PSD ----
st.subheader("パワースペクトル密度（Welch法）")
explain_button("PSDとは？", "周波数成分のエネルギー分布（Power Spectral Density）を示します。異常があると特定の周波数が強調されます。")
ff, pxx = compute_psd_welch(x_proc, sr)
fig_psd, ax_psd = plt.subplots(figsize=(11,3))
ax_psd.semilogy(ff, pxx)
ax_psd.set_xlabel("Frequency [Hz]"); ax_psd.set_ylabel("PSD")
st.pyplot(fig_psd); plt.close(fig_psd)

# ---- STFT Linear ----
st.subheader("STFTスペクトログラム（Linear）")
explain_button("STFTとは？の説明", "時間-周波数分析の一種。Linearは低周波の解析に向いています。")
F_stft, TT_stft, S_stft = compute_stft(x_proc, sr)
fig_stft, ax_stft = plt.subplots(figsize=(11,3.5))
ax_stft.pcolormesh(TT_stft, F_stft, S_stft, shading="auto")
ax_stft.set_ylim(0, 600)
ax_stft.set_xlabel("Time [s]"); ax_stft.set_ylabel("Frequency [Hz]")
st.pyplot(fig_stft); plt.close(fig_stft)

# ---- STFT Log ----
st.subheader("STFTスペクトログラム（Logスケール）")
explain_button("Logスケールとは？", "周波数軸を対数表示することで広範囲の特性を見やすくし、高周波の異常も検出しやすくなります。")
fig_stft_log, ax_stft_log = plt.subplots(figsize=(11, 3.5))
im2 = ax_stft_log.pcolormesh(TT_stft, F_stft, S_stft, shading="auto")
ax_stft_log.set_yscale("log")
ax_stft_log.set_ylim(max(10, min(F_stft)), min(600, max(F_stft)))
ax_stft_log.set_xlabel("Time [s]")
ax_stft_log.set_ylabel("Frequency [Hz] (log)")
st.pyplot(fig_stft_log)
plt.close(fig_stft_log)
# ---- 特徴量 ----
spec_cent = librosa.feature.spectral_centroid(y=x_proc, sr=sr)[0]
spec_bw   = librosa.feature.spectral_bandwidth(y=x_proc, sr=sr)[0]
rolloff   = librosa.feature.spectral_rolloff(y=x_proc, sr=sr)[0]
zcr       = librosa.feature.zero_crossing_rate(y=x_proc)[0]
feat = {
    "mean_centroid_Hz": float(np.mean(spec_cent)),
    "mean_bandwidth_Hz": float(np.mean(spec_bw)),
    "median_rolloff_Hz": float(np.median(rolloff)),
    "zcr_mean": float(np.mean(zcr)),
    "HLPR": float(hlpr)
}
st.subheader("簡易スペクトル特徴量（+HLPR）")
st.dataframe(pd.DataFrame([feat]), use_container_width=True)
if export_csv:
    st.download_button("CSVダウンロード", data=pd.DataFrame([feat]).to_csv(index=False).encode("utf-8"), file_name="features_hlpr.csv")

