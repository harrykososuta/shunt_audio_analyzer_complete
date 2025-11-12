# -*- coding: utf-8 -*-
"""
シャント音 解析ビューア 完全版（Cloud安定版）
 - 入力: MP4/WAV など（MP4は音声抽出→解析）
 - 前処理: ノッチ(50/60Hz), バンドパス, リサンプリング
 - 可視化: 時間波形, STFTスペクトログラム(Linear & Log)
 - 解析: 帯域包絡(Hilbert), Welch PSD, HLPR比
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

def explain_button(title: str, body_md: str):
    with st.expander(f"🛈 {title} の説明"):
        st.markdown(body_md)

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

st.title("シャント音 解析ビューア（STFT/PSD/HLPR）")
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

# 時間波形
st.subheader("時間波形")
explain_button("時間波形", "音声信号の全体の時間的な強弱やノイズを確認する基本的な可視化です。")
fig, ax = plt.subplots(figsize=(11,3))
ax.plot(t, x_proc, lw=0.6)
ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
st.pyplot(fig); plt.close(fig)

# HLPR
st.subheader("HLPR（高低周波ピーク比）")
explain_button("HLPRとは？", "高周波帯域（500–700Hz）と低周波帯域（100–250Hz）のピーク振幅を比率化。高い値は血流の乱れを示唆します。")
hlpr, high_peak, low_peak = calculate_hlpr(x_proc, sr)
st.metric("HLPR値", f"{hlpr:.3f}")
st.caption(f"高周波: 500–700 Hz / 低周波: 100–250 Hz")
st.caption("※ HLPR = 高周波ピーク ÷ 低周波ピーク")
if hlpr >= 0.35:
    st.error("⚠️ HLPRが0.35以上 → シャントトラブルの可能性があります")
else:
    st.success("HLPRは正常範囲内です")

# PSD
st.subheader("Welch パワースペクトル密度（PSD）")
explain_button("Welch PSDとは？", "時間信号を周波数成分に分解し、各周波数のエネルギー分布を表示します。ピークがある周波数帯が強いです。")
ff, pxx = compute_psd_welch(x_proc, sr)
fig_psd, ax_psd = plt.subplots(figsize=(11,3))
ax_psd.semilogy(ff, pxx)
ax_psd.set_xlabel("Frequency [Hz]")
ax_psd.set_ylabel("PSD")
st.pyplot(fig_psd); plt.close(fig_psd)

# STFT
F_stft, TT_stft, S_stft = compute_stft(x_proc, sr)

st.subheader("STFTスペクトログラム（Linear）")
explain_button("STFTとは？", "音の時間変化と周波数成分を同時に可視化。横軸が時間、縦軸が周波数、色が強度を表します。")
fig_stft_lin, ax_stft_lin = plt.subplots(figsize=(11, 3.5))
im = ax_stft_lin.pcolormesh(TT_stft, F_stft, S_stft, shading="auto")
ax_stft_lin.set_ylim(0, 600)
ax_stft_lin.set_xlabel("Time [s]")
ax_stft_lin.set_ylabel("Frequency [Hz]")
st.pyplot(fig_stft_lin)
plt.close(fig_stft_lin)

st.subheader("STFTスペクトログラム（Logスケール）")
fig_stft_log, ax_stft_log = plt.subplots(figsize=(11, 3.5))
im2 = ax_stft_log.pcolormesh(TT_stft, F_stft, S_stft, shading="auto")
ax_stft_log.set_yscale("log")
ax_stft_log.set_ylim(max(10, min(F_stft)), min(600, max(F_stft)))
ax_stft_log.set_xlabel("Time [s]")
ax_stft_log.set_ylabel("Frequency [Hz] (log)")
st.pyplot(fig_stft_log)
plt.close(fig_stft_log)

# 特徴量
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
