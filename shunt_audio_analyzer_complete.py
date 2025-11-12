# -*- coding: utf-8 -*-
"""
シャント音 解析ビューア 完全版（Cloud安定版）
 - 入力: MP4/WAV など（MP4は音声抽出→解析）
 - 前処理: ノッチ(50/60Hz), バンドパス, リサンプリング
 - 可視化: 時間波形, STFTスペクトログラム(縦軸狭め可), CWTスカログラム(帯域エネルギーCSV)
 - 解析: 帯域包絡(Hilbert), Welch PSD, 簡易特徴量
 - UI: 各解析に「説明」ボタン（expander のみに変更）
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

# ---- ページ設定 ----
st.set_page_config(page_title="Shunt Sound Analyzer - 完全版", layout="wide")

# ---- UI小道具（popover削除→expanderのみ） ----
def explain_button(title: str, body_md: str):
    with st.expander(f"ℹ️ {title} の説明"):
        st.markdown(body_md)

# ---- DSP utilities ----
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
# =============================================================================
# サイドバー
# =============================================================================
with st.sidebar:
    st.header("1) 音声の読み込み")
    up = st.file_uploader("WAV/MP3/FLAC/OGG/M4A", type=["wav","mp3","flac","ogg","m4a"])
    st.caption("MP4は内部で音声抽出→WAVに変換して解析します。")

    st.header("2) 前処理")
    target_sr = st.selectbox("解析サンプリング周波数", [2000, 4000, 8000, 16000], index=2)
    use_notch = st.checkbox("ノッチ除去（商用電源）", value=True)
    notch_freq = st.selectbox("ノッチ周波数", [50, 60], index=0)
    notch_q = st.slider("ノッチQ（鋭さ）", 10, 60, 30)
    bp_low = st.number_input("バンドパス下限 [Hz]", 0.0, 5000.0, 20.0, 10.0)
    bp_high = st.number_input("バンドパス上限 [Hz]", 50.0, 20000.0, 1200.0, 50.0)
    bp_order = st.slider("バンドパス次数", 2, 8, 4)

    st.header("3) STFTパラメータ")
    n_fft = st.selectbox("n_fft（窓長）", [512, 1024, 2048, 4096], index=2)
    hop = st.selectbox("hop_length", [64, 128, 256, 512], index=2)
    win = st.selectbox("窓関数", ["hann", "hamming", "blackman"], index=0)
    st.markdown("**縦軸レンジ（Hz）**")
    stft_fmin = st.number_input("表示下限", 0.0, 5000.0, 0.0, 10.0)
    stft_fmax = st.number_input("表示上限", 50.0, 20000.0, 1200.0, 50.0)

    st.header("4) CWTパラメータ")
    wavelet = st.selectbox("ウェーブレット", ["morl", "cmor1.5-1.0", "cmor1.5-0.5", "mexh"], index=0)
    fmin = st.number_input("CWT下限周波数 [Hz]", 10.0, 5000.0, 30.0, 10.0)
    fmax = st.number_input("CWT上限周波数 [Hz]", 20.0, 20000.0, 1200.0, 10.0)
    n_freqs = st.slider("周波数分割数", 32, 512, 128)  # 初期値やや軽め

    st.header("5) 帯域（包絡 & CWTエネルギー集計）")
    default_bands = [(150,300),(600,1200)]
    bands_text = st.text_area("例: 150-300,600-1200",
                              value=",".join([f"{a}-{b}" for (a,b) in default_bands]))
    def parse_bands(txt):
        out=[]
        for tok in txt.split(","):
            tok = tok.strip()
            if "-" in tok:
                a,b = tok.split("-")
                try: out.append((float(a), float(b)))
                except: pass
        return out
    bands = parse_bands(bands_text)

    st.header("6) 出力")
    export_csv = st.checkbox("CSV出力（帯域要約・時系列）", value=True)

# =============================================================================
# メイン
# =============================================================================
st.title("シャント音 解析ビューア（STFT/CWT/帯域包絡/PSD）")

if up is None:
    st.info("左のサイドバーから音声/動画ファイルをアップロードしてください。")
    st.stop()

# ---- 一時ファイルは /tmp に保存（Cloud安全） ----
TMP_DIR = Path(tempfile.gettempdir())
tmp_input = TMP_DIR / ("_input_" + Path(up.name).name)
tmp_wav   = TMP_DIR / "_extracted_audio.wav"

# ---- 入力を保存 → 読み込み ----
raw = up.read()
tmp_input.write_bytes(raw)

def load_audio_from_tmp(p: Path):
    """tmp_input から y, sr を返す。MP4は moviepy → WAV 抽出。失敗時は librosa 直読みにフォールバック。"""
    suf = p.suffix.lower()
    if suf == ".mp4":
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(str(p))
            clip.write_audiofile(str(tmp_wav), fps=16000, nbytes=2, ffmpeg_params=["-ac","1"])
            clip.close()
            y, sr = sf.read(str(tmp_wav))
            if y.ndim == 2: y = y.mean(axis=1)
            return y.astype(float), 16000
        except Exception as e:
            st.warning(f"MP4からの抽出に失敗（{e}）。librosaで直接読みに切り替えます。")
            y, sr = librosa.load(str(p), sr=None, mono=True)
            return y.astype(float), int(sr)
    else:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        return y.astype(float), int(sr)

y_raw, sr_raw = load_audio_from_tmp(tmp_input)

# ---- リサンプリング（解析用SRへ）----
if sr_raw != target_sr:
    from math import gcd
    g = gcd(sr_raw, target_sr)
    y = resample_poly(y_raw, target_sr//g, sr_raw//g)
    sr = target_sr
else:
    y = y_raw.copy()
    sr = sr_raw

t = np.arange(len(y))/sr
duration = len(y)/sr

# ---- 前処理 ----
x_proc = y.copy()
if use_notch:
    try:
        x_proc = apply_notch(x_proc, sr, freq=float(notch_freq), q=float(notch_q))
    except Exception:
        st.warning("ノッチ除去に失敗しました（パラメータを確認）")
x_proc = apply_bandpass(x_proc, sr, bp_low, bp_high, order=bp_order)

# ---- 時間波形 ----
st.subheader("時間波形（原波形／前処理後）")
explain_button("時間波形",
"""**何の解析？**  
録音したシャント音の振幅を時間軸で表示します。

**何がわかる？**  
・全体のS/N、アーチファクト（触れ/衝撃音）  
・特定イベント（圧迫・姿勢変化・挙上試験）とのタイミング対応  
""")
fig, ax = plt.subplots(2,1,figsize=(11,4), constrained_layout=True)
ax[0].plot(t, y, lw=0.7); ax[0].set_title(f"Original (sr={sr} Hz, {duration:.2f}s)")
ax[1].plot(t, x_proc, lw=0.7); ax[1].set_title(f"Processed (Notch {notch_freq}Hz={use_notch}, BP {bp_low}-{bp_high} Hz)")
for a in ax: a.set_xlabel("Time [s]"); a.set_ylabel("Amp")
st.pyplot(fig); plt.close(fig)

# ---- 帯域包絡（Hilbert） ----
st.subheader("周波数帯域ごとの包絡（Hilbert）")
explain_button("帯域包絡（Hilbert）",
"""**何の解析？**  
指定帯域でバンドパス後、**ヒルベルト変換**で包絡（エンベロープ）を計算し、帯域エネルギーの時間変化を可視化します。

**何がわかる？**  
・狭窄/乱流で**高周波帯**の寄与が増えやすい  
・イベント（圧迫/挙上）前後の帯域エネルギー変化  
""")
env_df = pd.DataFrame({"time_s": t})
summary_rows = []
for (a,b) in bands:
    try:
        _, env = band_envelope(x_proc, sr, (a,b), order=bp_order)
        env_df[f"{int(a)}-{int(b)}Hz_env"] = env
        summary_rows.append({
            "band_Hz": f"{int(a)}-{int(b)}",
            "mean_env": float(np.mean(env)),
            "median_env": float(np.median(env)),
            "p95_env": float(np.percentile(env,95)),
            "max_env": float(np.max(env))
        })
    except Exception as e:
        st.warning(f"帯域 {a}-{b} Hz でエラー: {e}")

if len(env_df.columns) > 1:
    fig2, ax2 = plt.subplots(figsize=(11,3.6))
    for c in env_df.columns:
        if c.endswith("_env"):
            ax2.plot(env_df["time_s"], env_df[c], lw=0.8, label=c.replace("_env",""))
    ax2.set_xlim(0,duration); ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Envelope (a.u.)")
    ax2.legend(ncol=3, fontsize=8)
    st.pyplot(fig2); plt.close(fig2)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

# ---- STFT ----
st.subheader("STFTスペクトログラム（|X|）")
explain_button("STFT（短時間フーリエ変換）",
"""**何の解析？**  
短い窓でフーリエ変換を連続実行し、時間×周波数の強度分布（**スペクトログラム**）を描きます。

**何がわかる？**  
・時間変動する高周波/低周波成分  
・**縦軸（周波数レンジ）を狭める**ことで目的帯域（例: 0–1.2kHz）を高解像に観察可能  
**Tips:** n_fftを大きく→周波数分解能↑、hopを小さく→時間分解能↑  
""")
F_stft, TT_stft, S_stft = compute_stft(x_proc, sr, n_fft=n_fft, hop=hop, win=win)
fig3, ax3 = plt.subplots(figsize=(11,3.8))
im = ax3.pcolormesh(TT_stft, F_stft, S_stft, shading="auto")
ax3.set_ylim(max(0, stft_fmin), min(sr/2, stft_fmax))
ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Frequency [Hz]")
cb = fig3.colorbar(im, ax=ax3); cb.set_label("Amplitude")
st.pyplot(fig3); plt.close(fig3)

# ---- CWT + 帯域エネルギー ----
st.subheader("CWTスカログラム（連続ウェーブレット変換）＋帯域エネルギー")
explain_button("CWT（連続ウェーブレット変換）",
"""**何の解析？**  
Morlet などのウェーブレットを使い、時間×周波数の**スカログラム**を描きます。  
高周波帯では時間分解能↑、低周波帯では周波数分解能↑の特性。

**何がわかる？**  
・狭窄/乱流に起因する**高周波成分（例：600–1200 Hz）**の持続・ピーク  
・イベントの瞬間的変化の検出（STFTより敏感な場面あり）  
**出力:** 指定帯域の**CWT帯域エネルギー時系列／要約CSV**  
""")
fmax_eff = min(fmax, sr/2 - 1)
freqs = np.geomspace(max(1.0, fmin), fmax_eff, num=int(n_freqs))
scales = freqs_to_scales(freqs, sr, wavelet_name=wavelet)
try:
    coef, _ = pywt.cwt(x_proc, scales, wavelet, sampling_period=1.0/sr)
    power = (np.abs(coef))**2
    fig4, ax4 = plt.subplots(figsize=(11,3.8))
    im2 = ax4.pcolormesh(t, freqs, power, shading="auto")
    ax4.set_yscale("log"); ax4.set_ylim(fmin, fmax_eff)
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Frequency [Hz]")
    ax4.set_title(f"CWT Scalogram | {wavelet}")
    cb2 = fig4.colorbar(im2, ax=ax4); cb2.set_label("Power")
    st.pyplot(fig4); plt.close(fig4)

    cwt_band_df = pd.DataFrame({"time_s": t})
    cwt_summary = []
    for (a,b) in bands:
        mask = (freqs >= max(a, fmin)) & (freqs <= min(b, fmax_eff))
        if np.any(mask):
            band_power = power[mask, :].mean(axis=0)  # 周波数方向に平均（和でもOK）
            col = f"CWT_{int(a)}-{int(b)}Hz"
            cwt_band_df[col] = band_power
            cwt_summary.append({
                "band_Hz": f"{int(a)}-{int(b)}",
                "mean_power": float(np.mean(band_power)),
                "median_power": float(np.median(band_power)),
                "p95_power": float(np.percentile(band_power,95)),
                "max_power": float(np.max(band_power))
            })

    if len(cwt_summary) > 0:
        fig5, ax5 = plt.subplots(figsize=(11,3.6))
        for c in cwt_band_df.columns:
            if c.startswith("CWT_"):
                ax5.plot(cwt_band_df["time_s"], cwt_band_df[c], lw=0.9, label=c.replace("CWT_",""))
        ax5.set_xlim(0, duration); ax5.set_xlabel("Time [s]"); ax5.set_ylabel("CWT band power (a.u.)")
        ax5.legend(ncol=3, fontsize=8)
        st.pyplot(fig5); plt.close(fig5)

        cwt_summ_df = pd.DataFrame(cwt_summary)
        st.dataframe(cwt_summ_df, use_container_width=True)

        if export_csv:
            st.download_button("CWT帯域エネルギー時系列CSVをダウンロード",
                               data=cwt_band_df.to_csv(index=False).encode("utf-8"),
                               file_name="cwt_band_power_timeseries.csv", mime="text/csv")
            st.download_button("CWT帯域エネルギー要約CSVをダウンロード",
                               data=cwt_summ_df.to_csv(index=False).encode("utf-8"),
                               file_name="cwt_band_power_summary.csv", mime="text/csv")
except Exception as e:
    st.warning(f"CWT計算中にエラー: {e}. 解析長・周波数分割数を小さくして再試行してください。")

# ---- PSD（Welch） ----
st.subheader("パワースペクトル密度（Welch）")
explain_button("Welch PSD",
"""**何の解析？**  
信号を重なり窓で区切り平均化する**Welch法**で、周波数成分の強度（密度）を推定します。

**何がわかる？**  
・全体のスペクトル分布  
・中心周波数帯の偏りや高周波寄与の増加  
""")
ff, pxx = compute_psd_welch(x_proc, sr, nperseg=4096 if sr>=8000 else 2048, noverlap=1024)
fig6, ax6 = plt.subplots(figsize=(11,3.0))
ax6.semilogy(ff, pxx)
ax6.set_xlim(0, min(sr/2, max(stft_fmax, fmax_eff)))
ax6.set_xlabel("Frequency [Hz]"); ax6.set_ylabel("PSD")
st.pyplot(fig6); plt.close(fig6)
if export_csv:
    psd_df = pd.DataFrame({"freq_Hz": ff, "PSD": pxx})
    st.download_button("PSD（Welch）CSVをダウンロード",
                       data=psd_df.to_csv(index=False).encode("utf-8"),
                       file_name="psd_welch.csv", mime="text/csv")

# ---- 簡易スペクトル特徴量 ----
st.subheader("簡易スペクトル特徴量")
explain_button("特徴量（重心・帯域比など）",
"""**何の解析？**  
・スペクトル重心 / 帯域比などのシンプル指標を算出。  
**使い方**  
・経時比較／群間比較／ROC解析の説明変数として利用可能。  
""")
spec_cent = librosa.feature.spectral_centroid(y=x_proc, sr=sr)[0]
spec_bw   = librosa.feature.spectral_bandwidth(y=x_proc, sr=sr)[0]
rolloff   = librosa.feature.spectral_rolloff(y=x_proc, sr=sr, roll_percent=0.95)[0]
zcr       = librosa.feature.zero_crossing_rate(y=x_proc)[0]
feat = {
    "mean_centroid_Hz": float(np.mean(spec_cent)),
    "median_centroid_Hz": float(np.median(spec_cent)),
    "mean_bandwidth_Hz": float(np.mean(spec_bw)),
    "rolloff95_Hz_median": float(np.median(rolloff)),
    "zcr_mean": float(np.mean(zcr))
}
st.dataframe(pd.DataFrame([feat]), use_container_width=True)

st.success("解析完了。必要に応じて各CSVをダウンロードしてください。")
st.caption("ヒント：CWTはn_freqs（周波数分割）と解析長に比例して計算が重くなります。必要に応じて解析長の短縮・リサンプリングを活用してください。")





