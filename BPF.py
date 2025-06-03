import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import convolve, freqz 

st.set_page_config(layout="wide")

# === SIDEBAR ===
st.sidebar.title("Spesifikasi Filter FIR (Windowing)")

audio_file = st.sidebar.file_uploader("🎵 Upload Audio", type=["wav", "mp3", "ogg", "flac"]) 
if audio_file:
    y_raw, sr_raw = librosa.load(audio_file, sr=None)
    dur = librosa.get_duration(y=y_raw, sr=sr_raw)
    st.sidebar.markdown(f"- Durasi: `{dur:.2f}` detik")
    st.sidebar.markdown(f"- Sample Rate: `{sr_raw}` Hz")

    D = np.abs(librosa.stft(y_raw))
    freqs = librosa.fft_frequencies(sr=sr_raw)
    power = np.mean(D, axis=1)
    threshold = np.max(power) * 0.01
    sig_freqs = freqs[power > threshold]

    if len(sig_freqs) > 0:
        max_freq = np.max(sig_freqs)
        st.sidebar.success(f"Frek. signifikan max: `{max_freq:.1f}` Hz")
        st.sidebar.info(f"Min Sampling Rate: `{int(np.ceil(2 * max_freq))} Hz`")
    else:
        st.sidebar.warning("Tidak ada frekuensi signifikan.")

    max_fs_khz = int(sr_raw / 1000)
    fs = st.sidebar.slider("Sampling Rate (kHz)", 4, max_fs_khz, max_fs_khz) * 1000
    delta_p = st.sidebar.number_input("𝛿 Passband", value=0.05)
    delta_s = st.sidebar.number_input("𝛿 Stopband", value=0.001)
    f1 = st.sidebar.slider("Frekuensi batas bawah Passband (kHz)", 0.1, fs / 2000 - 0.2, 1.0) * 1000
    f2 = st.sidebar.slider("Frekuensi batas atas Passband (kHz)", f1 / 1000 + 0.1, fs / 2000 - 0.1, 2.0) * 1000
    trans_bw = st.sidebar.slider("Lebar Transisi (Hz)", 10, 2000, 100)

    A_s = -20 * np.log10(delta_s)
    A_p = 20 * np.log10(1 + delta_p)
    st.sidebar.markdown(f"Atenuasi Stopband: `{A_s:.2f} dB`")
    st.sidebar.markdown(f"Ripple Passband: `{A_p:.2f} dB`")

    window_type = st.sidebar.selectbox("Window", ["Hamming", "Hanning", "Blackman", "Rectangular"])
    window_eq = {
        "Hamming": r"w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{N-1}\right)",
        "Hanning": r"w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)",
        "Blackman": r"w[n] = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right) + 0.08 \cos\left(\frac{4\pi n}{N-1}\right)",
        "Rectangular": r"w[n] = 1"
    }
    delta_f = trans_bw / fs
    N = int(np.ceil(3.3 / delta_f))
    if N % 2 == 0: N += 1
    st.sidebar.markdown(f"Panjang Filter (N): `{N}`")

    # === FUNGSI ===
    def get_window(win_type, N):
        n = np.arange(N)
        if win_type == 'Hamming':
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
        elif win_type == 'Hanning':
            return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        elif win_type == 'Blackman':
            return 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        else:
            return np.ones(N)

    def ideal_bandpass(fl, fh, fs, N):
        fc1, fc2 = fl / fs, fh / fs
        n = np.arange(N)
        M = (N - 1) / 2
        return (2 * fc2 * np.sinc(2 * fc2 * (n - M))) - (2 * fc1 * np.sinc(2 * fc1 * (n - M)))

    hD = ideal_bandpass(f1, f2, fs, N)
    wN = get_window(window_type, N)
    hN = hD * wN

    # === HALAMAN ===
    st.header("Aplikasi FIR Band-Pass Filter pada Audio File 🎶")
    section = st.radio("📂 Pilih Halaman", ["Output Audio & Grafik", "Perhitungan Filter"])

    if section == "Output Audio & Grafik":
        st.header("Output Audio dan Grafik")
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=int(fs))
        y_filtered = convolve(y, hN, mode='same') #kalo mau hasil filternya full audio, bisa pake Fungsi "convolve(..., mode='full')", tapi lebih aman same karena Tidak menggeser waktu, Tidak menambah delay, dan Aman untuk tumpang-tindih (overlay)

        # Inisialisasi variabel penting
        t = np.linspace(0, len(y)/fs, len(y))
        w, H = freqz(hN, worN=2048, fs=fs)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("🎵 **Audio Asli**")
            st.audio(audio_file)
        with col2:
            sf.write("filtered.wav", y_filtered, int(fs))
            st.markdown("🎵 **Audio Filtered**")
            st.audio("filtered.wav")
            with open("filtered.wav", "rb") as f:
                st.download_button("⬇️ Download Audio Filtered", f.read(), "filtered_output.wav", "audio/wav")


        # === WAVEFORM ===
        st.subheader("Waveform")
        st.markdown("**a. Waveform Audio Asli**")
        plt.figure(figsize=(10, 3))
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform - Original")
        st.pyplot(plt)

        st.markdown("**b. Waveform Audio Filtered**")
        plt.figure(figsize=(10, 3))
        plt.plot(t, y_filtered, color='orange')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform - Filtered")
        st.pyplot(plt)

        st.markdown("**c. Waveform Comparison (Overlay)**")
        plt.figure(figsize=(10, 3))
        plt.plot(t, y, label='Original')
        plt.plot(t, y_filtered, label='Filtered', alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Waveform Comparison")
        plt.legend()
        st.pyplot(plt)

        # === FREQUENCY RESPONSE ===
        st.subheader("Frequency Response")
        st.markdown("**a. Magnitude Response**")
        plt.figure(figsize=(10, 3))
        plt.plot(w, 20 * np.log10(np.abs(H) + 1e-6))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Magnitude Response")
        st.pyplot(plt)

        st.markdown("**b. Phase Response**")
        plt.figure(figsize=(10, 3))
        plt.plot(w, np.unwrap(np.angle(H)))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (radian)")
        plt.title("Phase Response")
        st.pyplot(plt)

        st.markdown("**c. Frequency Spectrum Comparison**")
        def plot_spectrum(sig, fs, label):
            f = np.fft.rfftfreq(2048, 1/fs)
            sp = np.abs(np.fft.rfft(sig, 2048))
            plt.plot(f, 20*np.log10(sp + 1e-6), label=label)

        plt.figure(figsize=(10, 3))
        plot_spectrum(y, fs, 'Original')
        plot_spectrum(y_filtered, fs, 'Filtered')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Spectrum Comparison")
        plt.legend()
        st.pyplot(plt)

        # === SPECTROGRAM ===
        st.subheader(" Spectrogram")
        st.markdown("**a. Spectrogram Audio Asli**")
        plt.figure(figsize=(10, 3))
        plt.specgram(y, Fs=fs)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram - Original")
        st.pyplot(plt)

        st.markdown("**b. Spectrogram Audio Filtered**")
        plt.figure(figsize=(10, 3))
        plt.specgram(y_filtered, Fs=fs)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram - Filtered")
        st.pyplot(plt)

        st.markdown("**c. Spectrogram Comparison (Side-by-Side)**")
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].specgram(y, Fs=fs)
        axs[0].set_title("Spectrogram - Original", pad=15)
        axs[1].specgram(y_filtered, Fs=fs)
        axs[1].set_title("Spectrogram - Filtered", pad=15)
        for ax in axs:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        st.pyplot(fig)


    elif section == "Perhitungan Filter":
        st.header("Menghitung Koefisien Filter FIR")

        st.markdown("#### Rumus Ideal Impulse Response")
        st.latex(r"h_D[n] = 2f_2 \cdot sinc(2f_2(n-M)) - 2f_1 \cdot sinc(2f_1(n-M))")

        st.markdown("#### Rumus Window Function")
        st.latex(window_eq[window_type])
        st.subheader("Grafik Window Function")
        fig_w2, ax_w2 = plt.subplots(figsize=(6, 3))
        ax_w2.plot(wN, color='darkorange')
        ax_w2.set_title(f"{window_type} Window Function (w[n])")
        ax_w2.set_xlabel("n")
        ax_w2.set_ylabel("Amplitude")
        ax_w2.grid()
        st.pyplot(fig_w2)

        st.markdown("#### Rumus Koefisien Filter FIR")
        st.latex(r"h[n] = h_D[n] \cdot w[n]")

        st.subheader("Tabel Koefisien Filter")
        df = pd.DataFrame({"Index[n]": np.arange(N), "hD[n]": hD, "w[n]": wN, "h[n]": hN})
        st.dataframe(df)
        st.download_button("⬇️ Download Koefisien Filter (.csv)", df.to_csv(index=False).encode('utf-8'), "koefisien_filter.csv", "text/csv")

        st.subheader("Grafik Impulse Response")
        fig, ax = plt.subplots()
        ax.stem(np.arange(N), hN)
        ax.set_title("Impulse Response")
        ax.set_xlabel("n")
        ax.set_ylabel("h[n]")
        ax.grid()
        st.pyplot(fig)

        st.info("Perhitungan menggunakan metode windowing: menghitung hD[n], dikalikan dengan fungsi window w[n] untuk mendapatkan h[n].")

