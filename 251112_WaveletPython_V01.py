# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, io, os, json
import plotly.graph_objects as go
from ssqueezepy import ssq_cwt, cwt as ss_cwt, Wavelet as SSqWavelet
import pywt
from pycwt import wavelet as pycwt_wavelet
import base64

st.set_page_config(layout="wide", page_title="Wavelet Analysis Lab")

# =====================================================
# ---- 1. Utility: Bead Segmentation  ------------------
# =====================================================
def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

# =====================================================
# ---- 2. Session State  -------------------------------
# =====================================================
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None
if "observations" not in st.session_state:
    st.session_state.observations = []

# =====================================================
# ---- 3. Step 1: Upload & Segmentation ---------------
# =====================================================
st.sidebar.header("Step 1 · Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        first_csv = [n for n in zip_ref.namelist() if n.endswith('.csv')][0]
        with zip_ref.open(first_csv) as f:
            sample_df = pd.read_csv(f)
    columns = sample_df.columns.tolist()
    seg_col = st.sidebar.selectbox("Column for Segmentation", columns)
    seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0)
    if st.sidebar.button("Bead Segmentation"):
        segmented_data = {}
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.csv'):
                    with zip_ref.open(file_name) as f:
                        df = pd.read_csv(f)
                    bead_ranges = segment_beads(df, seg_col, seg_thresh)
                    bead_dict = {}
                    for idx, (start, end) in enumerate(bead_ranges, start=1):
                        bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                    segmented_data[os.path.basename(file_name)] = bead_dict
        st.session_state.segmented_data = segmented_data
        st.session_state.observations.clear()
        st.success("✅ Bead segmentation complete!")

# =====================================================
# ---- 4. Step 2: Add Data  ----------------------------
# =====================================================
if st.session_state.segmented_data:
    st.sidebar.header("Step 2 · Add Data for Analysis")
    selected_csv = st.sidebar.selectbox("Select CSV File", list(st.session_state.segmented_data.keys()))
    bead_nums = list(st.session_state.segmented_data[selected_csv].keys())
    selected_bead = st.sidebar.selectbox("Select Bead Number", bead_nums)
    bead_df = st.session_state.segmented_data[selected_csv][selected_bead]
    signal_col = st.sidebar.selectbox("Select Signal Column", bead_df.columns.tolist())
    status = st.sidebar.selectbox("Weld Status", ["OK", "NOK"])

    if st.sidebar.button("➕ Add Data"):
        st.session_state.observations.append({
            "csv": selected_csv,
            "bead": selected_bead,
            "signal": signal_col,
            "status": status,
            "data": bead_df[signal_col].reset_index(drop=True)
        })

    if st.sidebar.button("🔄 Reset Analysis (keep segmentation)"):
        st.session_state.observations.clear()

# =====================================================
# ---- 5. Helper functions  ----------------------------
# =====================================================
def trim_to_shortest(signals):
    min_len = min(len(s) for s in signals)
    return [s[:min_len] for s in signals]

def normalize_mag(Z, mode, db_range):
    eps = 1e-12
    if mode == "Auto":
        return 20*np.log10(np.maximum(Z,eps)), None, None
    elif mode == "% of Max":
        Zp = 100*Z/np.max(Z)
        return Zp, 0, 100
    elif mode == "Fixed dB":
        Zdb = 20*np.log10(np.maximum(Z,eps))
        return Zdb, db_range[0], db_range[1]

def color(status): return "green" if status=="OK" else "red"

# =====================================================
# ---- 6. Wavelet Analysis Tabs ------------------------
# =====================================================
if st.session_state.observations:
    st.title("Wavelet Analysis Playground")
    tabs = st.tabs([
        "CWT (Scalogram)",
        "Synchrosqueezed CWT",
        "DWT Denoise + MRA",
        "MODWT + MRA",
        "Wavelet Packets",
        "Wavelet Coherence"
    ])

    # -------------------------------------------------
    #  CWT
    # -------------------------------------------------
    with tabs[0]:
        fs = st.number_input("Sampling Rate (Hz)", value=10000)
        norm_mode = st.selectbox("Normalization", ["Auto","% of Max","Fixed dB"])
        if norm_mode=="Fixed dB":
            dbrange = st.slider("dB Range", -120, 0, (-80,-20))
        else:
            dbrange = (-80,-20)
    
        fig = go.Figure()
        for obs in st.session_state.observations:
            x = obs["data"].to_numpy()
            # --- Stable CWT computation ---
            Wx, scales = ss_cwt(x, wavelet=('morlet', {'mu':6}))
            # derive frequencies manually
            freqs = pywt.scale2frequency('morl', scales) * fs
            Z, zmin, zmax = normalize_mag(np.abs(Wx), norm_mode, dbrange)
            t = np.arange(len(x))/fs
            fig.add_trace(go.Heatmap(
                z=Z, x=t, y=freqs, colorscale="Viridis",
                colorbar=dict(title="Magnitude"),
                name=f"{obs['csv']} Bead {obs['bead']}"
            ))
    
        fig.update_layout(
            title="Continuous Wavelet Transform (CWT Scalogram)",
            xaxis_title="Time [s]",
            yaxis_title="Frequency [Hz]"
        )
        st.plotly_chart(fig, use_container_width=True)


    # -------------------------------------------------
    #  SSQ-CWT
    # -------------------------------------------------
    with tabs[1]:
        fs = st.number_input("Sampling Rate (Hz) (SSQ)", value=10000)
        norm_mode = st.selectbox("Normalization (SSQ)", ["Auto","% of Max","Fixed dB"])
        if norm_mode=="Fixed dB":
            dbrange = st.slider("dB Range (SSQ)", -120, 0, (-80,-20))
        else:
            dbrange = (-80,-20)
        fig = go.Figure()
        for obs in st.session_state.observations:
            x = obs["data"].to_numpy()
            Tx, freqs = ssq_cwt(x, fs=fs, wavelet=('morlet',{'mu':6}))[:2]
            Z, zmin, zmax = normalize_mag(np.abs(Tx), norm_mode, dbrange)
            t = np.arange(len(x))/fs
            fig.add_trace(go.Heatmap(z=Z, x=t, y=freqs,
                                     colorscale="Viridis",
                                     name=f"{obs['csv']} Bead {obs['bead']}"))
        fig.update_layout(title="Synchrosqueezed CWT",
                          xaxis_title="Time [s]", yaxis_title="Frequency [Hz]")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    #  DWT Denoise + MRA
    # -------------------------------------------------
    with tabs[2]:
        wave = "db4"
        lev = 5
        for obs in st.session_state.observations:
            x = obs["data"].to_numpy()
            coeffs = pywt.wavedec(x, wavelet=wave, level=lev)
            sigma = np.median(np.abs(coeffs[-1]))/0.6745
            uth = sigma*np.sqrt(2*np.log(len(x)))
            coeffs[1:] = [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
            x_dn = pywt.waverec(coeffs, wavelet=wave)
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=x, mode="lines", name="Original", line=dict(color="gray")))
            fig.add_trace(go.Scatter(y=x_dn, mode="lines", name="Denoised", line=dict(color=color(obs["status"]))))
            st.subheader(f"{obs['csv']} Bead {obs['bead']} ({obs['status']})")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    #  MODWT + MRA
    # -------------------------------------------------
    with tabs[3]:
        for obs in st.session_state.observations:
            x = obs["data"].to_numpy()
            coeffs = pywt.modwt(x, wavelet="db4", level=6)
            mra = pywt.modwt_mra(coeffs)
            fig = go.Figure()
            for i, comp in enumerate(mra, 1):
                fig.add_trace(go.Scatter(y=comp, mode="lines", name=f"Level {i}"))
            st.subheader(f"{obs['csv']} Bead {obs['bead']} ({obs['status']})")
            fig.update_layout(title="MODWT MRA Components")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    #  Wavelet Packets
    # -------------------------------------------------
    with tabs[4]:
        for obs in st.session_state.observations:
            x = obs["data"].to_numpy()
            wp = pywt.WaveletPacket(x, wavelet='db4', mode='symmetric', maxlevel=4)
            nodes = [n.path for n in wp.get_level(4, 'natural')]
            energies = [np.sum(np.square(wp[n].data)) for n in nodes]
            fig = go.Figure(go.Bar(x=nodes, y=energies, marker_color=color(obs["status"])))
            fig.update_layout(title=f"Wavelet Packet Energies — {obs['csv']} Bead {obs['bead']}",
                              xaxis_title="Node Path", yaxis_title="Energy")
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    #  Wavelet Coherence
    # -------------------------------------------------
    with tabs[5]:
        if len(st.session_state.observations) < 2:
            st.info("Add at least two signals to compare for coherence.")
        else:
            fs = st.number_input("Sampling Rate (Hz) (Coherence)", value=10000)
            a, b = st.selectbox("Signal A", range(len(st.session_state.observations))), \
                   st.selectbox("Signal B", range(len(st.session_state.observations)))
            if a != b:
                obs_a = st.session_state.observations[a]
                obs_b = st.session_state.observations[b]
                x, y = trim_to_shortest([obs_a["data"], obs_b["data"]])
                dt = 1/fs
                mother = pycwt_wavelet.Morlet(6)
                W12, cross, coi, freq, signif, rsq, period, scale, wcoh, phase = \
                    pycwt_wavelet.wct(x, y, dt, mother)
                t = np.arange(len(x))/fs
                fig = go.Figure()
                fig.add_trace(go.Heatmap(z=rsq, x=t, y=1/period, colorscale="Turbo"))
                fig.update_layout(title="Wavelet Coherence",
                                  xaxis_title="Time [s]", yaxis_title="Frequency [Hz]")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Select two different signals.")
