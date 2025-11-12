# app.py (fixed)
import streamlit as st
import pandas as pd
import numpy as np
import zipfile, os, json

import plotly.graph_objects as go
import pywt
from ssqueezepy import cwt as ss_cwt, ssq_cwt
from ssqueezepy import Wavelet as SSqWavelet
from pycwt import wavelet as pycwt_wavelet

st.set_page_config(layout="wide", page_title="Wavelet Analysis Lab")

# ---------------------- Segmentation ----------------------
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

# ---------------------- Session ----------------------
if "segmented_data" not in st.session_state:
    st.session_state.segmented_data = None
if "observations" not in st.session_state:
    st.session_state.observations = []

# ---------------------- Sidebar Step 1 ----------------------
st.sidebar.header("Step 1: Upload & Segmentation")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type="zip", key="zip_upload")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        csvs = [n for n in zip_ref.namelist() if n.endswith('.csv')]
        if not csvs:
            st.sidebar.error("No CSV files in the ZIP.")
        else:
            with zip_ref.open(csvs[0]) as f:
                sample_df = pd.read_csv(f)
            columns = sample_df.columns.tolist()
            seg_col = st.sidebar.selectbox("Column for Segmentation", columns, key="seg_col")
            seg_thresh = st.sidebar.number_input("Segmentation Threshold", value=1.0, key="seg_th")
            if st.sidebar.button("Bead Segmentation", key="seg_btn"):
                segmented_data = {}
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref2:
                    for file_name in zip_ref2.namelist():
                        if file_name.endswith('.csv'):
                            with zip_ref2.open(file_name) as f:
                                df = pd.read_csv(f)
                            bead_ranges = segment_beads(df, seg_col, seg_thresh)
                            bead_dict = {}
                            for idx, (start, end) in enumerate(bead_ranges, start=1):
                                bead_dict[idx] = df.iloc[start:end+1].reset_index(drop=True)
                            segmented_data[os.path.basename(file_name)] = bead_dict
                st.session_state.segmented_data = segmented_data
                st.session_state.observations.clear()
                st.success("✅ Bead segmentation complete and locked!")

# ---------------------- Sidebar Step 2 ----------------------
if st.session_state.segmented_data:
    st.sidebar.header("Step 2: Add Data for Analysis")
    csv_keys = list(st.session_state.segmented_data.keys())
    selected_csv = st.sidebar.selectbox("Select CSV File", csv_keys, key="csv_sel")
    bead_nums = list(st.session_state.segmented_data[selected_csv].keys())
    selected_bead = st.sidebar.selectbox("Select Bead Number", bead_nums, key="bead_sel")
    bead_df = st.session_state.segmented_data[selected_csv][selected_bead]
    signal_col = st.sidebar.selectbox("Select Signal Column", bead_df.columns.tolist(), key="sig_col")
    status = st.sidebar.selectbox("Weld Status", ["OK", "NOK"], key="status_sel")

    if st.sidebar.button("➕ Add Data", key="add_btn"):
        st.session_state.observations.append({
            "csv": selected_csv,
            "bead": selected_bead,
            "signal": signal_col,
            "status": status,
            "data": bead_df[signal_col].reset_index(drop=True)
        })

    if st.sidebar.button("🔄 Reset Analysis (keep segmentation)", key="reset_btn"):
        st.session_state.observations.clear()

# ---------------------- Helpers ----------------------
def _color(status): return "green" if status == "OK" else "red"

def normalize_mag(Z, mode, db_range=None):
    eps = 1e-12
    Z = np.asarray(Z, dtype=float)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "Auto":
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        return Zdb, None, None
    elif mode == "% of Max":
        m = float(max(Z.max(), eps))
        Zp = 100.0 * (Z / m)
        return Zp, 0.0, 100.0
    elif mode == "Fixed dB":
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        zmin, zmax = db_range if db_range is not None else (-120.0, 0.0)
        return Zdb, float(zmin), float(zmax)
    else:
        Zdb = 20.0 * np.log10(np.maximum(Z, eps))
        return Zdb, None, None

def safe_heatmap(fig, Z_disp, t, freqs, zmin=None, zmax=None, colorscale="Viridis", name=""):
    Z_disp = np.nan_to_num(np.asarray(Z_disp, float), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(np.asarray(t, float), nan=0.0, posinf=0.0, neginf=0.0)
    freqs = np.nan_to_num(np.asarray(freqs, float), nan=0.0, posinf=0.0, neginf=0.0)
    fig.add_trace(go.Heatmap(
        z=Z_disp.tolist(), x=t.tolist(), y=freqs.tolist(),
        zmin=zmin, zmax=zmax, colorscale=colorscale, colorbar=dict(title="Magnitude"),
        name=name
    ))

def _cwt_freqs_from_scales(scales, fs):
    return pywt.scale2frequency('morl', np.asarray(scales, float)) * float(fs)

def _auto_trim_by_bead(observations):
    out = {}
    for obs in observations:
        out.setdefault(obs["bead"], []).append(obs)
    trimmed = {}
    for bead, obs_list in out.items():
        min_len = min(len(o["data"]) for o in obs_list)
        trimmed[bead] = [(o, o["data"].iloc[:min_len].reset_index(drop=True)) for o in obs_list]
    return trimmed

def _pairwise_trim(a: pd.Series, b: pd.Series):
    n = min(len(a), len(b))
    return a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)

def _compute_coi_curve(x, fs, morlet_omega0=6):
    dt = 1.0 / float(fs)
    mother = pycwt_wavelet.Morlet(morlet_omega0)
    x0 = np.asarray(x, float) - float(np.mean(x))
    W, scales, freqs, coi, *_ = pycwt_wavelet.cwt(x0, dt, mother)
    t = np.arange(len(x0)) * dt
    fcoi = 1.0 / np.maximum(np.asarray(coi, float), 1e-12)
    fcoi = np.nan_to_num(fcoi, nan=0.0, posinf=0.0, neginf=0.0)
    return t, fcoi

# ---------------------- Main ----------------------
if st.session_state.observations:
    st.title("Wavelet Analysis")
    tabs = st.tabs([
        "CWT (Scalogram)",
        "Synchrosqueezed CWT",
        "DWT Denoise + MRA",
        "MODWT + MRA",
        "Wavelet Packets",
        "Wavelet Coherence"
    ])

    # -------- CWT --------
    with tabs[0]:
        fs = st.number_input("Sampling Rate (Hz) — CWT", value=10000, min_value=1, key="fs_cwt")
        norm_mode = st.selectbox("Normalization — CWT", ["Auto", "% of Max", "Fixed dB"], key="norm_cwt")
        dbrange = st.slider("dB Range — CWT", -140, 0, (-100, -20), key="dbr_cwt") if norm_mode == "Fixed dB" else None
        show_coi = st.checkbox("Show COI (cone of influence)", value=True, key="coi_cwt")
        with st.expander("Advanced (CWT)", expanded=False):
            vpo = st.slider("Voices/Octave", 16, 64, 32, step=8, key="vpo_cwt")
            mu = st.slider("Morlet μ", 3, 12, 6, key="mu_cwt")

        fig_cwt = go.Figure()
        by_bead = _auto_trim_by_bead(st.session_state.observations)
        for bead, items in by_bead.items():
            for idx, (obs, s) in enumerate(items):
                x = s.to_numpy(float)
                Wx, scales = ss_cwt(x, wavelet=SSqWavelet(('morlet', {'mu': mu})), nv=vpo)
                freqs = _cwt_freqs_from_scales(scales, fs)
                Z = np.abs(Wx).astype(float)
                Z_disp, zmin, zmax = normalize_mag(Z, norm_mode, dbrange)
                t = np.arange(len(x), float) / float(fs)
                safe_heatmap(fig_cwt, Z_disp, t, freqs, zmin=zmin, zmax=zmax,
                             colorscale="Viridis",
                             name=f"{obs['csv']} · Bead {obs['bead']} ({obs['status']})")
                if show_coi:
                    try:
                        tcoi, fcoi = _compute_coi_curve(x, fs, morlet_omega0=6)
                        fig_cwt.add_trace(go.Scatter(x=tcoi.tolist(), y=fcoi.tolist(),
                                                     mode="lines",
                                                     line=dict(color="white", width=2, dash="dash"),
                                                     name=f"COI · {obs['csv']} · Bead {obs['bead']}"))
                    except Exception:
                        pass
        st.plotly_chart(fig_cwt, use_container_width=True, key="plt_cwt")

    # -------- SSQ-CWT --------
    with tabs[1]:
        fs = st.number_input("Sampling Rate (Hz) — SSQ", value=10000, min_value=1, key="fs_ssq")
        norm_mode = st.selectbox("Normalization — SSQ", ["Auto", "% of Max", "Fixed dB"], key="norm_ssq")
        dbrange = st.slider("dB Range — SSQ", -140, 0, (-100, -20), key="dbr_ssq") if norm_mode == "Fixed dB" else None
        show_coi = st.checkbox("Show COI — SSQ", value=True, key="coi_ssq")
        with st.expander("Advanced (SSQ)", expanded=False):
            vpo = st.slider("Voices/Octave — SSQ", 16, 64, 32, step=8, key="vpo_ssq")
            mu = st.slider("Morlet μ — SSQ", 3, 12, 6, key="mu_ssq")
            show_ridge = st.checkbox("Plot ridge frequency", value=False, key="ridge_ssq")

        fig_ssq = go.Figure()
        by_bead = _auto_trim_by_bead(st.session_state.observations)
        for bead, items in by_bead.items():
            for idx, (obs, s) in enumerate(items):
                x = s.to_numpy(float)
                Tx, ssq_freqs = ssq_cwt(x, fs=float(fs), wavelet=('morlet', {'mu': mu}), nv=vpo)[:2]
                Z = np.abs(Tx).astype(float)
                Z_disp, zmin, zmax = normalize_mag(Z, norm_mode, dbrange)
                t = np.arange(len(x), float) / float(fs)
                safe_heatmap(fig_ssq, Z_disp, t, ssq_freqs, zmin=zmin, zmax=zmax,
                             colorscale="Viridis",
                             name=f"{obs['csv']} · Bead {obs['bead']} ({obs['status']})")
                if show_ridge:
                    ridge_idx = np.argmax(Z, axis=0)
                    ridge_freq = ssq_freqs[ridge_idx]
                    fig_ssq.add_trace(go.Scatter(x=t.tolist(), y=ridge_freq.tolist(),
                                                 mode="lines", line=dict(color="white", width=2),
                                                 name=f"Ridge · {obs['csv']} · Bead {obs['bead']}"))
                if show_coi:
                    try:
                        tcoi, fcoi = _compute_coi_curve(x, fs, morlet_omega0=6)
                        fig_ssq.add_trace(go.Scatter(x=tcoi.tolist(), y=fcoi.tolist(),
                                                     mode="lines",
                                                     line=dict(color="white", width=2, dash="dash"),
                                                     name=f"COI · {obs['csv']} · Bead {obs['bead']}"))
                    except Exception:
                        pass
        st.plotly_chart(fig_ssq, use_container_width=True, key="plt_ssq")

    # -------- DWT Denoise + MRA --------
    with tabs[2]:
        with st.expander("Advanced (DWT)", expanded=False):
            wave = st.selectbox("Wavelet family", ["db4", "db6", "sym8", "coif3"], index=0, key="wave_dwt")
            maxlev_cap = st.slider("Max levels cap", 1, 10, 6, key="levcap_dwt")

        # Always render one chart per observation (unique keys)
        for i, obs in enumerate(st.session_state.observations):
            x = obs["data"].to_numpy(float)
            maxlev = min(maxlev_cap, pywt.dwt_max_level(len(x), pywt.Wavelet(wave).dec_len))
            coeffs = pywt.wavedec(x, wavelet=wave, level=maxlev, mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uth = sigma * np.sqrt(2 * np.log(len(x)))
            coeffs_th = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
            x_dn = pywt.waverec(coeffs_th, wavelet=wave, mode="symmetric")

            comps = []
            for j in range(1, len(coeffs)):
                ckeep = [np.zeros_like(c) for c in coeffs]
                ckeep[0] = np.zeros_like(coeffs[0])
                ckeep[j] = coeffs[j]
                comp = pywt.waverec(ckeep, wavelet=wave, mode="symmetric")
                comps.append(np.asarray(comp, float))

            fig_dwt = go.Figure()
            fig_dwt.add_trace(go.Scatter(y=x.tolist(), mode="lines", name="Original", line=dict(color="gray")))
            fig_dwt.add_trace(go.Scatter(y=x_dn.tolist(), mode="lines", name="Denoised", line=dict(color=_color(obs["status"]))))
            for j, c in enumerate(comps, 1):
                fig_dwt.add_trace(go.Scatter(y=c.tolist(), mode="lines", name=f"D{j}"))

            fig_dwt.update_layout(title=f"DWT Denoise + MRA — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                                  xaxis_title="Index", yaxis_title="Amplitude")
            st.plotly_chart(fig_dwt, use_container_width=True, key=f"plt_dwt_{i}")

    # -------- MODWT + MRA --------
    with tabs[3]:
        with st.expander("Advanced (MODWT)", expanded=False):
            wave = st.selectbox("Wavelet family (MODWT)", ["db4", "db6", "sym8", "coif3"], index=0, key="wave_modwt")
            levels = st.slider("Levels (MODWT)", 1, 10, 6, key="lev_modwt")

        for i, obs in enumerate(st.session_state.observations):
            x = obs["data"].to_numpy(float)
            coeffs = pywt.modwt(x, wavelet=wave, level=levels)
            mra = pywt.modwt_mra(coeffs)

            fig_modwt = go.Figure()
            for j, comp in enumerate(mra, 1):
                comp = np.asarray(comp, float)
                fig_modwt.add_trace(go.Scatter(y=comp.tolist(), mode="lines", name=f"L{j}"))
            fig_modwt.update_layout(title=f"MODWT MRA — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                                    xaxis_title="Index", yaxis_title="Component")
            st.plotly_chart(fig_modwt, use_container_width=True, key=f"plt_modwt_{i}")

    # -------- Wavelet Packets --------
    with tabs[4]:
        with st.expander("Advanced (Packets)", expanded=False):
            wave = st.selectbox("Wavelet family (packets)", ["db4", "db6", "sym8", "coif3"], index=0, key="wave_pkt")
            maxlevel = st.slider("Max level (packets)", 1, 8, 4, key="lev_pkt")
            criterion = st.selectbox("Best-basis criterion", ["entropy", "shannon", "sure", "logenergy"], index=0, key="crit_pkt")
        crit = "entropy" if criterion in ["entropy", "shannon"] else criterion

        for i, obs in enumerate(st.session_state.observations):
            x = obs["data"].to_numpy(float)
            wp = pywt.WaveletPacket(data=x, wavelet=wave, mode='symmetric', maxlevel=maxlevel)
            level_nodes = wp.get_best_level(decomp_level=maxlevel, criterion=crit)
            if not level_nodes:
                st.info(f"No packet nodes at level {maxlevel} for {obs['csv']} · Bead {obs['bead']}")
                continue
            nodes = [n.path for n in level_nodes]
            energies = [float(np.sum(np.square(np.asarray(n.data, float)))) for n in level_nodes]

            fig_pkt = go.Figure(go.Bar(x=nodes, y=energies, marker_color=_color(obs["status"])))
            fig_pkt.update_layout(title=f"Wavelet Packet Energies — {obs['csv']} · Bead {obs['bead']} ({obs['status']})",
                                  xaxis_title="Node", yaxis_title="Energy")
            st.plotly_chart(fig_pkt, use_container_width=True, key=f"plt_pkt_{i}")

    # -------- Wavelet Coherence --------
    with tabs[5]:
        if len(st.session_state.observations) < 2:
            st.info("Add at least two signals to compare for coherence.")
        else:
            fs = st.number_input("Sampling Rate (Hz) — Coherence", value=10000, min_value=1, key="fs_coh")
            idxs = list(range(len(st.session_state.observations)))
            def fmt(i):
                o = st.session_state.observations[i]
                return f"{o['csv']} · Bead {o['bead']} ({o['status']})"
            a = st.selectbox("Signal A", idxs, key="coh_a", format_func=fmt)
            b = st.selectbox("Signal B", idxs, index=min(1, len(idxs)-1), key="coh_b", format_func=fmt)
            if a == b:
                st.warning("Select two different signals.")
            else:
                obs_a = st.session_state.observations[a]
                obs_b = st.session_state.observations[b]
                x, y = _pairwise_trim(obs_a["data"], obs_b["data"])
                dt = 1.0 / float(fs)
                mother = pycwt_wavelet.Morlet(6)

                W12, cross, coi, freq, signif, rsq, period, scale, wcoh, phase = \
                    pycwt_wavelet.wct(np.asarray(x, float), np.asarray(y, float), dt, mother)

                t = np.arange(len(x), float) * dt
                rsq = np.asarray(rsq, float)
                period = np.asarray(period, float)
                freq_hz = 1.0 / np.maximum(period, 1e-12)

                fig_coh = go.Figure()
                safe_heatmap(fig_coh, rsq, t, freq_hz, zmin=0.0, zmax=1.0, colorscale="Turbo",
                             name=f"A: {fmt(a)} · B: {fmt(b)}")

                try:
                    coi = np.asarray(coi, float)
                    fcoi = 1.0 / np.maximum(coi, 1e-12)
                    fig_coh.add_trace(go.Scatter(x=t.tolist(), y=fcoi.tolist(),
                                                 mode="lines",
                                                 line=dict(color="white", width=2, dash="dash"),
                                                 name="COI"))
                except Exception:
                    pass

                st.plotly_chart(fig_coh, use_container_width=True, key="plt_coh")

else:
    st.info("1) Upload ZIP and segment beads. 2) Add one or more signals to analyze in the wavelet tabs.")
