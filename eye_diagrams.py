import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ==============================================================
# PULSE SHAPE FUNCTIONS (as used in textbook binary_eye.m)
# ==============================================================

def p_nrz(Tau):
    """
    NRZ (Non-Return-to-Zero) pulse.
    Flat at +1 for the entire symbol period Tau samples.

    WHAT IT IS: The simplest pulse. Voltage stays constant for
    the full bit duration Tb. No return to zero between bits.

    Shape: ████████  (filled rectangle, full width)
    """
    return np.ones(Tau)


def p_rz(Tau):
    """
    RZ (Return-to-Zero) pulse.
    +1 for first half of symbol period, 0 for second half.

    WHAT IT IS: Signal goes high for half a bit period, then
    RETURNS TO ZERO before the next bit starts.

    Shape: ████____  (half-filled rectangle)

    WHY USE IT: Easier clock recovery (the zero-crossing gives
    a timing reference). But uses 2× bandwidth of NRZ.
    """
    pulse = np.zeros(Tau)
    pulse[:Tau // 2] = 1.0   # first half = 1, second half = 0
    return pulse


def p_sinc(Tau, n_periods=6):
    """
    Sinc pulse (ideal Nyquist pulse), truncated to n_periods.

    p(t) = sinc(t/Tb) = sin(πt/Tb) / (πt/Tb)

    WHAT IT IS: The mathematically ideal pulse.
    - Zero at t = ±Tb, ±2Tb, ±3Tb, ... (zero ISI)
    - Minimum possible bandwidth (rectangular spectrum)

    Shape: ∿∿∿∿∿█∿∿∿∿∿  (tall center peak, oscillating tails)

    WHY NOT USE IT IN PRACTICE:
    - It extends to infinity in time (impossible to implement)
    - Very sensitive to timing errors (sidelobes are large)

    Parameters:
        Tau       : samples per symbol period
        n_periods : how many Tb periods to include on each side
    """
    total_len = 2 * n_periods * Tau + 1
    t = np.arange(-(total_len // 2), total_len // 2 + 1)
    # np.sinc(x) = sin(πx)/(πx), so sinc(t/Tau) does what we want
    pulse = np.sinc(t / Tau)
    return pulse


def p_rc(Tau, beta=0.5, n_periods=4):
    """
    Raised Cosine pulse (Equation 7.36 from Lathi).

    p(t) = sinc(t/Tb) × cos(πβt/Tb) / (1 - (2βt/Tb)²)

    WHAT IT IS: The industry-standard pulse.
    - Like sinc but with a cosine "rolloff" in frequency
    - Still zero ISI at t = ±Tb, ±2Tb, ... (Nyquist criterion)
    - Bandwidth = (1+β)/(2Tb) — slightly more than sinc
    - Tails decay as 1/t³ vs 1/t for sinc → much more robust

    Parameters:
        Tau      : samples per symbol period (= oversampling factor T in textbook)
        beta     : rolloff factor (0 ≤ β ≤ 1)
                   β=0 → sinc, β=0.5 → standard, β=1 → cosine
        n_periods: truncation length (textbook uses Td=4)

    TEXTBOOK NOTE: The textbook calls this prcos(rolloff, length, T)
    where T = Tau (oversampling factor). We implement the same thing.
    """
    total_len = 2 * n_periods * Tau + 1
    t = np.arange(-(total_len // 2), total_len // 2 + 1, dtype=float)

    pulse = np.zeros(total_len)

    for i, ti in enumerate(t):
        if ti == 0:
            pulse[i] = 1.0
        elif beta != 0 and abs(abs(2 * beta * ti / Tau) - 1) < 1e-6:
            # Special singular point: t = ±Tau/(2*beta)
            pulse[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
        else:
            numer = np.sinc(ti / Tau) * np.cos(np.pi * beta * ti / Tau)
            denom = 1.0 - (2 * beta * ti / Tau) ** 2
            pulse[i] = numer / denom

    return pulse


# ==============================================================
# CORE FUNCTION: make_eye_diagram
# (Python equivalent of MATLAB's eyediagram function)
# ==============================================================
def make_eye_diagram(waveform, Tau, offset=0, n_traces=None):
    """
    Creates eye diagram data by chopping waveform into 2*Tau segments.

    This replicates MATLAB's eyediagram(y, 2*Tau, Tau, Tau/2) call.

    HOW IT WORKS:
    1. Start at 'offset' samples into the waveform
    2. Take segments of length 2*Tau samples each
    3. Store all segments — these are the "traces"
    4. Plotting all traces overlaid gives the eye diagram

    Parameters:
    -----------
    waveform  : numpy array — the signal to analyze
    Tau       : int — samples per symbol period
    offset    : int — starting offset in samples (like Tau/2 in textbook)
                This shifts which part of the bit period we see.
                offset=0     → see full bit transition
                offset=Tau/2 → centered on bit, shows eye opening clearly
    n_traces  : int or None — max traces to plot (None = all)

    Returns:
    --------
    traces    : list of numpy arrays, each of length 2*Tau
    t_eye     : normalized time axis [0, 2] (in symbol periods)
    """
    segment_len = 2 * Tau

    # Start from offset position
    start = offset % Tau
    traces = []

    idx = start
    while idx + segment_len <= len(waveform):
        traces.append(waveform[idx: idx + segment_len])
        idx += Tau   # advance by one symbol period

    if n_traces is not None:
        traces = traces[:n_traces]

    # Normalized time axis: 0 to 2 symbol periods
    t_eye = np.linspace(0, 2, segment_len)

    return traces, t_eye


# ==============================================================
# FUNCTION: plot_single_eye — Plot one eye diagram
# ==============================================================
def plot_single_eye(ax, traces, t_eye, title, color='steelblue',
                    alpha=0.08, show_annotations=True):
    """
    Plots one eye diagram on a given matplotlib axis.

    Parameters:
    -----------
    ax     : matplotlib axis
    traces : list of trace arrays (from make_eye_diagram)
    t_eye  : time axis
    title  : plot title string
    color  : trace color
    alpha  : transparency of each trace (low = can see density)
    show_annotations : whether to add sampling instant line etc.
    """
    for trace in traces:
        ax.plot(t_eye, trace, color=color, alpha=alpha, linewidth=0.7)

    if show_annotations:
        # Mark optimal sampling instant (center of eye = t=1.0)
        ax.axvline(x=1.0, color='red', linestyle='--',
                   linewidth=1.5, label='Optimal sampling instant', zorder=5)
        # Decision threshold at 0 (for polar signaling)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1.0,
                   alpha=0.6, label='Decision threshold (0)', zorder=5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (symbol periods)', fontsize=9)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.grid(True, alpha=0.2)

    if show_annotations:
        ax.legend(fontsize=7, loc='upper right')


# ==============================================================
# MAIN FUNCTION: plot_all_eye_diagrams
# Exactly replicates textbook's binary_eye.m for all 4 pulse types
# ==============================================================
def plot_all_eye_diagrams(n_symbols=400, Tau=64, beta=0.5, Td=4,
                           title_prefix="Binary Eye Diagrams"):
    """
    Generates and plots eye diagrams for all 4 pulse types.

    This is the DIRECT Python equivalent of binary_eye.m from the textbook.

    TEXTBOOK MAPPING:
    -----------------
    data = sign(randn(1,400))      ← 400 random ±1 bits
    Tau = 64                        ← 64 samples per symbol
    dataup = upsample(data, Tau)   ← impulse train
    yrz  = conv(dataup, prz(Tau))  ← convolve with RZ pulse
    ynrz = conv(dataup, pnrz(Tau)) ← convolve with NRZ pulse
    yrc  = conv(dataup, prcos(β, Td, Tau)) ← RC pulse
    eyediagram(y, 2*Tau, Tau, Tau/2) ← plot eye

    Parameters:
    -----------
    n_symbols : number of random binary symbols (textbook uses 400)
    Tau       : samples per symbol period (textbook uses 64)
    beta      : rolloff factor for RC pulse (textbook uses 0.5)
    Td        : RC pulse truncation in symbol periods (textbook uses 4)
    title_prefix : main title string

    Returns:
    --------
    fig : matplotlib figure with 4 eye diagrams (2×2 grid)
    """
    # === STEP 1: Generate random binary data (±1) ===
    # This matches: data = sign(randn(1, 400)) in MATLAB
    np.random.seed(42)
    data = np.sign(np.random.randn(n_symbols))
    data[data == 0] = 1   # handle rare case of exactly 0

    # === STEP 2: Upsample (create impulse train) ===
    # MATLAB: dataup = upsample(data, Tau)
    # Python equivalent: insert (Tau-1) zeros between each symbol
    dataup = np.zeros(n_symbols * Tau)
    for i, val in enumerate(data):
        dataup[i * Tau] = val

    # === STEP 3: Create pulse shapes ===
    pulse_nrz  = p_nrz(Tau)
    pulse_rz   = p_rz(Tau)
    pulse_sinc = p_sinc(Tau, n_periods=6)
    pulse_rc   = p_rc(Tau, beta=beta, n_periods=Td)

    # === STEP 4: Convolve impulse train with each pulse shape ===
    # MATLAB: yrz = conv(dataup, prz(Tau))
    #         yrz = yrz(1:end-Tau+1)  ← trim
    yrz   = np.convolve(dataup, pulse_rz)[:len(dataup) - Tau + 1]
    ynrz  = np.convolve(dataup, pulse_nrz)[:len(dataup) - Tau + 1]
    ysinc = np.convolve(dataup, pulse_sinc)
    # For sinc: trim to remove startup transient (6 periods on each side)
    trim_sinc = 6 * Tau
    ysinc = ysinc[trim_sinc: trim_sinc + len(dataup)]

    # RC pulse: trim 2*Td*Tau from each side (textbook does this)
    # MATLAB: yrc = yrc(2*Td*Tau : end-2*Td*Tau+1)
    yrc = np.convolve(dataup, pulse_rc)
    trim_rc = 2 * Td * Tau
    if trim_rc < len(yrc) // 2:
        yrc = yrc[trim_rc: -trim_rc + 1] if trim_rc > 0 else yrc

    # === STEP 5: Generate eye diagram traces ===
    # Offset = Tau//2 centers the eye nicely (matches MATLAB's Tau/2 offset)
    offset = Tau // 2

    traces_rz,   t_eye = make_eye_diagram(yrz,   Tau, offset=offset)
    traces_nrz,  _     = make_eye_diagram(ynrz,  Tau, offset=offset)
    traces_sinc, _     = make_eye_diagram(ysinc, Tau, offset=offset)
    traces_rc,   _     = make_eye_diagram(yrc,   Tau, offset=0)   # RC: no offset needed

    # === STEP 6: Plot all 4 eye diagrams ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix}\n'
                 f'({n_symbols} random bits, Tau={Tau} samples/symbol, RC β={beta})',
                 fontsize=13, fontweight='bold')

    # Limit traces for performance (200 is enough to see the pattern)
    MAX_TRACES = 200

    plot_single_eye(axes[0, 0], traces_rz[:MAX_TRACES],   t_eye,
                    'RZ Eye Diagram\n(Returns to zero each bit)',
                    color='darkorange')

    plot_single_eye(axes[0, 1], traces_nrz[:MAX_TRACES],  t_eye,
                    'NRZ Eye Diagram\n(Non-Return-to-Zero)',
                    color='steelblue')

    plot_single_eye(axes[1, 0], traces_sinc[:MAX_TRACES], t_eye,
                    'Sinc Pulse Eye Diagram\n(Ideal Nyquist, truncated to 6T)',
                    color='darkgreen')

    plot_single_eye(axes[1, 1], traces_rc[:MAX_TRACES],   t_eye,
                    f'Raised Cosine Eye Diagram\n(β={beta}, truncated to {Td}T)',
                    color='crimson')

    plt.tight_layout()
    return fig


# ==============================================================
# FUNCTION: plot_eye_with_noise — Show how noise closes the eye
# ==============================================================
def plot_eye_with_noise(Tau=64, beta=0.5, Td=4, n_symbols=400,
                         snr_levels=[None, 20, 10, 6]):
    """
    Shows how adding AWGN noise progressively closes the eye.

    This demonstrates the eye diagram as a diagnostic tool:
    - No noise    → wide open eye
    - Small noise → slightly fuzzy but still open
    - Large noise → eye closes → many bit errors

    Parameters:
    -----------
    Tau        : samples per symbol
    beta       : RC rolloff factor
    Td         : RC truncation length
    n_symbols  : number of symbols
    snr_levels : list of SNR values in dB (None = no noise)
    """
    np.random.seed(42)
    data = np.sign(np.random.randn(n_symbols))
    data[data == 0] = 1

    dataup = np.zeros(n_symbols * Tau)
    for i, val in enumerate(data):
        dataup[i * Tau] = val

    pulse = p_rc(Tau, beta=beta, n_periods=Td)
    trim  = 2 * Td * Tau

    yrc = np.convolve(dataup, pulse)
    yrc = yrc[trim: -trim + 1] if trim > 0 else yrc

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f'Eye Diagram: Effect of AWGN Noise on RC Pulse (β={beta})\n'
                 f'Noise closes the eye → reduces noise margin',
                 fontsize=12, fontweight='bold')

    titles = ['No Noise\n(SNR = ∞)',
              f'SNR = {snr_levels[1]} dB\n(Low noise)',
              f'SNR = {snr_levels[2]} dB\n(Medium noise)',
              f'SNR = {snr_levels[3]} dB\n(High noise)']

    colors = ['crimson', 'darkorange', 'steelblue', 'purple']

    for ax, snr_db, title, color in zip(axes, snr_levels, titles, colors):
        if snr_db is None:
            y = yrc.copy()
        else:
            sig_pwr = np.mean(yrc ** 2)
            snr_lin = 10 ** (snr_db / 10)
            noise_std = np.sqrt(sig_pwr / snr_lin)
            noise = np.random.normal(0, noise_std, len(yrc))
            y = yrc + noise

        traces, t_eye = make_eye_diagram(y, Tau, offset=0)
        plot_single_eye(ax, traces[:200], t_eye, title,
                        color=color, show_annotations=True)

    plt.tight_layout()
    return fig


# ==============================================================
# FUNCTION: plot_pulse_spectra — Full spectrum analysis
# ==============================================================
def plot_pulse_spectra(Tau=64, beta_list=[0.0, 0.25, 0.5, 0.75, 1.0], Td=6):
    """
    Detailed comparison of all pulse spectra 
    SHOWS:
    1. Time domain: All 4 pulse types side by side
    2. Frequency domain: RC for different β values
    3. RC vs Sinc direct comparison (both time and frequency)

    NYQUIST BANDWIDTH: B = 1/(2Tb) = 1/(2*Tau) (normalized)
    This is the MINIMUM bandwidth needed to send without ISI.
    """
    NFFT = 8192
    # Time axis for pulses (in samples, normalized by Tau)
    t_rect = np.arange(Tau)
    t_long = np.arange(-(Td * Tau), Td * Tau + 1)

    # Frequency axis (normalized: f * Tau)
    freq = np.fft.fftshift(np.fft.fftfreq(NFFT)) * Tau   # f*Tau normalized

    # ─── Figure 1: All 4 pulses in time & frequency ───────────────
    fig1, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig1.suptitle('Pulse Shape Comparison: Time Domain (top) and Frequency Domain (bottom)\n'
                  'All amplitudes normalized to 1 at t=0',
                  fontsize=12, fontweight='bold')

    pulse_data = [
        ('NRZ', p_nrz(Tau), t_rect, 'steelblue'),
        ('RZ',  p_rz(Tau),  t_rect, 'darkorange'),
        ('Sinc (6T)', p_sinc(Tau, 6), np.arange(-(6*Tau), 6*Tau+1), 'darkgreen'),
        (f'RC (β=0.5)', p_rc(Tau, 0.5, Td), t_long, 'crimson'),
    ]

    for col, (name, pulse, t_p, color) in enumerate(pulse_data):
        ax_t = axes[0, col]
        ax_f = axes[1, col]

        # Time domain
        t_norm = t_p / Tau   # normalize by Tau → units of symbol periods
        ax_t.plot(t_norm, pulse, color=color, linewidth=2)
        ax_t.set_title(f'{name}\nTime Domain', fontsize=10, fontweight='bold')
        ax_t.set_xlabel('t / Tb (symbol periods)')
        ax_t.set_ylabel('p(t)')
        ax_t.axhline(0, color='k', linewidth=0.5)
        ax_t.axvline(0, color='k', linewidth=0.5, alpha=0.3)
        ax_t.grid(True, alpha=0.3)

        # Mark zero crossings at integer multiples of Tb
        for n in range(-Td, Td + 1):
            if n != 0:
                ax_t.axvline(x=n, color='gray', linestyle=':', alpha=0.2)

        # Clip x-axis for readability
        ax_t.set_xlim([max(t_norm[0], -5), min(t_norm[-1], 5)])

        # Frequency domain (magnitude spectrum)
        P = np.abs(np.fft.fftshift(np.fft.fft(pulse, NFFT)))
        P = P / np.max(P)   # normalize to peak = 1
        ax_f.plot(freq, P, color=color, linewidth=2)
        ax_f.set_title(f'{name}\nFrequency Domain', fontsize=10, fontweight='bold')
        ax_f.set_xlabel('f × Tb (normalized frequency)')
        ax_f.set_ylabel('|P(f)| (normalized)')
        ax_f.set_xlim([-2.5, 2.5])
        ax_f.set_ylim([0, 1.15])
        ax_f.axvline(x=0.5, color='red', linestyle='--', alpha=0.6,
                     linewidth=1.5, label='Nyquist BW (f=1/2Tb)')
        ax_f.axvline(x=-0.5, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        ax_f.legend(fontsize=7)
        ax_f.grid(True, alpha=0.3)

    plt.tight_layout()

    # ─── Figure 2: RC for multiple β values ────────────────────────
    fig2, (ax_t2, ax_f2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Raised Cosine Pulse: Effect of Rolloff Factor β\n'
                  'β=0 → Sinc (minimum bandwidth, hardest) | β=1 → maximum rolloff (easiest)',
                  fontsize=12, fontweight='bold')

    colors_beta = plt.cm.plasma(np.linspace(0.1, 0.9, len(beta_list)))

    for beta, color in zip(beta_list, colors_beta):
        pulse = p_rc(Tau, beta, Td)
        t_norm = np.arange(-(Td * Tau), Td * Tau + 1) / Tau

        # Time domain
        ax_t2.plot(t_norm, pulse, color=color, linewidth=2,
                   label=f'β={beta}')

        # Frequency domain
        P = np.abs(np.fft.fftshift(np.fft.fft(pulse, NFFT)))
        P = P / np.max(P)
        ax_f2.plot(freq, P, color=color, linewidth=2, label=f'β={beta}')

    # Add sinc (β=0 limit) for reference
    p_sinc_ref = p_sinc(Tau, n_periods=Td)
    t_sinc_ref = np.arange(-(Td * Tau), Td * Tau + 1) / Tau
    # (already plotted as β=0 case)

    ax_t2.axhline(0, color='k', linewidth=0.5)
    for n in range(-4, 5):
        if n != 0:
            ax_t2.axvline(x=n, color='gray', linestyle=':', alpha=0.2)
    ax_t2.set_title('Time Domain', fontsize=11)
    ax_t2.set_xlabel('t / Tb')
    ax_t2.set_ylabel('p(t)')
    ax_t2.set_xlim([-5, 5])
    ax_t2.legend(title='Rolloff β', fontsize=9)
    ax_t2.grid(True, alpha=0.3)
    ax_t2.set_title('RC Pulse: Time Domain\n'
                    '(Zero crossings at t = ±Tb, ±2Tb, ... for all β → zero ISI!)',
                    fontsize=10)

    ax_f2.axvline(x=0.5, color='k', linestyle='--',
                  linewidth=2, label='Nyquist BW', alpha=0.5)
    ax_f2.axvline(x=-0.5, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax_f2.set_xlim([-2, 2])
    ax_f2.set_ylim([0, 1.15])
    ax_f2.set_xlabel('f × Tb (normalized)')
    ax_f2.set_ylabel('|P(f)| (normalized)')
    ax_f2.legend(title='Rolloff β', fontsize=9)
    ax_f2.grid(True, alpha=0.3)
    ax_f2.set_title('RC Pulse: Frequency Domain\n'
                    'Bandwidth = (1+β)/(2Tb). More β → more bandwidth used',
                    fontsize=10)

    # Add bandwidth annotations
    for beta in [0.5, 1.0]:
        bw = (1 + beta) / 2   # normalized bandwidth
        ax_f2.annotate(f'BW(β={beta})\n= {bw:.2f}/Tb',
                       xy=(bw, 0.15), xytext=(bw + 0.2, 0.35),
                       arrowprops=dict(arrowstyle='->', color='gray'),
                       fontsize=8, color='gray')

    plt.tight_layout()

    return fig1, fig2


# ==============================================================
# FUNCTION: plot_eye_comparison_with_noise
# Shows all 4 pulses + noise effects in one comprehensive figure
# ==============================================================
def plot_eye_comparison_with_noise(Tau=64, beta=0.5, Td=4,
                                    n_symbols=500, snr_db=10):
    """
    2×4 grid: top row = no noise, bottom row = with noise.
    Directly shows how noise affects different pulse eye diagrams.
    """
    np.random.seed(7)
    data = np.sign(np.random.randn(n_symbols))
    data[data == 0] = 1

    dataup = np.zeros(n_symbols * Tau)
    for i, val in enumerate(data):
        dataup[i * Tau] = val

    # Generate all 4 waveforms
    offset = Tau // 2

    yrz   = np.convolve(dataup, p_nrz(Tau))[:len(dataup) - Tau + 1]
    ynrz  = np.convolve(dataup, p_rz(Tau))[:len(dataup) - Tau + 1]

    sinc_p = p_sinc(Tau, 6)
    ysinc = np.convolve(dataup, sinc_p)
    trim_s = 6 * Tau
    ysinc = ysinc[trim_s: trim_s + len(dataup)]

    rc_p = p_rc(Tau, beta, Td)
    trim_rc = 2 * Td * Tau
    yrc = np.convolve(dataup, rc_p)
    yrc = yrc[trim_rc: -trim_rc + 1] if trim_rc > 0 else yrc

    waveforms = {'NRZ':   (yrz,   offset, 'steelblue'),
                 'RZ':    (ynrz,  offset, 'darkorange'),
                 'Sinc':  (ysinc, offset, 'darkgreen'),
                 f'RC β={beta}': (yrc, 0, 'crimson')}

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(f'Eye Diagrams — All Pulse Types\n'
                 f'Top: No Noise | Bottom: AWGN SNR={snr_db} dB | Tau={Tau}, {n_symbols} symbols',
                 fontsize=12, fontweight='bold')

    pulse_names = list(waveforms.keys())

    for col, name in enumerate(pulse_names):
        y, off, color = waveforms[name]
        sig_pwr = np.mean(y ** 2)
        snr_lin = 10 ** (snr_db / 10)
        noise_std = np.sqrt(sig_pwr / snr_lin)

        # Top row: no noise
        traces, t_eye = make_eye_diagram(y, Tau, offset=off)
        plot_single_eye(axes[0, col], traces[:200], t_eye,
                        f'{name}\n(No Noise)', color=color,
                        show_annotations=(col == 0))

        # Bottom row: with noise
        y_noisy = y + np.random.normal(0, noise_std, len(y))
        traces_n, _ = make_eye_diagram(y_noisy, Tau, offset=off)
        plot_single_eye(axes[1, col], traces_n[:200], t_eye,
                        f'{name}\n(SNR={snr_db} dB)', color=color,
                        show_annotations=(col == 0))

    plt.tight_layout()
    return fig
