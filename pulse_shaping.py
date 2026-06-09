"""
RAISED COSINE (RC) PULSE — Equation 7.36 from Lathi (EXACT):

   TEXTBOOK FORM (Eq. 7.36):
   p(t) = Rb · cos(π·Rb·t) / (1 - 4·Rb²·t²) · sinc(π·Rb·t)

   where:
     Rb = 1/Tb = bit rate (symbols per second)
     sinc(x) = sin(x)/x  (UNNORMALIZED sinc!)

   Expanding sinc(π·Rb·t) = sin(π·Rb·t)/(π·Rb·t):
   p(t) = cos(π·Rb·t)·sin(π·Rb·t) / (π·t·(1 - 4·Rb²·t²))
        = sin(2π·Rb·t) / (2π·t·(1 - 4·Rb²·t²))

   This is the MAXIMUM ROLLOFF (β=1) raised cosine pulse.

   GENERAL FORM (covers all rolloff factors 0 ≤ β ≤ 1):
   p(t) = sinc(t/Tb) · cos(πβt/Tb) / (1 - (2βt/Tb)²)
   - β=0 → reduces to sinc pulse (minimum bandwidth)
   - β=1 → Eq. 7.36 (maximum rolloff)
   - ZERO ISI at multiples of Tb (Nyquist criterion satisfied)
   - Bandwidth = (1+β)/(2Tb) — slightly more than sinc but finite rolloff

BANDWIDTH COMPARISON:
---------------------
- Rectangular: theoretically infinite (sinc spectrum)
- Sinc:        B = 1/(2Tb) Hz  [minimum possible]
- RC (β=0):    B = 1/(2Tb) Hz  [same as sinc]
- RC (β=0.5):  B = 0.75/Tb Hz
- RC (β=1):    B = 1/Tb Hz     [double sinc bandwidth — this is Eq 7.36]
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================
# FUNCTION: rect_pulse — Rectangular (NRZ) pulse
# ==============================================================
def rect_pulse(t, Tb):
    """
    Rectangular pulse of unit height, duration Tb, centered at t=0.

    p(t) = 1  for |t| <= Tb/2
           0  otherwise

    Parameters:
        t  : time array
        Tb : bit duration
    Returns:
        p  : pulse values at times t
    """
    p = np.zeros_like(t, dtype=float)
    p[np.abs(t) <= Tb / 2] = 1.0
    return p


# ==============================================================
# FUNCTION: sinc_pulse — Ideal sinc pulse
# ==============================================================
def sinc_pulse(t, Tb):
    """
    Ideal sinc pulse for zero-ISI transmission.

    p(t) = sinc(t/Tb) = sin(π*t/Tb) / (π*t/Tb)

    Note: np.sinc(x) = sin(πx)/(πx), so we use np.sinc(t/Tb)

    Parameters:
        t  : time array
        Tb : bit duration
    Returns:
        p  : sinc pulse values
    """
    return np.sinc(t / Tb)


# ==============================================================
# FUNCTION: rc_pulse_eq736 — Equation 7.36 from Lathi
# ==============================================================
def rc_pulse_eq736(t, Rb):
    """
    Raised Cosine pulse — EXACT implementation of Equation 7.36 from Lathi.

    TEXTBOOK FORMULA:
    -----------------
    p(t) = Rb · [cos(π·Rb·t) / (1 - 4·Rb²·t²)] · sinc(π·Rb·t)

    where sinc(x) = sin(x)/x  (Lathi's unnormalized sinc convention)

    Expanding fully:
    p(t) = Rb · cos(π·Rb·t) · sin(π·Rb·t) / [π·Rb·t · (1 - 4·Rb²·t²)]
         = sin(2π·Rb·t) / [2π·t · (1 - 4·Rb²·t²)]

    - Rb = 1/Tb = bit rate in bits/second
    - This is the β=1 (maximum rolloff) case of the general RC family
    - The Rb multiplier makes peak value = Rb at t=0 (energy normalization)
    - Zero crossings at t = k/Rb = k·Tb for all integers k ≠ 0  → ZERO ISI
    - Bandwidth = 1/Tb = Rb Hz  (twice the Nyquist minimum)

    SPECIAL CASES (L'Hôpital's rule applied):
    ------------------------------------------
    t = 0:         p(0) = Rb  (both numerator and denominator → 0, limit = Rb)
    t = ±1/(2Rb):  p = π·Rb/4  (denominator → 0, use separate limit)

    Parameters:
    -----------
    t  : numpy array — time values (seconds)
    Rb : float — bit rate = 1/Tb (bits per second)

    Returns:
    --------
    p  : numpy array — pulse amplitude at each time t
    """
    p = np.zeros_like(t, dtype=float)
    Tb = 1.0 / Rb

    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            # Limit as t→0: use L'Hôpital or Taylor series
            # sin(2π·Rb·t)/(2π·t) → Rb as t→0, and 1/(1-4Rb²t²) → 1
            p[i] = Rb

        elif abs(4 * Rb**2 * ti**2 - 1) < 1e-6:
            # Singular point: t = ±1/(2·Rb) = ±Tb/2
            # Use limit: lim = π·Rb/4  (derived via L'Hôpital)
            p[i] = (np.pi * Rb) / 4.0

        else:
            # General case: expand sinc(π·Rb·t) = sin(π·Rb·t)/(π·Rb·t)
            numerator   = np.cos(np.pi * Rb * ti) * np.sin(np.pi * Rb * ti)
            denominator = np.pi * Rb * ti * (1.0 - 4.0 * Rb**2 * ti**2)
            p[i] = numerator / denominator

    return p


# ==============================================================
# FUNCTION: raised_cosine_pulse — General RC pulse (all β values)
# ==============================================================
def raised_cosine_pulse(t, Tb, beta):
    """
    General Raised Cosine pulse for any rolloff factor β.

    FORMULA:
    p(t) = sinc(t/Tb) · cos(π·β·t/Tb) / (1 - (2·β·t/Tb)²)

    RELATIONSHIP TO EQ 7.36:
    - Set β=1 and Tb=1/Rb → same shape as rc_pulse_eq736, just peak=1 instead of Rb
    - rc_pulse_eq736(t, Rb) = Rb · raised_cosine_pulse(t, Tb=1/Rb, beta=1)
    - This function is more general: works for any β ∈ [0, 1]

    Parameters:
        t    : time array
        Tb   : bit duration (seconds)
        beta : rolloff factor (0 ≤ beta ≤ 1)
               beta=0 → sinc pulse
               beta=1 → Equation 7.36 shape (normalized to peak=1)

    Returns:
        p    : raised cosine pulse values
    """
    if beta == 0:
        # Degenerate case: RC becomes sinc
        return np.sinc(t / Tb)

    p = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if ti == 0:
            p[i] = 1.0   # limit as t→0 is 1
        else:
            denom_inner = (2 * beta * ti / Tb) ** 2
            if np.abs(denom_inner - 1) < 1e-6:
                # Special point t = ±Tb/(2*beta): use limit
                p[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
            else:
                numerator = np.sinc(ti / Tb) * np.cos(np.pi * beta * ti / Tb)
                denominator = 1 - denom_inner
                p[i] = numerator / denominator

    return p


# ==============================================================
# FUNCTION: plot_eq736 — Dedicated plot for Equation 7.36
# This is the key plot the professor wants
# ==============================================================
def plot_eq736(Rb=1.0, n_periods=6):
    """
    Dedicated demonstration of Equation 7.36 — exactly as in the textbook.

    Shows:
    1. Time domain: p(t) = Rb·cos(πRbt)/(1-4Rb²t²)·sinc(πRbt)
    2. Frequency domain spectrum of this pulse
    3. Comparison with sinc (β=0) to show extra bandwidth used
    4. Verification of zero ISI property

    Parameters:
    -----------
    Rb       : float — bit rate (default 1.0 Hz for normalized demo)
    n_periods: int — time range shown = ±n_periods/Rb
    """
    Tb = 1.0 / Rb
    dt = Tb / 100   # fine time resolution
    t  = np.arange(-n_periods * Tb, n_periods * Tb + dt, dt)

    # Compute Eq 7.36 pulse
    p_736  = rc_pulse_eq736(t, Rb)

    # Compute sinc (for comparison — this is the β=0 limit)
    p_sinc = sinc_pulse(t, Tb)   # peak = 1

    # Scale sinc to same peak as eq736 (peak = Rb) for fair visual comparison
    p_sinc_scaled = p_sinc * Rb

    # General RC with several β values (all normalized to peak=1 then scaled to Rb)
    betas = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Equation 7.36 (Lathi): Raised Cosine Pulse\n'
        r'$p(t) = R_b \cdot \frac{\cos(\pi R_b t)}{1-4R_b^2 t^2} \cdot \mathrm{sinc}(\pi R_b t)$'
        f'        [Rb = {Rb} bps, Tb = {Tb} s]',
        fontsize=12, fontweight='bold'
    )

    # ── Subplot 1: Eq 7.36 in time domain ──────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(t / Tb, p_736, 'r-', linewidth=2.5, label='Eq 7.36: RC (β=1)', zorder=3)
    ax1.plot(t / Tb, p_sinc_scaled, 'b--', linewidth=1.8,
             label='Sinc (β=0) [scaled]', alpha=0.8)

    # Mark zero crossings — at t = k*Tb (ISI-free condition)
    for k in range(-n_periods, n_periods + 1):
        if k != 0:
            ax1.axvline(x=k, color='gray', linestyle=':', alpha=0.4, linewidth=1)
    ax1.scatter([k for k in range(-5, 6) if k != 0],
                [rc_pulse_eq736(np.array([k * Tb]), Rb)[0] for k in range(-5, 6) if k != 0],
                color='red', s=60, zorder=5, label='Zero crossings at t=k·Tb\n(zero ISI!)')

    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.axvline(0, color='k', linewidth=0.8, alpha=0.3)
    ax1.set_xlabel('t / Tb  (normalized time)', fontsize=10)
    ax1.set_ylabel('p(t)', fontsize=10)
    ax1.set_title('Time Domain: Eq 7.36 vs Sinc', fontsize=11, fontweight='bold')
    ax1.set_xlim([-5.5, 5.5])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Annotate peak value
    ax1.annotate(f'Peak = Rb = {Rb}',
                 xy=(0, Rb), xytext=(1.5, Rb * 0.9),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')

    # ── Subplot 2: Frequency domain ────────────────────────────────
    ax2 = axes[0, 1]
    NFFT = 8192
    freq = np.fft.fftshift(np.fft.fftfreq(NFFT, dt))
    freq_norm = freq * Tb   # normalize: f*Tb

    P_736  = np.abs(np.fft.fftshift(np.fft.fft(p_736,  NFFT)))
    P_sinc = np.abs(np.fft.fftshift(np.fft.fft(p_sinc_scaled, NFFT)))

    # Normalize both to max=1 for shape comparison
    P_736  /= np.max(P_736)
    P_sinc /= np.max(P_sinc)

    ax2.plot(freq_norm, P_736,  'r-',  linewidth=2.5, label='Eq 7.36 RC (β=1)')
    ax2.plot(freq_norm, P_sinc, 'b--', linewidth=1.8, label='Sinc (β=0)', alpha=0.8)

    # Mark key bandwidths
    ax2.axvline(x=0.5,  color='blue',  linestyle='--', linewidth=1.5, alpha=0.7,
                label='Nyquist BW = 1/(2Tb)')
    ax2.axvline(x=-0.5, color='blue',  linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=1.0,  color='red',   linestyle='--', linewidth=1.5, alpha=0.7,
                label='RC BW = 1/Tb  (2× Nyquist)')
    ax2.axvline(x=-1.0, color='red',   linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.fill_betweenx([0, 1.1], -0.5, 0.5, alpha=0.1, color='blue', label='Sinc occupancy')
    ax2.fill_betweenx([0, 1.1], -1.0, 1.0, alpha=0.08, color='red', label='RC occupancy')

    ax2.set_xlabel('f × Tb  (normalized frequency)', fontsize=10)
    ax2.set_ylabel('|P(f)|  (normalized)', fontsize=10)
    ax2.set_title('Frequency Domain: RC (Eq 7.36) vs Sinc', fontsize=11, fontweight='bold')
    ax2.set_xlim([-2.5, 2.5])
    ax2.set_ylim([0, 1.15])
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Subplot 3: Family of RC curves (all β) in time domain ──────
    ax3 = axes[1, 0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(betas)))
    for beta, color in zip(betas, colors):
        p_b = raised_cosine_pulse(t, Tb, beta) * Rb   # scale to Rb peak
        lbl = f'β={beta}' + (' ← Eq 7.36' if beta == 1.0 else '')
        lw  = 2.5 if beta == 1.0 else 1.5
        ax3.plot(t / Tb, p_b, color=color, linewidth=lw, label=lbl)

    for k in range(-5, 6):
        if k != 0:
            ax3.axvline(x=k, color='gray', linestyle=':', alpha=0.2)
    ax3.axhline(0, color='k', linewidth=0.8)
    ax3.set_xlabel('t / Tb', fontsize=10)
    ax3.set_ylabel('p(t)', fontsize=10)
    ax3.set_title('RC Family: All β Values\n(All share same zero crossings — zero ISI)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlim([-5.5, 5.5])
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Subplot 4: Family of RC curves in frequency domain ──────────
    ax4 = axes[1, 1]
    for beta, color in zip(betas, colors):
        p_b = raised_cosine_pulse(t, Tb, beta)
        P_b = np.abs(np.fft.fftshift(np.fft.fft(p_b, NFFT)))
        P_b /= np.max(P_b)
        bw  = (1 + beta) / 2   # normalized bandwidth
        lbl = f'β={beta}  [BW={(1+beta)/2:.2f}/Tb]' + (' ← Eq 7.36' if beta == 1.0 else '')
        lw  = 2.5 if beta == 1.0 else 1.5
        ax4.plot(freq_norm, P_b, color=color, linewidth=lw, label=lbl)

    ax4.axvline(x=0.5,  color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.axvline(x=-0.5, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    ax4.text(0.52, 0.85, 'Nyquist\nBW', fontsize=8, color='gray')
    ax4.set_xlabel('f × Tb  (normalized frequency)', fontsize=10)
    ax4.set_ylabel('|P(f)|  (normalized)', fontsize=10)
    ax4.set_title('Frequency Domain: RC Family\nBandwidth = (1+β)/(2Tb)',
                  fontsize=11, fontweight='bold')
    ax4.set_xlim([-2, 2])
    ax4.set_ylim([0, 1.15])
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig



# ==============================================================
# FUNCTION: apply_pulse_shaping — Convolve bitstream with pulse
# ==============================================================
def apply_pulse_shaping(bitstream_polar, Tb, samples_per_bit,
                         pulse_type='rc', beta=0.5):
    """
    Applies pulse shaping to a polar bitstream (+1/-1 values).

    HOW IT WORKS:
    - Start with impulses at each bit instant (delta function train)
    - Convolve with the chosen pulse shape
    - Result: shaped waveform where each bit's pulse doesn't interfere
      with neighbors at the sampling instants (zero ISI)

    Parameters:
    -----------
    bitstream_polar  : numpy array of +1/-1 values (NOT 0/1)
    Tb               : bit duration (seconds)
    samples_per_bit  : samples per bit period
    pulse_type       : 'rect', 'sinc', or 'rc'
    beta             : rolloff factor (only for RC pulse)

    Returns:
    --------
    t        : time axis
    waveform : pulse-shaped waveform
    pulse    : the pulse shape used (for plotting)
    t_pulse  : time axis for the pulse shape
    """
    dt = Tb / samples_per_bit
    N_bits = len(bitstream_polar)

    # Create impulse train: one impulse per bit at t = 0, Tb, 2Tb, ...
    impulse_train = np.zeros(N_bits * samples_per_bit)
    for i, bit in enumerate(bitstream_polar):
        impulse_train[i * samples_per_bit] = bit  # amplitude = +1 or -1

    # Create the pulse shape (duration: ±4*Tb for sinc/RC, Tb for rect)
    if pulse_type == 'rect':
        pulse_duration = Tb
        t_pulse = np.arange(-pulse_duration / 2,
                             pulse_duration / 2, dt)
        pulse = rect_pulse(t_pulse, Tb)

    elif pulse_type == 'sinc':
        # Truncated sinc: we use ±6Tb
        n_periods = 6
        t_pulse = np.arange(-n_periods * Tb, n_periods * Tb, dt)
        pulse = sinc_pulse(t_pulse, Tb)

    elif pulse_type == 'rc':
        n_periods = 6
        t_pulse = np.arange(-n_periods * Tb, n_periods * Tb, dt)
        pulse = raised_cosine_pulse(t_pulse, Tb, beta)

    else:
        raise ValueError(f"Unknown pulse type: '{pulse_type}'. "
                         "Choose from: 'rect', 'sinc', 'rc'")

    # Convolve impulse train with pulse shape.
    # The pulse peak is 1.0 at t=0. The impulse_train contains ±1 at bit starts.
    # After convolution each bit appears as ±p(t - i*Tb), giving correct amplitude.
    waveform_full = np.convolve(impulse_train, pulse, mode='full')

    # Remove the group delay of the (symmetric) pulse: (len(pulse)-1)//2 samples
    delay_samples = (len(t_pulse) - 1) // 2
    waveform = waveform_full[delay_samples: delay_samples + len(impulse_train)]

    # Time axis
    t = np.arange(len(waveform)) * dt

    return t, waveform, pulse, t_pulse


# ==============================================================
# FUNCTION: plot_pulse_shapes — Compare all pulse shapes
# ==============================================================
def plot_pulse_shapes(Tb, samples_per_bit, betas=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """
    Plots all pulse shapes in time and frequency domain for comparison.
    RC vs Sinc comparison.
    """
    dt = Tb / samples_per_bit
    n_periods = 8
    t = np.arange(-n_periods * Tb, n_periods * Tb, dt)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pulse Shape Comparison: Rectangular, Sinc, and Raised Cosine',
                 fontsize=13)

    # Time domain: Sinc vs RC
    ax1 = axes[0, 0]
    p_sinc = sinc_pulse(t, Tb)
    ax1.plot(t / Tb, p_sinc, 'b-', linewidth=2, label='Sinc (β=0)')
    for beta in betas:
        p_rc = raised_cosine_pulse(t, Tb, beta)
        ax1.plot(t / Tb, p_rc, '--', linewidth=1.5,
                 label=f'RC β={beta}')
    ax1.axhline(0, color='k', linewidth=0.5)
    ax1.set_xlabel('t/Tb (normalized time)')
    ax1.set_ylabel('p(t)')
    ax1.set_title('Time Domain: Sinc vs Raised Cosine')
    ax1.set_xlim([-6, 6])
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Mark zero crossings at integer multiples of Tb (Nyquist condition)
    for n in range(-5, 6):
        if n != 0:
            ax1.axvline(x=n, color='gray', linestyle=':', alpha=0.3)

    # Frequency domain: Sinc vs RC
    ax2 = axes[0, 1]
    Lfft = 4096
    faxis = np.fft.fftshift(np.fft.fftfreq(Lfft, dt))
    P_sinc = np.fft.fftshift(np.fft.fft(p_sinc, Lfft))
    ax2.plot(faxis * Tb, np.abs(P_sinc), 'b-', linewidth=2, label='Sinc')
    for beta in betas:
        p_rc = raised_cosine_pulse(t, Tb, beta)
        P_rc = np.fft.fftshift(np.fft.fft(p_rc, Lfft))
        ax2.plot(faxis * Tb, np.abs(P_rc), '--', linewidth=1.5,
                 label=f'RC β={beta}')
    ax2.set_xlabel('f*Tb (normalized frequency)')
    ax2.set_ylabel('|P(f)|')
    ax2.set_title('Frequency Domain: Sinc vs Raised Cosine')
    ax2.set_xlim([-2, 2])
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Nyquist BW')
    ax2.axvline(x=-0.5, color='k', linestyle='--', alpha=0.5)

    # Time domain: Rectangular
    ax3 = axes[1, 0]
    p_rect = rect_pulse(t, Tb)
    ax3.plot(t / Tb, p_rect, 'g-', linewidth=2, label='Rectangular')
    ax3.set_xlabel('t/Tb (normalized time)')
    ax3.set_ylabel('p(t)')
    ax3.set_title('Time Domain: Rectangular Pulse')
    ax3.set_xlim([-3, 3])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Frequency domain: Rectangular
    ax4 = axes[1, 1]
    P_rect = np.fft.fftshift(np.fft.fft(p_rect, Lfft))
    ax4.plot(faxis * Tb, np.abs(P_rect), 'g-', linewidth=2, label='Rectangular (sinc spectrum)')
    ax4.set_xlabel('f*Tb (normalized frequency)')
    ax4.set_ylabel('|P(f)|')
    ax4.set_title('Frequency Domain: Rectangular Pulse (sinc-shaped spectrum)')
    ax4.set_xlim([-5, 5])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
