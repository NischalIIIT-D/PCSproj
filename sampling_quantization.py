import numpy as np
import matplotlib.pyplot as plt


# ==============================================================
# FUNCTION 1: uniquan — Uniform Quantizer
# (Direct Python translation of the MATLAB uniquan.m from textbook)
# ==============================================================
def uniquan(sig_in, L):
    """
    Uniform quantizer — translates the textbook's uniquan.m to Python.

    HOW IT WORKS:
    - Find the max and min of the signal → defines the range
    - Divide this range into L equal levels
    - Each sample is rounded to the nearest level
    - SQNR = 20 * log10(||signal|| / ||noise||)

    Parameters:
        sig_in : numpy array — input signal samples
        L      : int — number of quantization levels (must be power of 2, e.g. 4, 8, 16)

    Returns:
        q_out  : quantized signal (same shape as sig_in)
        Delta  : quantization step size
        SQNR   : Signal-to-Quantization-Noise Ratio in dB
    """
    sig_pmax = np.max(sig_in)       # positive peak
    sig_nmax = np.min(sig_in)       # negative peak (most negative value)
    Delta = (sig_pmax - sig_nmax) / L   # step size

    # Define the L quantization levels: nmax+Delta/2, nmax+3*Delta/2, ...
    q_level = np.arange(sig_nmax + Delta/2, sig_pmax, Delta)

    # Convert input to indices in [0.5, L+0.5] range, then round
    sigp = (sig_in - sig_nmax) / Delta + 0.5   # shift to [0.5, L+0.5]
    qindex = np.round(sigp).astype(int)          # round to 1,2,...,L
    qindex = np.clip(qindex, 1, L) - 1          # clip and convert to 0-indexed

    q_out = q_level[qindex]   # map index back to quantization level values

    # SQNR calculation
    noise = sig_in - q_out
    if np.linalg.norm(noise) == 0:
        SQNR = np.inf
    else:
        SQNR = 20 * np.log10(np.linalg.norm(sig_in) / np.linalg.norm(noise))

    return q_out, Delta, SQNR


# ==============================================================
# FUNCTION 2: SamplenQuant — THE MAIN MODULE FUNCTION
# ==============================================================
def SamplenQuant(signal, t, fs_original, fs_new, num_bits):
    """
    SamplenQuant: Samples a continuous signal and quantizes it to bits.

    This is the key module. Given any input signal, it:
      1. Downsamples from fs_original to fs_new
      2. Quantizes each sample to num_bits bits
      3. Returns the bitstream (PCM output)

    HOW SAMPLING WORKS HERE:
    - The signal is already a discrete array sampled at fs_original Hz
    - To resample at fs_new Hz, we take every Nfactor = fs_original/fs_new th sample
    - This is called downsampling

    Parameters:
    -----------
    signal      : numpy array — input signal (sampled at fs_original)
    t           : numpy array — time axis corresponding to signal
    fs_original : float — original sampling frequency (Hz)
    fs_new      : float — desired new sampling frequency (Hz)
    num_bits    : int — bits per sample (L = 2^num_bits quantization levels)

    Returns:
    --------
    t_sampled      : time instants of samples
    s_sampled      : sampled signal (before quantization)
    s_quantized    : quantized signal values (after quantization)
    s_zoh          : zero-order hold signal (staircase — what PCM looks like in time)
    bitstream      : numpy array of 0s and 1s (the PCM bitstream)
    Delta          : quantization step size
    SQNR           : Signal-to-Quantization-Noise Ratio in dB
    L              : number of quantization levels used
    """
    # --- Input validation ---
    Nfactor = fs_original / fs_new
    assert Nfactor == int(Nfactor), \
        f"fs_original/fs_new must be an integer! Got {Nfactor}"
    Nfactor = int(Nfactor)

    L = 2 ** num_bits   # number of quantization levels

    # --- STEP 1: SAMPLING ---
    # Take every Nfactor-th sample (downsample)
    s_sampled = signal[::Nfactor]
    t_sampled = t[::Nfactor]

    # --- STEP 2: QUANTIZATION ---
    s_quantized, Delta, SQNR = uniquan(s_sampled, L)

    # --- STEP 3: ZERO-ORDER HOLD (ZOH) ---
    # In PCM, between sample instants, the signal holds its last value
    # This creates the staircase (ZOH) waveform
    p_zoh = np.ones(Nfactor)                        # rectangular pulse of width Ts
    s_zoh_short = np.kron(s_quantized, p_zoh)        # repeat each sample Nfactor times
    s_zoh = s_zoh_short[:len(signal)]               # trim to original length

    # --- STEP 4: CONVERT QUANTIZED LEVELS TO BITS (PCM encoding) ---
    # Find the index of each quantized sample in the quantization levels
    sig_nmax = np.min(s_sampled)
    level_indices = np.round((s_quantized - sig_nmax) / Delta - 0.5).astype(int)
    level_indices = np.clip(level_indices, 0, L - 1)

    # Convert each level index to a binary number with num_bits bits
    bitstream = []
    for idx in level_indices:
        bits = [(idx >> (num_bits - 1 - b)) & 1 for b in range(num_bits)]
        bitstream.extend(bits)
    bitstream = np.array(bitstream)

    return t_sampled, s_sampled, s_quantized, s_zoh, bitstream, Delta, SQNR, L


# ==============================================================
# FUNCTION 3: bits_to_signal — Decode bits back to signal (for reconstruction)
# ==============================================================
def bits_to_signal(bitstream, num_bits, Delta, sig_nmax):
    """
    Converts PCM bitstream back to quantized amplitude values.
    (This is the decoder — used at the receiver side)

    Parameters:
        bitstream : numpy array of 0s and 1s
        num_bits  : bits per sample
        Delta     : quantization step size
        sig_nmax  : minimum value of original signal (needed to reconstruct levels)

    Returns:
        reconstructed : numpy array of amplitude values
    """
    n_samples = len(bitstream) // num_bits
    reconstructed = []
    for i in range(n_samples):
        bits = bitstream[i * num_bits: (i + 1) * num_bits]
        # Convert binary to integer index
        idx = int(''.join(map(str, bits.astype(int))), 2)
        # Convert index back to amplitude
        amplitude = sig_nmax + (idx + 0.5) * Delta
        reconstructed.append(amplitude)
    return np.array(reconstructed)


# ==============================================================
# FUNCTION 4: plot_sampling_quantization — Visualization
# ==============================================================
def plot_sampling_quantization(t, signal, t_sampled, s_sampled,
                                s_quantized, s_zoh, SQNR, L, fs_new, num_bits,
                                title_prefix=""):
    """
    Plots the original signal, sampled signal, and quantized (PCM) signal.
    Also shows the frequency spectrum.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"{title_prefix}Sampling & Quantization | fs={fs_new}Hz | "
                 f"L={L} levels ({num_bits} bits) | SQNR={SQNR:.2f} dB", fontsize=13)

    # Plot 1: Original vs Sampled
    ax1 = axes[0]
    ax1.plot(t, signal, 'k-', linewidth=1.5, label='Original signal g(t)')
    ax1.stem(t_sampled, s_sampled, linefmt='b-', markerfmt='bo',
             basefmt='k-', label='Samples')
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Signal and its Uniform Samples')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Original vs ZOH (staircase PCM signal)
    ax2 = axes[1]
    ax2.plot(t, signal, 'k--', linewidth=1.5, label='Original signal')
    ax2.plot(t[:len(s_zoh)], s_zoh, 'b-', linewidth=1.5,
             label=f'PCM signal ({L}-level ZOH)')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Original Signal vs {L}-level PCM Signal (Zero-Order Hold)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Quantization levels visualization
    ax3 = axes[2]
    ax3.plot(t_sampled, s_sampled, 'k-o', markersize=4, label='Samples')
    ax3.plot(t_sampled, s_quantized, 'r-s', markersize=4, label='Quantized')
    for i, (ts, sq) in enumerate(zip(t_sampled, s_quantized)):
        ax3.vlines(ts, s_sampled[i], sq, colors='gray', linestyles='dotted', alpha=0.5)
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Quantization Error Visualization (gray lines = error)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrum(t, signal, s_zoh, fs_original, fs_new, title_prefix=""):
    """
    Plots frequency spectra of original and sampled/quantized signal.
    """
    N = len(signal)
    Lfft = int(2 ** np.ceil(np.log2(N) + 1))
    Fmax = 1 / (2 * (t[1] - t[0]))
    Faxis = np.linspace(-Fmax, Fmax, Lfft)

    Xsig = np.fft.fftshift(np.fft.fft(signal, Lfft))
    Xzoh = np.fft.fftshift(np.fft.fft(s_zoh[:len(signal)], Lfft))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f"{title_prefix}Frequency Spectra", fontsize=13)

    axes[0].plot(Faxis, np.abs(Xsig))
    axes[0].set_title('Spectrum of Original Signal g(t)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_xlim([-150, 150])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Faxis, np.abs(Xzoh))
    axes[1].set_title(f'Spectrum of Sampled+Quantized Signal (fs={fs_new} Hz)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_xlim([-150, 150])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
