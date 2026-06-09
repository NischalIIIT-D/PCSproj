# End-to-End Digital Communication System Simulator

A modular Python-based end-to-end baseband digital communication system. Covers PCM sampling &amp; quantization, line codes (NRZ/RZ/Bipolar), raised cosine pulse shaping, AWGN channel, matched filter receiver, and BER vs SNR waterfall curve.

## Features

- Signal generation using configurable sinusoidal sources
- Uniform sampling and quantization
- PCM encoding
- Line coding schemes:
  - Polar NRZ
  - Unipolar NRZ
  - Bipolar AMI
- Pulse shaping:
  - Rectangular Pulse
  - Raised Cosine Pulse
- AWGN channel simulation
- Matched filter receiver
- Bit Error Rate (BER) computation
- BER vs SNR (Waterfall Curve) analysis

## System Architecture

```text
Analog Signal
      │
      ▼
Sampling
      │
      ▼
Quantization
      │
      ▼
PCM Encoding
      │
      ▼
Line Coding
      │
      ▼
Pulse Shaping
      │
      ▼
AWGN Channel
      │
      ▼
Matched Filter
      │
      ▼
Bit Detection
      │
      ▼
BER Calculation
