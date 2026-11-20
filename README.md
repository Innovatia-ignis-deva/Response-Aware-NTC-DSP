# Response-Aware NTC Thermal Transient Detection on Edge Hardware

**Ideated by Devashrit Panda**

> Real-time detection and deconvolution of rapid, short-duration thermal transients from low-cost Negative Temperature Coefficient (NTC) thermistors on ESP32 edge microcontrollers.

---

## Overview

Low-cost NTC thermistors suffer from an inherent first-order thermal time constant ($\tau$) caused by the thermal mass of the sensing bead and encapsulation. When exposed to brief thermal impulses—such as micro-convective airflow shifts, localized skin/tissue contact spikes, or transient heat pulses—the raw ADC voltage curve severely attenuates peak amplitude and smears temporal onset.

**Response-Aware-NTC-DSP** overcomes this hardware bottleneck purely in software. By modeling the sensor's thermal transfer function in discrete time and applying regularized inverse lag compensation on the ESP32, the pipeline reconstructs rapid thermal events in real time without requiring expensive fast-response micro-thermocouples or external analog differentiator circuits.

---

## System Architecture

```mermaid
flowchart LR
    A[Low-Cost NTC Thermistor<br/>Voltage Divider] --> B[ESP32 12-bit ADC<br/>Oversampling & Steinhart-Hart]
    B --> C[Circular Ring Buffer<br/>& Savitzky-Golay / EMA Gate]
    C --> D[Inverse Thermal Lag<br/>Deconvolution Filter]
    D --> E[Adaptive Baseline &<br/>Transient Event Trigger]
