# Theory

This document explains the physics and signal-processing ideas behind Response-Aware-NTC-DSP, and how each idea maps onto `dsp_pipeline/ntcfinal.py`.

Contents
1. The measurement chain
2. NTC resistance to temperature
3. The voltage divider and ADC
4. The thermal lag model
5. What lag does to a short pulse
6. Inverting the lag, and why it amplifies noise
7. Derivative estimation with Savitzky–Golay filters
8. Sensitivity to an error in τ
9. Estimating τ from a step response
10. Window features and the classifier
11. Reference channel and differential features
12. Streaming detection
13. Evaluation metrics
14. Symbols

---

## 1. The measurement chain

A thermistor reading passes through three stages, and each one distorts the true temperature $T_{\text{true}}(t)$ differently:

```
T_true(t) --> [ thermal lag ] --> T_m(t) --> [ NTC + divider + ADC ] --> raw counts --> [ conversion ] --> T_m estimate
              slow, linear        bead temp     fast, nonlinear, noisy                                 (this is what you log)
```

- **Thermal lag** is slow (seconds) and is the dominant problem for short events.
- **Electrical conversion** is fast (microseconds to milliseconds) but nonlinear and adds noise and quantization.

The method in this repository removes (approximately) the first effect while controlling the noise of the second.

---

## 2. NTC resistance to temperature

A thermistor's resistance falls as temperature rises. The widely used Beta model is

$$\frac{1}{T} = \frac{1}{T_0} + \frac{1}{\beta}\ln\!\left(\frac{R}{R_0}\right)$$

with temperatures in kelvin, $R_0$ the resistance at $T_0$ (usually 25 °C = 298.15 K), and $\beta$ the material constant (3950 K is typical for 10 kΩ beads).

`adc_to_temperature_steinhart_hart()` implements exactly this equation. Despite the function name, it is the **one-parameter Beta equation**, not the full three-coefficient Steinhart–Hart equation. Over a narrow range (say 0–50 °C) the difference is usually small; over wide ranges, or when absolute accuracy matters, use the datasheet's Steinhart–Hart coefficients instead.

The local sensitivity is

$$\frac{dR}{dT} = -\frac{\beta}{T^2}\,R \quad\Rightarrow\quad \text{about } -4.4\%/\text{K at } 25\,^\circ\text{C for } \beta = 3950.$$

---

## 3. The voltage divider and ADC

The assumed circuit has the NTC on the supply side and a fixed resistor $R_f$ to ground:

$$V_{out} = V_{ref}\,\frac{R_f}{R_{ntc} + R_f} \quad\Rightarrow\quad R_{ntc} = R_f\left(\frac{V_{ref}}{V_{out}} - 1\right)$$

The ADC converts $V_{out}$ to counts: $V_{out} = (\text{counts}/\text{adc\_max})\,V_{ref}$. The code clips counts to $[1, \text{adc\_max}-1]$ to avoid division by zero at the rails.

**Sensitivity and resolution.** Near the divider's midpoint ($R_{ntc} = R_f$):

$$\left|\frac{dV_{out}}{dT}\right| \approx \frac{V_{ref}}{4}\,\frac{\beta}{T^2} \approx 36.7\ \text{mV/K at 25 °C}$$

One 12-bit count is $3.3/4095 \approx 0.8$ mV, so the ideal quantization step is about **0.022 °C per count** at 25 °C. Real ESP32 ADC noise is several counts, so the practical resolution is worse. Averaging $N$ readings reduces white noise by $\sqrt{N}$.

**Self-heating.** The divider dissipates $P = I^2 R_{ntc}$ in the bead. The maximum occurs at $R_{ntc}=R_f$: $P_{max} = V_{ref}^2/(4R_f) \approx 0.27$ mW for 10 kΩ at 3.3 V. `inspect_hardware_diagnostics()` estimates this from the data.

---

## 4. The thermal lag model

Treat the bead plus its encapsulation as a single lumped thermal mass $C$ (J/K) coupled to the surroundings through a thermal resistance $R_{th}$ (K/W). The heat balance is

$$C\,\frac{dT_m}{dt} = \frac{T_{\text{true}} - T_m}{R_{th}}$$

Define $\tau = R_{th}C$. Rearranging gives the equation at the heart of this project:

$$\boxed{T_{\text{true}}(t) = T_m(t) + \tau\,\frac{dT_m(t)}{dt}}$$

In the Laplace domain the sensor is a first-order low-pass filter

$$H(s) = \frac{T_m(s)}{T_{\text{true}}(s)} = \frac{1}{1+\tau s}, \qquad |H(j\omega)| = \frac{1}{\sqrt{1+(\omega\tau)^2}}$$

with a $-3$ dB corner at $f_c = 1/(2\pi\tau)$. For $\tau = 2.2$ s, $f_c \approx 0.072$ Hz. Anything changing faster than roughly once per 14 s is attenuated.

**Step response** for a step of size $\Delta T$ at $t_s$:

$$T_m(t) = T_{base} + \Delta T\left(1 - e^{-(t-t_s)/\tau}\right)$$

The sensor reaches 63.2% of the step after $\tau$, and the 10–90% rise time is $\approx 2.2\,\tau$.

> **Assumption:** this is a single time constant. Real sensors often have a second, slower one (coating, leads, mounting). See `limitations.md`.

---

## 5. What lag does to a short pulse

For a rectangular pulse of height $A$ and width $w$ applied to a first-order system, the measured peak (reached at the end of the pulse) is

$$T_{peak} = A\left(1 - e^{-w/\tau}\right)$$

and the response then decays with time constant $\tau$. For $\tau = 2.2$ s:

| Pulse width $w$ | Peak seen by sensor (fraction of $A$) |
|---|---|
| 0.5 s | 20% |
| 1 s | 37% |
| 2 s | 60% |
| 5 s | 90% |
| 10 s | 99% |

So a 1-second, 5 °C pulse appears as a ~1.8 °C bump that is smeared over ~7 seconds. This is the **peak attenuation, onset delay, and temporal smearing** described in the project summary.

**Detectability bound.** If the baseline noise is $\sigma$ and you require the measured peak to exceed $k\sigma$ to detect an event, then

$$A_{min} = \frac{k\sigma}{1-e^{-w/\tau}}.$$

With $\sigma = 0.06$ °C and $k=5$, a 1 s pulse needs $A \gtrsim 0.8$ °C and a 0.5 s pulse needs $A \gtrsim 1.5$ °C (at $\tau = 2.2$ s). Software cannot recover information that is below the noise floor.

---

## 6. Inverting the lag, and why it amplifies noise

The inverse of $H(s)$ is $1 + \tau s$, a **differentiator** with gain

$$|H^{-1}(j\omega)| = \sqrt{1+(\omega\tau)^2},$$

rising at +20 dB/decade. For $\tau = 2.2$ s:

| Frequency | Gain of $H^{-1}$ |
|---|---|
| 0.07 Hz | 1.4 |
| 1 Hz | 14 |
| 5 Hz | 69 |
| 10 Hz (Nyquist at 20 Hz) | 138 |

The signal of interest lives at low frequencies; ADC and quantization noise extends to the Nyquist frequency. Naively computing $T_m + \tau\,\Delta T_m/\Delta t$ with a first difference therefore amplifies noise by up to two orders of magnitude.

**Numerical example.** For white noise of standard deviation $\sigma$ sampled every $\Delta t$, a first difference has noise standard deviation $\sqrt{2}\,\sigma/\Delta t$. With $\sigma=0.06$ °C and $\Delta t=0.05$ s:

- derivative noise: $\approx 1.70$ °C/s,
- after multiplying by $\tau = 2.2$ s: **≈ 3.7 °C of noise** on the compensated signal.

That swamps the 1–5 °C events we want to detect. This is the "false transient triggers" problem. The remedy is a **regularized** inverse: low-pass filter first (or equivalently, estimate the derivative with a smoothing filter), accepting a controlled loss of bandwidth in exchange for noise suppression.

---

## 7. Derivative estimation with Savitzky–Golay filters

A Savitzky–Golay (SG) filter fits a low-order polynomial to a sliding window by least squares and reads the derivative from the fitted polynomial. In `_compute_signal_derivatives_and_comp()`:

| Quantity | Filter | At 20 Hz |
|---|---|---|
| $dT/dt$ and $d^2T/dt^2$ | SG, window ≈ 1.1·fs samples (forced odd), polynomial order 3 | 21 samples (1.05 s) |
| $d(\hat T)/dt$ of the compensated signal | SG, window ≈ 0.6·fs (forced odd), polynomial order 2 | 13 samples |

Noise gain of the derivative (standard deviation in units of $\sigma/\Delta t$, computed from the filter coefficients):

| Estimator | Noise gain |
|---|---|
| First difference | 1.414 |
| SG (window 21, order 3) | 0.091 |
| SG (window 13, order 2) | 0.074 |

The SG window-21 derivative is about **15× quieter** than the first difference. With the example above, the compensation term's noise falls from ≈ 3.7 °C to ≈ 0.24 °C.

**The price:** an SG filter of window length $M$ low-passes the signal. Its effective time resolution is on the order of its window (about 1 s here), so pulses much shorter than that are smoothed. This is the central **noise-versus-bandwidth trade-off** of the method. Longer windows give cleaner derivatives and slower detection; shorter windows are the reverse.

Note also that `savgol_filter` is a **centered (non-causal)** filter. In offline batch mode it uses data on both sides of every point. In streaming mode the estimate at the newest samples relies on edge handling and is less reliable. See `limitations.md`.

---

## 8. Sensitivity to an error in τ

Let the true time constant be $\tau$ and the value used in the model be $\tau_m$. The reconstructed signal is

$$\hat T(t) = T_m + \tau_m \dot T_m = T_{\text{true}} + (\tau_m - \tau)\,\dot T_m.$$

So the reconstruction error is **proportional to the τ error times the measured slope**. A 20% τ error ($\tau_m - \tau = 0.44$ s at $\tau = 2.2$ s) on a 1 °C/s slope yields a 0.44 °C error. Underestimating τ leaves residual lag; overestimating τ causes overshoot. This is why the step-response calibration matters, and why the benchmark has a `--tau-true` option to test mismatch on synthetic data.

---

## 9. Estimating τ from a step response

`calibrate_tau()`:

1. Converts raw counts to °C, regularizes the timestamps onto a uniform grid (§10), and requires ≥ 15 samples.
2. Takes the first 15% of samples as baseline: median gives $T_{base}$, sample standard deviation gives the noise $\sigma$.
3. Finds onset as the first point deviating by more than $\max(3\sigma, 0.25\,^\circ\text{C})$.
4. Fits $T(t) = T_{base} + \Delta T\,(1-e^{-(t-t_s)/\tau})$ for $(t_s,\Delta T,\tau,T_{base})$ with bounded nonlinear least squares (`scipy.optimize.curve_fit`); $\tau\in[0.05, 60]$ s.
5. If the fit fails, falls back to the time at which the response reaches 63.2% of its final change.
6. Reports $\tau$, $\sigma$, the fit RMSE, and the step amplitude.

A fit RMSE close to $\sigma$ supports the single-exponential model; a much larger RMSE indicates extra dynamics.

---

## 10. Preprocessing and window features

**Timestamp regularization** (`_clean_and_regularize_series`): sort by time, drop non-finite values, average duplicate timestamps, and, if the sampling interval varies by more than 5% of its median, linearly interpolate onto a uniform grid. SG filters assume uniform spacing, so this step matters.

**Compensated signal and drift residual:**

$$\hat T = T + \tau\,\dot T, \qquad r(t) = \hat T(t) - \operatorname{median}_{3\,\text{s}}(T)(t).$$

A positive or negative excursion in $r(t)$ means the compensated temperature has departed from the slow baseline.

**Windowing.** The session is cut into windows of `--window-sec` (default 4 s) every `--step-sec` (default 2 s). A window's label is the majority label of its samples.

**Features** (`RESP_FEATURE_NAMES`, computed per window):

| Feature | Meaning |
|---|---|
| `temp_mean` | mean measured temperature |
| `peak_dT_dt` | derivative value at the largest magnitude (signed) |
| `peak_d2T_dt2` | second-derivative value at the largest magnitude |
| `peak_comp_dT_dt` | slope of the compensated signal at its largest magnitude |
| `comp_peak_excursion` | signed largest deviation of $\hat T$ from the window median |
| `peak_drift_residual` | largest magnitude of $r(t)$ |
| `ma_slope` | slope of a moving-average-smoothed temperature across the window |

If a reference channel exists, two differential features are appended (`diff_temp_mean`, `peak_diff_dT_dt`).

**Classifier.** A `RandomForestClassifier` predicts one of `baseline`, `drift`, `heating`, `cooling`. Using a small set of hand-designed physical features keeps the model tiny and interpretable, and it makes the contribution of the response-aware features easy to measure against the baselines below.

---

## 11. Reference channel and differential features

A second NTC measuring the ambient air (shielded from the stimulus) sees the same slow drift as the main sensor but not the local impulse. The difference

$$\Delta T(t) = T_{main}(t) - T_{ref}(t)$$

cancels common-mode ambient change, so a slow drift produces little in $\Delta T$ while a local impulse produces a clear excursion. This helps the classifier separate `drift` from `heating` and `cooling`.

---

## 12. Streaming detection

`StreamingThermalDetector` keeps a rolling buffer, re-evaluates every `step_sec`, and applies:
- a **confidence gate** (`--min-confidence`, default 0.55): predictions below it are ignored;
- **debouncing** (`--debounce-windows`, default 2): a new state must win several consecutive evaluations before it is confirmed.

This suppresses single-window flicker at the cost of latency (see `limitations.md`).

---

## 13. Evaluation metrics

`evaluate_loso()` runs **leave-one-session-out** cross-validation: train on all sessions but one, test on the held-out one, repeat. It compares three pipelines:

| Pipeline | Features |
|---|---|
| `1_Raw_Signal` | window mean and start-to-end change |
| `2_Moving_Average` | smoothed mean, slope, and range |
| `3_Response_Aware` | the full feature set above |

Reported: accuracy, macro-F1 (the unweighted mean of per-class F1), per-class precision/recall/F1, a confusion matrix, and **false alarms per hour**: the number of windows whose true label is `baseline` or `drift` but that were predicted as `heating` or `cooling`, divided by the total duration represented by `windows × step_sec`. Because windows overlap, this is a per-window rate, not a count of distinct false events.

---

## 14. Symbols

| Symbol | Meaning |
|---|---|
| $T_{\text{true}}$ | true temperature at the sensor's surroundings |
| $T_m$ | bead temperature (the measured signal) |
| $\hat T$ | lag-compensated estimate |
| $\tau$ | thermal time constant (s) |
| $\tau_m$ | τ value used by the model |
| $C$, $R_{th}$ | thermal capacitance, thermal resistance |
| $\beta$ | NTC material constant (K) |
| $R_0, T_0$ | nominal resistance at reference temperature |
| $R_f$ | fixed divider resistor |
| $\sigma$ | baseline noise standard deviation (°C) |
| $\Delta t$, $f_s$ | sampling interval, sampling frequency |
