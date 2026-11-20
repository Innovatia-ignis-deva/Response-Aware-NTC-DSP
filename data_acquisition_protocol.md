# Data Acquisition Protocol

This protocol explains how to collect thermistor data that the pipeline in `dsp_pipeline/ntcfinal.py` can use: first to **calibrate the thermal time constant τ**, then to **train and benchmark** the transient detector on real sessions.

> **Status:** The repository ships with no recorded data. All published benchmark numbers come from the built-in synthetic generator. Following this protocol and contributing results is how the method gets validated on real hardware.

Several steps below are tied to specific behaviors of `ntcfinal.py` (for example, `--mode calibrate` treats the first 15% of a step recording as baseline). Those are flagged with **[code]**.

---

## 0. Overview of the workflow

| Phase | Goal | Output | Pipeline command |
|---|---|---|---|
| 1 | Build and bench-check the circuit | verified wiring, ADC range check | none |
| 2 | Measure the noise floor | `noise_floor.csv`, σ | `--mode calibrate` (σ is also reported there) |
| 3 | Measure τ with step responses | ≥ 5 step CSVs per direction, τ ± spread | `--mode calibrate --step-csv` |
| 4 | Record labeled transient sessions | ≥ 5 session CSVs + metadata | none (recording) |
| 5 | Check data quality | diagnostics report | automatic when loading |
| 6 | Benchmark / train | metrics, model | `--mode benchmark` / `--mode train` |

---

## 1. Equipment

**Required**
- ESP32 dev board (any variant with ADC1 pins GPIO32–39)
- 10 kΩ NTC thermistor (nominal R₀ = 10 kΩ at 25 °C, B ≈ 3950 K). Note its exact part number and datasheet B value.
- 10 kΩ fixed resistor, **1% or better, low tempco** (metal film)
- 100 nF ceramic capacitor
- USB cable, computer with Python 3.9+ and `pyserial`
- A stable ambient environment (no HVAC draft, no direct sun)

**Strongly recommended**
- A second identical NTC + resistor divider for an ambient **reference channel** (`ref_raw_adc`). The pipeline uses it to create differential features, which help separate slow ambient drift from real transients.
- A fast reference sensor for later offline comparison (see §9).
- A stopwatch, metronome, or phone timer for scheduled stimuli.

**Stimulus sources** (choose at least one heating and one cooling method; see §7)
- Hair dryer on low, or a heat gun on its lowest setting with a diffuser
- A small cup of warm water (45–55 °C) or ice water, used with the shuttle method
- A cold pack or chilled metal block
- Compressed-air duster (use carefully; inverted cans spray very cold liquid)

---

## 2. Circuit

```
3V3 ----[ NTC ]----+----[ R_fixed = 10k ]---- GND
                   |
                   +---- GPIO34 (ADC1_CH6)
                   |
                 [100 nF] to GND, close to the pin
```

The NTC is on the **3V3 side**, so the output voltage **rises with temperature**. This is the topology assumed by `adc_to_temperature_steinhart_hart()` **[code]**. If you wire it the other way round, temperatures will come out inverted.

For the optional reference channel, build a second identical divider on GPIO34's neighbor, GPIO35.

**Wiring and mounting rules**
1. Use ADC1 pins only. ADC2 conflicts with Wi-Fi and is less reliable.
2. Keep Wi-Fi and Bluetooth **off** during recording.
3. Keep wires short (< 30 cm) and twisted or shielded if longer. Do not let leads move during a run.
4. Mount the bead the same way every time (see §6). Mounting changes τ.
5. Power from the laptop USB port or a clean 5 V supply; avoid noisy hubs.

### Expected ADC counts (sanity check)

With 10 kΩ / 10 kΩ and B = 3950 K, ideal 12-bit counts are roughly:

| Temperature | R_NTC | V_out | ≈ Counts |
|---|---|---|---|
| 0 °C | 33.6 kΩ | 0.76 V | 940 |
| 10 °C | 20.2 kΩ | 1.09 V | 1360 |
| 25 °C | 10.0 kΩ | 1.65 V | 2047 |
| 40 °C | 5.3 kΩ | 2.16 V | 2680 |
| 50 °C | 3.6 kΩ | 2.43 V | 3015 |
| 60 °C | 2.5 kΩ | 2.64 V | 3280 |

The ESP32 ADC is not ideal, so expect deviations. Two consequences:
- At 11 dB attenuation the manufacturer's accurate range is roughly 0.15–2.45 V. With this divider that corresponds to about **0–50 °C**. Above that, readings compress and become nonlinear. If you need to test hotter stimuli, either keep the bead's temperature below ~50 °C or choose a smaller R_fixed (and pass the matching value, see §10).
- If any raw count is ≤ 0 or ≥ 4095 the pipeline logs an "ADC rail clipping" warning **[code]**. Treat clipped sessions as invalid.

---

## 3. Firmware and capture

1. Install the Arduino IDE (or PlatformIO) and the ESP32 board package.
2. Put `esp32_logger.ino` inside a folder named `esp32_logger` (the Arduino IDE requires the folder and the sketch to share a name). Upload it to the board.
3. Leave the defaults (20 Hz, 16× oversampling) for your first runs. 20 Hz matches the pipeline's default time step of 0.05 s **[code]**.

**Capturing a session.** Do not copy and paste from the Arduino Serial Monitor; use a script so the file is a clean CSV:

```python
# tools/capture.py  --  usage: python capture.py COM5 data/raw/20250101_A_session01_raw.csv
import sys, time, serial

port, out = sys.argv[1], sys.argv[2]
with serial.Serial(port, 115200, timeout=1) as ser, open(out, "w", newline="") as f:
    time.sleep(2)                 # the ESP32 resets when the port opens
    ser.reset_input_buffer()
    ser.write(b"g")               # start recording
    started = False
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("timestamp_ms"):
                started = True
            if started and line:
                f.write(line + "\n")
    except KeyboardInterrupt:
        ser.write(b"x")           # stop on Ctrl+C
```

Timestamps restart at 0 for every recording. Each row's `timestamp_ms` is the actual sampling time, so jitter is measurable.

---

## 4. Phase 2: Warm-up and noise floor

1. Power the board and let it sit **10 minutes** before recording. The ESP32 and the divider drift while warming up.
2. Place the sensor in still ambient air, shielded from drafts (a small cardboard box with a vent hole works).
3. Record **5 minutes** with no stimulus. Label is `none`.
4. Compute σ (standard deviation of the converted temperature after removing a linear trend). Typical well-behaved results are 0.03–0.15 °C with 16× oversampling.

The step-response calibration also reports baseline noise. A noise floor much above ~0.2 °C usually means a wiring, grounding, or supply problem; fix this before going further.

Keep this file as `data/raw/noise_floor_<date>.csv`.

---

## 5. Phase 3: Step-response calibration (measuring τ)

τ is the single most important parameter. It must be measured **in the same medium, airflow, and mounting** as the transient sessions. A bead that has τ ≈ 2 s in still air can have τ well below 1 s in moving air or water.

### 5.1 How the code uses the file **[code]**

`calibrate_tau()` in `ntcfinal.py`:
- needs `timestamp_ms` and either `temp_c` or `raw_adc`, and **at least 15 valid samples**;
- uses the **first 15% of the recording** as the baseline for the starting temperature and noise estimate;
- detects step onset as the first point deviating from that baseline by more than max(3σ, 0.25 °C);
- fits `T(t) = T_base + ΔT·(1 − exp(−(t − t_start)/τ))` with bounded nonlinear least squares (τ constrained to 0.05–60 s);
- if the fit fails, falls back to the 63.2% rise-time estimate;
- initializes ΔT from the **last sample**, so the recording must end on a settled value.

### 5.2 Recording recipe

| Segment | Duration | Requirement |
|---|---|---|
| Pre-step baseline | ≥ 15 s (and ≥ 15% of the total length) | sensor at rest, equilibrated |
| Step | ≤ 0.3 s | apply the new temperature as fast as you can |
| Hold | ≥ 5τ, recommended **30–60 s** | do not touch or move anything; end settled |

Total length: **60–90 s**. If you expect τ ≈ 2 s, 5τ = 10 s, so 30 s of hold is a comfortable margin.

**Step size:** 5–15 °C. Larger steps give a better signal-to-noise ratio but increase nonlinearity (convection and the NTC curve). Keep the final temperature within the 0–50 °C ADC-safe range.

**Ways to make a fast step** (best to worst reproducibility)
1. **Shuttle method:** mount the bead on a stick or in a small clamp. Move it in under 0.3 s from ambient air into a still warm-air zone (just above a mug of warm water) or a cold zone (above ice water, not touching). Use the same motion every time.
2. **Heated or cooled air jet:** hair dryer on low aimed through a card with a flap that you open to start the step. Use the same distance each time.
3. **Direct contact:** touching a warm or cold block. This measures contact response, not air response, so only use it if that is your deployment case.

### 5.3 Repetitions

- Record **at least 5 steps in each direction** (5 heating, 5 cooling), letting the sensor fully re-equilibrate (≥ 60 s) between runs.
- Run `--mode calibrate` on each:

```bash
python dsp_pipeline/ntcfinal.py --mode calibrate \
    --step-csv data/raw/step_heat_01.csv --out-json data/calibration/step_heat_01.json
```

- Report τ as **mean ± standard deviation**, separately for heating and cooling.

### 5.4 Acceptance criteria

| Check | Pass |
|---|---|
| Repeatability of τ across runs | within about ±10% |
| Fit RMSE (printed as "Step fit RMSE") | within roughly 2–3× the noise σ |
| Step amplitude | ≥ 5 °C and ≥ 20× σ |
| Hold ends settled | final 5 s change < 2σ |
| Heating τ vs cooling τ | similar; differences > 20% should be reported |

If the RMSE is much larger than the noise, the response is not a single exponential (a bead plus its coating plus mounting often behaves as two time constants). The first-order model, and therefore the compensation, will be only approximate. Record this in your results. It is useful scientific information, not a failure of the protocol.

Use the mean τ as `--tau` in later steps.

---

## 6. Mounting, medium, and environment (control and record these)

τ and self-heating depend heavily on how the sensor is installed. For every session, keep these constant and log them in the metadata file:

- **Mounting:** free-hanging in air on thin leads, glued to a surface, or embedded. Do not change between calibration and sessions.
- **Medium and airflow:** still air, fan-driven air, or liquid. Record an approximate airflow (m/s) if you can measure it.
- **Orientation** of the bead relative to the stimulus.
- **Ambient temperature** at the start and end of the session.
- **Enclosure:** protective box, shield, or open bench.

---

## 7. Phase 4: Transient sessions (the labeled dataset)

### 7.1 Label definitions

The classifier uses four labels (`LABELS` in `ntcfinal.py`):

| Label | Meaning (physical stimulus, not the shape of the curve) |
|---|---|
| `baseline` | No stimulus; ambient is stable |
| `drift` | Slow ambient change of roughly 0.4–1.2 °C over a few seconds |
| `heating` | A brief heating impulse is applied (about 0.5–1.5 s long) |
| `cooling` | A brief cooling impulse is applied (about 0.5–1.5 s long) |

**Important convention:** labels describe what was **physically done to the sensor**, not what the measured curve looks like. After a heating impulse ends, the bead keeps cooling for several τ and the raw curve decays like a cooling transient. That decay tail is labeled `baseline`, because the true temperature has already returned to ambient. This is exactly the situation the response-aware features are meant to resolve.

This convention mirrors the synthetic generator. If you adopt a different convention, state it in your dataset documentation.

### 7.2 Slot-based design (recommended)

The pipeline classifies **4-second windows with a 2-second step** by default, and each window's label is the **majority** label of its samples **[code]**. To keep labels clean:

- Divide the session into **4-second slots**, starting at t = 0.
- In a heating or cooling slot, start the stimulus **~1 s after the slot starts** and hold it for **~1 s**.
- Label the entire slot with its class.
- Between events, insert **baseline recovery time ≥ 5τ** (about 11 s for τ = 2.2 s) so that the raw signal returns to its starting level. Label this recovery as `baseline`.
- Randomize the order of event types (the synthetic generator shuffles them).

Because the window step (2 s) is half the window length, some windows will straddle two slots. That is expected and slightly lowers accuracy; do not try to avoid it by adjusting labels afterwards.

### 7.3 Recommended dataset size

To match the scale of the synthetic benchmark and give leave-one-session-out cross-validation meaning:

| Item | Minimum | Better |
|---|---|---|
| Sessions | 5 | 10+ |
| Events per class per session | 12 | 20+ |
| Session length (with recovery) | ~12–15 min | 20 min |
| Sessions on different days / ambient conditions | 2 days | 3+ |
| Different physical sensors | 1 | 2+ (to test transfer) |

Record a few extra sessions with **shorter recovery time** (back-to-back events) as a separate stress set.

### 7.4 Making the schedule

Generate a schedule once, print it or play audio cues, and run through it. Example generator:

```python
# tools/make_schedule.py
import random, csv
random.seed(7)
LABELS = ["baseline", "drift", "heating", "cooling"]
events = LABELS * 12
random.shuffle(events)
t = 20.0                       # initial settling baseline
rows = [(0.0, t, "baseline")]
for lbl in events:
    rows.append((t, t + 4.0, lbl))
    t += 4.0
    if lbl in ("heating", "cooling", "drift"):
        rows.append((t, t + 11.0, "baseline"))   # >= 5*tau recovery
        t += 11.0
with open("schedules/session01_schedule.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["start_s", "end_s", "label"]); w.writerows(rows)
print(f"Total duration: {t/60:.1f} min")
```

Keep schedule files **outside** the session data directory (see §8).

### 7.5 Stimulus delivery

- **Heating impulse:** open the card flap on a low hair dryer for ~1 s, or shuttle the bead over a warm water mug for ~1 s. Aim for a stimulus of roughly **2–6 °C above ambient at the sensor**.
- **Cooling impulse:** short burst from a cold source (chilled block, shuttle over ice water, brief duster puff at a safe distance). Roughly 2–6 °C below ambient.
- **Drift:** slowly move a warm or cold object near the sensor, or slowly open a door to a cooler room. Aim for a 0.4–1.2 °C change over about 2–4 s.
- **Baseline:** do nothing.

Use audio or visual countdown cues to time the stimulus to the schedule. Log the actual stimulus start in your notes if you deviate.

### 7.6 Applying labels after recording

Record with label `none` (the default in the firmware), then apply labels from the schedule. Hand-typed labels have around a second of human delay, which matters when the events themselves last about a second.

```python
# tools/apply_labels.py  --  usage: python apply_labels.py raw.csv schedule.csv out.csv [offset_s]
import sys, pandas as pd
raw, sched, out = sys.argv[1:4]
offset = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # seconds between recording start and schedule start
df = pd.read_csv(raw)
s = pd.read_csv(sched)
t = df["timestamp_ms"] / 1000.0 - offset
df["label"] = "none"
for _, r in s.iterrows():
    df.loc[(t >= r.start_s) & (t < r.end_s), "label"] = r.label
df.to_csv(out, index=False)
print(df["label"].value_counts())
```

Rows left as `none` are dropped by the loader with a warning **[code]**.

### 7.7 Session metadata

Write one JSON file per session (same base name, `.meta.json`):

```json
{
  "session_id": "20250101_A_session01",
  "date": "2025-01-01",
  "operator": "",
  "board": "ESP32-DevKitC",
  "firmware": "esp32_logger.ino, 20 Hz, 16x oversample, ADC_11db",
  "ntc_part_number": "",
  "ntc_beta_k": 3950,
  "ntc_r0_ohm": 10000,
  "r_fixed_ohm": 10000,
  "v_ref_v": 3.3,
  "mounting": "free-hanging on 30 cm leads",
  "medium": "still air",
  "airflow_m_s": null,
  "ambient_start_c": null,
  "ambient_end_c": null,
  "heating_stimulus": "hair dryer low, card flap, ~1 s",
  "cooling_stimulus": "chilled aluminum block, ~1 s",
  "tau_used_s": null,
  "reference_channel": false,
  "schedule_file": "schedules/session01_schedule.csv",
  "notes": ""
}
```

---

## 8. File format and layout

**Session CSV columns**

| Column | Required | Description |
|---|---|---|
| `timestamp_ms` | yes | integer milliseconds since recording start |
| `raw_adc` | yes (or `temp_c`) | ADC counts, floats allowed (e.g., 2047.31) |
| `label` | yes for benchmark/train | one of `baseline`, `drift`, `heating`, `cooling` |
| `ref_raw_adc` | optional | reference NTC counts (or `ref_temp_c` in °C) |

**Suggested layout**

```
data/
├── raw/            # untouched captures, never edited
├── sessions/       # labeled CSVs ONLY (what --data-dir points to)
├── calibration/    # step CSVs + JSON outputs
├── metadata/       # *.meta.json
└── README.md
schedules/          # schedule CSVs (kept out of data/sessions)
```

> **Warning [code]:** `load_session_csvs()` loads **every** `*.csv` in `--data-dir` and demands a `label` column and a `timestamp_ms` column. Put nothing else (schedules, calibration files, noise files) in that directory.

Never overwrite raw captures. All processing must produce new files.

---

## 9. Optional: reference sensor for ground truth

The pipeline's `ref_temp_c` is an **ambient reference NTC** used for differential features. It is not a ground-truth fast sensor. To check how well the compensation recovers the true waveform, you can additionally log a faster sensor that sees the same stimulus:

- A fine-wire bare thermocouple junction typically responds faster than a bead NTC (confirm by measuring its own step response). Common thermocouple amplifier breakouts convert at only a few to ~10 samples per second, so also check that it is fast enough for your impulse width.
- Place it as close as possible to the NTC bead, in the same flow.
- Log it with its own timestamps, and sync the two recordings with a sharp stimulus (e.g., the first card-flap opening) used as a shared time mark.

Ground-truth comparison scripts are not part of the repository yet; this is on the roadmap.

---

## 10. Phase 5: Quality checks before analysis

Run `--mode calibrate` or load the data; `inspect_hardware_diagnostics()` reports **[code]**:
- ADC clipping at the low and high rails
- non-monotonic and duplicate timestamps
- median sampling interval and jitter (standard deviation)
- worst-case self-heating power estimate

**Checklist**

| Check | Target |
|---|---|
| Rail clipping | 0 samples |
| Non-monotonic or duplicate timestamps | 0 |
| Median dt | ≈ 0.050 s at 20 Hz |
| dt jitter std | < 5% of dt (the pipeline resamples onto a uniform grid if jitter exceeds 5%) |
| Self-heating power | see §11 |
| Labels | every row valid or `none`; class counts roughly balanced |
| Baseline noise σ | similar to the §4 noise floor |

> **Heads-up [code]:** The command-line interface exposes `--adc-max`, `--v-ref`, `--window-sec`, `--step-sec`, and `--tau`, but **not** the thermistor β, R₀, or R_fixed. If your parts differ from 10 kΩ / 10 kΩ / β = 3950, edit the defaults of `adc_to_temperature_steinhart_hart()` (and `inspect_hardware_diagnostics()`), or convert to `temp_c` yourself before saving the CSV.

---

## 11. Self-heating check

The divider dissipates power in the bead, which warms it above ambient. With 10 kΩ / 10 kΩ at 3.3 V the worst case (R_NTC = R_fixed) is

$$P_{max} = \frac{V^2}{4R} = \frac{3.3^2}{4 \times 10^4} \approx 0.27\ \text{mW}.$$

For a small bead with a dissipation constant on the order of 1–2 mW/°C in still air, that is a self-heating offset of roughly 0.1–0.3 °C (check your datasheet). This offset **changes with airflow**, which can mimic a cooling transient. Mitigations:

1. Use a larger R_fixed or lower supply to cut the current (at the cost of sensitivity and ADC range).
2. Power the divider from a GPIO and enable it only during sampling.
3. If duty-cycled, pass `--duty-cycle` so the diagnostic estimate reflects it **[code]**.
4. Keep the airflow condition the same between calibration and sessions.

---

## 12. Phase 6: Running the pipeline

```bash
# 1) Estimate tau (repeat for every step file, then average)
python dsp_pipeline/ntcfinal.py --mode calibrate --step-csv data/raw/step_heat_01.csv

# 2) Leave-one-session-out benchmark on real data
python dsp_pipeline/ntcfinal.py --mode benchmark --data-dir data/sessions \
    --tau 2.4 --out-json results/benchmark.json --plot-out results/benchmark.png

# 3) Sensitivity of results to tau error: repeat with tau at -20%, -10%, +10%, +20%
python dsp_pipeline/ntcfinal.py --mode benchmark --data-dir data/sessions --tau 1.9

# 4) Train and save a model
python dsp_pipeline/ntcfinal.py --mode train --data-dir data/sessions \
    --tau 2.4 --save-model thermal_model.joblib

# 5) Predict on a new unlabeled session (needs timestamp_ms and raw_adc or temp_c)
python dsp_pipeline/ntcfinal.py --mode predict --load-model thermal_model.joblib \
    --input-csv data/new_session.csv
```

When reporting results, include: τ (mean ± std, heating and cooling), noise σ, number of sessions and events, the confusion matrix, per-class precision/recall, and the tau-sensitivity sweep.

---

## 13. Common pitfalls

- **The ESP32 resets when the serial port is opened.** Wait 2 s before sending `g` (the capture script does).
- **Labels typed by hand are late.** Use the schedule method.
- **Hot sensor drift after power-up.** Warm up 10 minutes.
- **Touching the leads.** Hands heat the wires and conduct heat to the bead.
- **Breath and body heat** near the sensor create spurious transients.
- **Changing mounting or airflow between calibration and sessions** invalidates τ.
- **A step recording that does not end settled** makes the amplitude estimate unreliable.
- **Putting non-session CSVs in `data/sessions/`** crashes the loader.
- **Saturating the ADC** with stimuli hotter than about 50 °C (at this divider).

---

## 14. Safety

- Hair dryers and heat guns can burn. Keep the stimulus at the minimum needed (a few °C above ambient at the sensor).
- Do not use mains-powered heating equipment near wet setups.
- Compressed-air dusters can spray very cold liquid. Wear eye protection.
- This project is **not a medical device**. Do not use it for patient measurement or any safety-critical decision.

---

## 15. Contributing data

If you share data, please include the session CSVs, the metadata JSON, the schedule, the calibration JSONs, and the exact command lines and software versions used. Cite the repository (see `CITATION.cff`).
