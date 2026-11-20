/*
  esp32_logger.ino
  Response-Aware-NTC-DSP  |  ESP32 NTC data logger
  Ideated and developed by Devashrit Panda  |  Apache License 2.0

  PURPOSE
    Samples an NTC thermistor voltage divider at a fixed rate and streams
    CSV over USB serial in the exact format that dsp_pipeline/ntcfinal.py expects:

        timestamp_ms,raw_adc,label                    (default)
        timestamp_ms,raw_adc,ref_raw_adc,label        (if USE_REF_CHANNEL = 1)

  WIRING (matches the defaults in ntcfinal.py: r_fixed = 10 kOhm, NTC 10 kOhm @ 25 C)

        3V3 ----[ NTC ]----+----[ R_fixed 10k ]---- GND
                           |
                           +---- GPIO34 (ADC1_CH6)
                           |
                         [100 nF] to GND   (place close to the ESP32 pin)

    The NTC is on the 3V3 side and the fixed resistor is on the GND side, so the
    measured voltage RISES as temperature rises. ntcfinal.py assumes this topology.

    Optional reference NTC (ambient, shielded from the stimulus) uses an identical
    divider on GPIO35 (ADC1_CH7). Set USE_REF_CHANNEL to 1 to log it.

    Use ADC1 pins only (GPIO 32-39). ADC2 pins conflict with Wi-Fi and are less
    reliable. GPIO34/35/36/39 are input-only, which is fine here.

  SERIAL COMMANDS (115200 baud; send a single character, no newline needed)
    g   start recording (prints the CSV header, timestamps restart at 0)
    x   stop recording
    b   set label = baseline
    d   set label = drift
    h   set label = heating
    c   set label = cooling
    n   set label = none   (default; rows with label "none" are ignored by ntcfinal.py)
    ?   print help (only while not recording)

  NOTES
    - While recording, the sketch prints ONLY CSV rows, so a capture is a clean CSV.
      Status lines (prefixed with '#') are printed only when not recording.
    - Typing labels by hand has ~1 s of human delay. For publishable data, prefer
      logging with label "none" and applying labels afterwards from a timed schedule
      (see docs/data_acquisition_protocol.md).
    - timestamp_ms is the actual time the sample was taken (not the ideal tick time),
      so timing jitter is visible to inspect_hardware_diagnostics() in ntcfinal.py.
    - 64-bit timer: no rollover issues for any realistic session length.
*/

#include <Arduino.h>
#include <ctype.h>
#include "esp_timer.h"

// ===================== USER CONFIGURATION =====================
#define NTC_PIN          34      // ADC1_CH6
#define USE_REF_CHANNEL  0       // 1 = also log a reference NTC on REF_PIN
#define REF_PIN          35      // ADC1_CH7

static const uint32_t BAUD_RATE        = 115200;
static const uint32_t SAMPLE_RATE_HZ   = 20;   // 20 Hz matches the pipeline's default dt = 0.05 s
static const uint8_t  OVERSAMPLE_COUNT = 16;   // back-to-back reads averaged per sample (~1 ms total)
// ==============================================================

static const int64_t SAMPLE_PERIOD_US = 1000000LL / SAMPLE_RATE_HZ;

static bool        recording    = false;
static int64_t     t0_us        = 0;
static int64_t     next_tick_us = 0;
static const char *currentLabel = "none";

// Average several back-to-back conversions. Output is a float so the sub-count
// resolution gained by averaging is preserved (e.g. 2047.31).
static float readAveraged(uint8_t pin) {
  uint32_t acc = 0;
  for (uint8_t i = 0; i < OVERSAMPLE_COUNT; i++) {
    acc += analogRead(pin);
  }
  return (float)acc / (float)OVERSAMPLE_COUNT;
}

static void printHelp() {
  Serial.println("# esp32_logger ready");
  Serial.println("# g=start x=stop | labels: b=baseline d=drift h=heating c=cooling n=none");
  Serial.printf("# rate=%lu Hz, oversample=%u, ref_channel=%d\n",
                (unsigned long)SAMPLE_RATE_HZ, (unsigned)OVERSAMPLE_COUNT, (int)USE_REF_CHANNEL);
}

static void startRecording() {
#if USE_REF_CHANNEL
  Serial.println("timestamp_ms,raw_adc,ref_raw_adc,label");
#else
  Serial.println("timestamp_ms,raw_adc,label");
#endif
  t0_us        = esp_timer_get_time();
  next_tick_us = t0_us;
  recording    = true;
}

static void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)tolower((unsigned char)Serial.read());
    switch (c) {
      case 'g': if (!recording) startRecording(); break;
      case 'x': recording = false; break;
      case 'b': currentLabel = "baseline"; break;
      case 'd': currentLabel = "drift";    break;
      case 'h': currentLabel = "heating";  break;
      case 'c': currentLabel = "cooling";  break;
      case 'n': currentLabel = "none";     break;
      case '?': if (!recording) printHelp(); break;
      default: break;  // ignore CR/LF and anything else
    }
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(12);                       // 0..4095, matches --adc-max 4095
  analogSetPinAttenuation(NTC_PIN, ADC_11db);     // widest input range (~0.15-2.45 V recommended)
#if USE_REF_CHANNEL
  analogSetPinAttenuation(REF_PIN, ADC_11db);
#endif
  delay(500);
  printHelp();
}

void loop() {
  handleSerialCommands();
  if (!recording) { yield(); return; }

  int64_t now = esp_timer_get_time();
  if (now < next_tick_us) { yield(); return; }

  // Fixed-grid scheduling (no cumulative drift). If we fell badly behind
  // (e.g. the host stalled the serial port), resync instead of bursting.
  if (now - next_tick_us > 5 * SAMPLE_PERIOD_US) next_tick_us = now;
  next_tick_us += SAMPLE_PERIOD_US;

  long long ts_ms = (long long)((now - t0_us) / 1000);   // actual sample time
  float adc = readAveraged(NTC_PIN);

#if USE_REF_CHANNEL
  float ref = readAveraged(REF_PIN);
  Serial.printf("%lld,%.2f,%.2f,%s\n", ts_ms, adc, ref, currentLabel);
#else
  Serial.printf("%lld,%.2f,%s\n", ts_ms, adc, currentLabel);
#endif
}