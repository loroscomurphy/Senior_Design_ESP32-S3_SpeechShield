# SpeechShield

ESP32-S3 ultrasonic speech jammer with on-device AI voice activity detection. Listens for speech via I2S MEMS microphone, runs a TensorFlow Lite Micro VAD model, and drives an AD9833 DDS chip to sweep a 20–25 kHz jamming signal when speech is detected.

Dual-core FreeRTOS: Core 0 handles mic capture + inference, Core 1 handles the jammer sweep independently.

---

## Hardware

| Component | Notes |
|-----------|-------|
| ESP32-S3 DevKitC-1-N16R8V | 16MB flash, 8MB PSRAM required |
| AD9833 DDS module | SPI, generates ultrasonic sine wave |
| I2S MEMS microphone | INMP441 or SPH0645LM4H |
| WS2812B RGB LED | Status indicator (currently unused in firmware) |
| LiPo 3.7V 1000mAh+ | ~2–3 hr runtime under continuous jamming |
| TP4056 charger module | With protection circuit |

---

## Pin Assignments

| GPIO | Component | Signal |
|------|-----------|--------|
| 10 | AD9833 | FNC (chip select) |
| 11 | AD9833 | DAT (MOSI) |
| 12 | AD9833 | CLK (SPI clock) |
| 13 | I2S mic | SCK (bit clock) |
| 15 | I2S mic | WS (word select / LRCLK) |
| 16 | I2S mic | SD (data) |
| 48 | WS2812B | Data |
| 6 | — | Battery ADC (unused) |

---

## VAD Model

Custom 3-layer CNN trained for this project on the Google Speech Commands v2 dataset.

| Property | Value |
|----------|-------|
| Architecture | 3-layer CNN |
| Training data | Google Speech Commands v2 (896 samples/class, balanced) |
| Input shape | `[1, 32, 13]` — 32 time frames × 13 MFCCs |
| Frame size | ~32ms per frame at 16kHz |
| Quantization | int8 |
| Model size | 21.6 KB |
| Tensor arena | 32 KB |
| Test accuracy | 97.22% |
| Inference time | ~10ms on ESP32-S3 |
| Training environment | Kaggle + Purdue Scholar HPC |

Quantization parameters (from `vad_model_data.h`):
- Input: scale=`0.489906`, zero_point=`75`
- Output: scale=`0.003906`, zero_point=`-128`

The model expects true MFCCs (FFT → mel filterbank → log → DCT), not raw energy bands. See Known Issues #2.

---

## How It Works

```
[I2S mic] → AITask (Core 0) → TFLite VAD → jammerAllowed flag
                                                    ↓
                              JammerTask (Core 1) → AD9833 sweep (20–25 kHz)
```

**AITask (Core 0)**
1. Reads 16,000 samples (1 sec @ 16kHz, 16-bit mono) from I2S
2. DC offset removal + noise gate (energy < 1e-6 threshold → skip)
3. Transient filter: sound must span ≥5 frames or it's ignored (claps, snaps)
4. MFCC features (Hanning → 512-pt FFT → mel filterbank → log → DCT-II) → quantized int8 → fed to TFLite input tensor
5. If inference score > 0.80 → set `jammerAllowed = true`, update `lastSpeechTime`
6. If no speech for 3s → set `jammerAllowed = false`

**JammerTask (Core 1)**
- Sweeps 20,000–25,000 Hz in 25 Hz steps, 600µs per step
- Only runs when `jammerAllowed == true && currentState == STATE_JAMMING`

---

## Build & Flash

Requires PlatformIO.

```bash
# Install PlatformIO CLI
pip install platformio

# Clone and enter repo
git clone https://github.com/loroscomurphy/Senior_Design_ESP32-S3_SpeechShield.git
cd Senior_Design_ESP32-S3_SpeechShield

# Build and flash
pio run -t upload

# Serial monitor
pio device monitor --baud 115200
```

The TFLite model binary must be present in `include/vad_model_data.h` as a C array. The constants `kVadFftSize`, `kVadHopLength`, `kVadTimeFrames`, `kVadMfccFeatures`, `kVadInputScale`, `kVadInputZeroPoint`, `kVadOutputScale`, `kVadOutputZeroPoint` must match the model's quantization parameters.

---

## Tuning

**Noise gate** (`NOISE_GATE_THRESH = 0.000001f`): raise if triggering on HVAC/ambient noise, lower if missing soft speech. Uncomment the `[DSP] Energy Score` serial print to calibrate.

**VAD threshold** (`ACTIVE_SPEECH_THRESH = 0.80f`): raise to reduce false positives, lower to catch quieter speech.

---

## Known Issues

1. **No error checking on I2S init or `AllocateTensors()`** — failures are silent. Add return value checks if debugging initialization issues.

2. **`currentState` race condition** — accessed across cores without atomic protection. Works in practice on 32-bit ESP32-S3 but should be `std::atomic<SystemState>`.

---

## Legal

Ultrasonic jamming devices may be restricted in some jurisdictions. Use only in controlled environments with appropriate authorization.
