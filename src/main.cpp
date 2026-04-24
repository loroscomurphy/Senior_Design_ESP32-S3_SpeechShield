#include <Arduino.h>  
#include <SPI.h>  
#include <AD9833.h>  
#include <driver/i2s.h>  
#include <esp_heap_caps.h>  
#include <TensorFlowLite_ESP32.h>  
#include <tensorflow/lite/micro/all_ops_resolver.h>  
#include <tensorflow/lite/micro/micro_error_reporter.h>  
#include <tensorflow/lite/micro/micro_interpreter.h>  
#include <tensorflow/lite/schema/schema_generated.h>  
#include <atomic>  
#include <cmath>  
  
#include "vad_model_data.h"  
  
// =============================================================================  
// CONFIG  
// =============================================================================  
#define I2S_BUFFER_SIZE     512    
#define I2S_DMA_BUF_COUNT   4      
#define AUDIO_BUFFER_LEN    16000
#define USE_TFLITE_MODEL    true  
  
#define NOISE_GATE_THRESH     0.000001f // Equivalent to an "Energy Score" of 6.0
#define ACTIVE_SPEECH_THRESH  0.80f  
#define ENERGY_SPEECH_THRESH  0.01f        // kept for when you re-enable fallback  
  
#define DEBUG_VAD             true         // set false when not testing  
  
// =============================================================================  
// PINS  
// =============================================================================  
#define FNC_PIN         10  
#define DAT_PIN         11  
#define CLK_PIN         12  
#define RGB_LED_PIN     48  
#define BATTERY_PIN     6  
#define I2S_WS_PIN      15  
#define I2S_SCK_PIN     13  
#define I2S_SD_PIN      16  
  
// =============================================================================  
// GLOBALS  
// =============================================================================  
AD9833 gen(FNC_PIN);  
  
enum SystemState { STATE_JAMMING, STATE_LISTENING };  
std::atomic<SystemState> currentState{STATE_JAMMING};   // fixed: now atomic  
std::atomic<bool> jammerAllowed{false};  
  
int16_t* audio_buffer = nullptr;  
  
// TFLite  
constexpr int kArenaSize = 32 * 1024;  
alignas(16) uint8_t tensor_arena[kArenaSize];  
tflite::MicroInterpreter* interpreter = nullptr;  
TfLiteTensor* input_tensor  = nullptr;  
TfLiteTensor* output_tensor = nullptr;  
  
unsigned long lastSpeechTime = 0;  
  
// FFT buffers  
static float tw_re[kVadFftSize / 2];  
static float tw_im[kVadFftSize / 2];  
static float hann[kVadFftSize];  
static float fft_buf_re[kVadFftSize];  
static float fft_buf_im[kVadFftSize];  
  
void JammerTask(void* pv);  
void AITask(void* pv);  
  
// =============================================================================  
// MFCC EXTRACTION (WITH TRANSIENT NOISE FILTER)  
// =============================================================================  
static bool compute_features() {  
    // 1. DC OFFSET REMOVAL  
    long sum = 0;  
    for (int i = 0; i < AUDIO_BUFFER_LEN; i++) sum += audio_buffer[i];  
    int16_t mean = sum / AUDIO_BUFFER_LEN;  
  
    // 2. GLOBAL NOISE GATE CHECK  
    float total_energy = 0.0f;  
    for (int i = 0; i < AUDIO_BUFFER_LEN; i++) {  
        audio_buffer[i] -= mean;  
        float s = audio_buffer[i] / 32768.0f;  
        total_energy += s * s;  
    }  
  
    float avg_energy = total_energy / AUDIO_BUFFER_LEN; // This is the "Energy Score"  
    //Serial.printf("[DSP] Energy Score: %.2f\n", avg_energy * 1000000.0f); //Debugging: Print a scaled version of the energy for easier reading.  
  
    if (avg_energy < NOISE_GATE_THRESH) {  
        return false; // Room is entirely quiet  
    }  
  
    // 3. FEATURE EXTRACTION & TRANSIENT FILTER  
    int frame_length = kVadFftSize;  
    int hop = kVadHopLength;  
      
    // NEW: Counter to track how long the sound lasts  
    int active_frames = 0;  
  
    for (int frame = 0; frame < kVadTimeFrames; frame++) {  
        int start = frame * hop;  
        float frame_energy_sum = 0.0f; // Track energy of this specific slice  
  
        for (int band = 0; band < kVadMfccFeatures; band++) {  
            float energy = 0.0f; // Calculate energy for this specific band/slice  
            int band_start = (band * frame_length) / (kVadMfccFeatures * 2);  
            int band_end = ((band + 1) * frame_length) / (kVadMfccFeatures * 2);  
  
            // Calculate energy for this specific band/slice to apply the transient filter  
            for (int i = band_start; i < band_end && (start + i) < AUDIO_BUFFER_LEN; i++) {  
                float sample = audio_buffer[start + i] / 32768.0f;  
                energy += sample * sample;  
            }  
              
            frame_energy_sum += energy; // Accumulate the slice's total energy  
  
            // Log compression & Quantization  
            float value = logf(energy + 1e-6f);  
            // FIXED: correct int8 quantization for the model  
            int q = (int)roundf(value / kVadInputScale + kVadInputZeroPoint);  
            input_tensor->data.int8[frame * kVadMfccFeatures + band] = (int8_t)constrain(q, -128, 127);  
        }  
          
        // NEW: Check if this specific fraction of a second is loud  
        float avg_frame_energy = frame_energy_sum / frame_length;  
        if (avg_frame_energy > NOISE_GATE_THRESH) {  
            active_frames++;  
        }  
    }  
      
    // Serial.printf("[DSP] Sustained Frames: %d / 32\n", active_frames); //Debugging  
  
    // NEW: The "Clap / Snap" Filter  
    // If the sound didn't last for at least 5 frames, it's not speech.  
    if (active_frames < 5) {  
        // Serial.println("[GATE] Transient Noise Ignored (Clap/Snap/Click)."); //Debugging  
        return false;  
    }  
  
    return true; // Sustained sound detected, let the AI analyze it!  
}  
  
// =============================================================================  
// SETUP  
// =============================================================================  
void setup() {  
    Serial.begin(115200);  
    delay(1000);  
    neopixelWrite(RGB_LED_PIN, 0, 0, 0);  
  
    audio_buffer = (int16_t*)heap_caps_malloc(AUDIO_BUFFER_LEN * sizeof(int16_t),  
                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);  
    if (!audio_buffer) {  
        audio_buffer = (int16_t*)malloc(AUDIO_BUFFER_LEN * sizeof(int16_t));  
    }  
    if (!audio_buffer) while(1) delay(1000);  
  
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);  
    gen.begin();  
    gen.setWave(AD9833_OFF);  
  
    i2s_config_t i2s_config = {  
        .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),  
        .sample_rate          = 16000,  
        .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,  
        .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,  
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,  
        .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,  
        .dma_buf_count        = I2S_DMA_BUF_COUNT,  
        .dma_buf_len          = I2S_BUFFER_SIZE,  
        .use_apll             = false  
    };  
  
    i2s_pin_config_t pin_cfg = {  
        .bck_io_num = I2S_SCK_PIN,  
        .ws_io_num = I2S_WS_PIN,  
        .data_out_num = -1,  
        .data_in_num = I2S_SD_PIN  
    };  
  
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);  
    i2s_set_pin(I2S_NUM_0, &pin_cfg);  
  
    // TFLite Init  
#if USE_TFLITE_MODEL
    const tflite::Model* model = tflite::GetModel(vad_model_data);  
    static tflite::MicroErrorReporter error_reporter;  
    static tflite::AllOpsResolver resolver;  
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &error_reporter);  
    interpreter = &static_interpreter;  
    interpreter->AllocateTensors();  
    input_tensor  = interpreter->input(0);  
    output_tensor = interpreter->output(0);  
#endif
  
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096, NULL, 2, NULL, 1);  
    xTaskCreatePinnedToCore(AITask,     "Brain",  14000, NULL, 1, NULL, 0);  
}  
  
void loop() { vTaskDelete(NULL); }  
  
// =============================================================================  
// JAMMER TASK (Core 1)  
// =============================================================================  
void JammerTask(void* pv) {  
    float currentFreq = 20000.0f;  
    for (;;) {  
        if (jammerAllowed.load() && currentState.load() == STATE_JAMMING) {  
            gen.setWave(AD9833_SINE);  
            gen.setFrequency(currentFreq, 0);  
            currentFreq += 25.0f;  
            if (currentFreq > 25000.0f) currentFreq = 20000.0f;  
            delayMicroseconds(600);  
        } else {  
            gen.setWave(AD9833_OFF);  
            vTaskDelay(pdMS_TO_TICKS(20));  
        }  
    }  
}  
  
// =============================================================================  
// AI TASK (Core 0) - TinyML with transient filter + VAD debug  
// =============================================================================  
void AITask(void* pv) {  
    size_t bytesRead;  
  
    for (;;) {  
        currentState.store(STATE_LISTENING);  
        vTaskDelay(pdMS_TO_TICKS(50));  
        i2s_zero_dma_buffer(I2S_NUM_0);  
  
        i2s_read(I2S_NUM_0, (void*)audio_buffer, AUDIO_BUFFER_LEN * sizeof(int16_t),  
                 &bytesRead, portMAX_DELAY);  
  
        currentState.store(STATE_JAMMING);  
  
        float speech_prob = 0.0f;  
  
#if USE_TFLITE_MODEL  
        if (compute_features()) {  
            if (interpreter->Invoke() == kTfLiteOk) {  
                speech_prob = (output_tensor->data.int8[0] - kVadOutputZeroPoint)  
                              * kVadOutputScale;  
  
                if (DEBUG_VAD) {  
                    Serial.printf("[VAD] prob=%.3f  %s\n", speech_prob,  
                                  (speech_prob > ACTIVE_SPEECH_THRESH) ? "SPEECH!" : "");  
                }  
            }  
        } else {  
            speech_prob = 0.0f;  
            // Debugging: Print when silence/noise gate is detected
            /*
            Serial.println("[GATE] Silence detected, skipping inference.");
            */
        }  
#endif  
  
        /* === FALLBACK ENERGY DETECTOR COMMENTED OUT FOR TESTING TINYML ===  
        bool use_fallback = false;  
        if (!USE_TFLITE_MODEL || !compute_features()) use_fallback = true;  
  
        if (use_fallback) {  
            float total_energy = 0.0f;  
            for (int i = 0; i < AUDIO_BUFFER_LEN; i++) {  
                float s = audio_buffer[i] / 32768.0f;  
                total_energy += s * s;  
            }  
            float avg_energy = total_energy / AUDIO_BUFFER_LEN;  
            if (avg_energy > ENERGY_SPEECH_THRESH) {  
                speech_prob = 1.0f;  
            }  
        }  
        // ================================================================ */  
  
        if (speech_prob > ACTIVE_SPEECH_THRESH) {  
            jammerAllowed.store(true);  
            lastSpeechTime = millis();  
            vTaskDelay(pdMS_TO_TICKS(180000)); // Jam for 3 min, turned off by timer after speech ends
            // Debugging
            /*
            Serial.println(">>> SPEECH DETECTED! JAMMING! <<<");
            */
        }  
        else if (millis() - lastSpeechTime > 3000) {  
            // If no speech has been detected for 3 seconds, turn off the jammer  
            jammerAllowed.store(false);  
            vTaskDelay(pdMS_TO_TICKS(200));  
        }  
    }  
}
