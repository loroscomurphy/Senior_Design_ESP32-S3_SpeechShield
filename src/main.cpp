/*
 * =============================================================================
 * ESP32-S3 VAD REACTIVE ULTRASONIC JAMMER
 * Senior Design Project - Final Calibrated Version (v2.0)
 * =============================================================================
 */
/*
#include <Arduino.h>
#include <SPI.h>
#include <AD9833.h>
#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Your trained model header
#include "vad_model_data.h"

// =============================================================================
// HARDWARE & PIN DEFINITIONS
// =============================================================================
#define FNC_PIN         10
#define DAT_PIN         11
#define CLK_PIN         12
#define RGB_LED_PIN     48    // Internal ESP32-S3 NeoPixel
#define BATTERY_PIN     6
#define I2S_WS_PIN      15
#define I2S_SCK_PIN     13
#define I2S_SD_PIN      16

// =============================================================================
// CALIBRATION & TUNING
// =============================================================================
#define SPEECH_THRESHOLD 0.10f       // Lowered for better responsiveness
#define NOISE_GATE       0.000002f   // Allows speech through, filters silence
#define DIGITAL_GAIN     100.0f      // Boosts signal before AI analysis
#define JAM_WINDOW_MS    5000        // Jam duration in milliseconds
#define START_FREQ       20000.0
#define END_FREQ         25000.0
#define FREQ_STEP        25.0

// =============================================================================
// GLOBALS
// =============================================================================
AD9833 gen(FNC_PIN);
enum SystemState { STATE_JAMMING, STATE_LISTENING };
volatile SystemState currentState = STATE_JAMMING;
volatile bool jammerAllowed = false;

unsigned long lastSpeechTime = 0;

// TFLite Buffers
int16_t audio_raw[16000]; 
constexpr int kArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Task Prototypes
void JammerTask(void* pv);
void AITask(void* pv);

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);
    while(!Serial); 
    Serial.println("\n--- VAD JAMMER: FINAL CALIBRATED SYSTEM ---");

    // 1. AD9833 Signal Gen Setup
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);
    gen.begin();
    gen.setWave(AD9833_OFF);
    Serial.println("[HW] Signal Generator: Initialized");

    // 2. I2S Microphone Setup
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 1024
    };
    i2s_pin_config_t pin_cfg = {
        .bck_io_num = I2S_SCK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = -1,
        .data_in_num = I2S_SD_PIN
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_cfg);
    Serial.println("[HW] I2S Microphone: Configured");

    // 3. TFLite AI Setup
    static tflite::MicroErrorReporter error_reporter;
    static tflite::AllOpsResolver resolver;
    const tflite::Model* model = tflite::GetModel(vad_model_data);
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &error_reporter);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("[AI] TensorFlow Lite: Ready");

    // 4. Create Dual Core Tasks
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096, NULL, 2, NULL, 1); // Core 1
    xTaskCreatePinnedToCore(AITask, "Brain", 10000, NULL, 1, NULL, 0);      // Core 0

    Serial.println("--- SYSTEM RUNNING ---\n");
}

void loop() { vTaskDelete(NULL); }

// =============================================================================
// CORE 1: FREQUENCY SWEEPER (JAMMER)
// =============================================================================
void JammerTask(void* pv) {
    float currentFreq = START_FREQ;
    for (;;) {
        if (jammerAllowed && currentState == STATE_JAMMING) {
            gen.setWave(AD9833_SINE);
            gen.setFrequency(currentFreq, 0);
            
            currentFreq += FREQ_STEP;
            if (currentFreq > END_FREQ) currentFreq = START_FREQ;
            
            // 600us delay creates a distinct sweep identifiable on scope
            delayMicroseconds(600); 
        } else {
            gen.setWave(AD9833_OFF);
            vTaskDelay(pdMS_TO_TICKS(20)); 
        }
    }
}

// =============================================================================
// CORE 0: THE BRAIN (VAD & CONTROL)
// =============================================================================
void AITask(void* pv) {
    size_t bytesRead;
    for (;;) {
        // --- 1. LISTEN PHASE ---
        currentState = STATE_LISTENING;
        neopixelWrite(RGB_LED_PIN, 0, 0, 30); // Dim Blue: Listening
        
        vTaskDelay(pdMS_TO_TICKS(60)); 
        i2s_zero_dma_buffer(I2S_NUM_0); 

        Serial.println("[SYSTEM] Capturing 1s Audio Window...");
        i2s_read(I2S_NUM_0, &audio_raw, sizeof(audio_raw), &bytesRead, portMAX_DELAY);
        
        // --- 2. PRE-PROCESSING (DC OFFSET & ENERGY) ---
        long sum = 0;
        float total_energy = 0;
        for(int i=0; i<16000; i++) sum += audio_raw[i];
        int16_t mean = sum / 16000; // Average DC Offset

        for(int i=0; i<16000; i++) {
            audio_raw[i] -= mean; // Center the waveform at 0
            float s = audio_raw[i] / 32768.0f;
            total_energy += s * s;
        }
        float avg_energy = total_energy / 16000.0f;

        // --- 3. FEATURE EXTRACTION ---
        for (int t = 0; t < kVadTimeFrames; t++) {
            for (int f = 0; f < kVadMfccFeatures; f++) {
                float frame_energy = 1e-8f; 
                int startIdx = t * kVadHopLength;
                
                // Analyze a wider slice (128 samples) per feature frame
                for(int i=0; i < 128; i++) {
                    float s = audio_raw[(startIdx + i) % 16000] / 32768.0f;
                    frame_energy += s * s;
                }
                
                // Natural log calculation with digital gain boost
                float logE = logf(frame_energy * DIGITAL_GAIN); 
                input->data.int8[t * kVadMfccFeatures + f] = 
                    (int8_t)constrain((logE / kVadInputScale + kVadInputZeroPoint), -128, 127);
            }
        }

        // --- 4. INFERENCE & LOGIC ---
        if (interpreter->Invoke() == kTfLiteOk) {
            float prob = (output->data.int8[0] - kVadOutputZeroPoint) * kVadOutputScale;
            
            // Force 0 if below noise floor to prevent "phantom" jamming
            if (avg_energy < NOISE_GATE) prob = 0.0f;

            Serial.printf("[AI] Energy: %.8f | Prob: %.2f\n", avg_energy, prob);

            if (prob > SPEECH_THRESHOLD) { 
                jammerAllowed = true;
                lastSpeechTime = millis();
                neopixelWrite(RGB_LED_PIN, 50, 0, 0); // Bright Red: JAMMING
                currentState = STATE_JAMMING;
                
                Serial.printf(">>> DETECTED! Jamming active for %d seconds <<<\n", JAM_WINDOW_MS / 1000);
                vTaskDelay(pdMS_TO_TICKS(JAM_WINDOW_MS)); 
            } else {
                // Return to standby after silence timeout
                if (millis() - lastSpeechTime > 3000) {
                    jammerAllowed = false;
                    //neopixelWrite(RGB_LED_PIN, 0, 20, 0); // Dim Green: Ready
                    Serial.println("[SILENCE] Jammer Standby.");
                    vTaskDelay(pdMS_TO_TICKS(400)); 
                }
            }
        }
    }
}
*/
/*
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

#include "vad_model_data.h"

// =============================================================================
// HARDWARE & CONFIG
// =============================================================================
#define FNC_PIN 10
#define DAT_PIN 11
#define CLK_PIN 12
#define RGB_LED_PIN 48
#define BATTERY_PIN 6
#define I2S_WS_PIN 15
#define I2S_SCK_PIN 13
#define I2S_SD_PIN 16

// Teammate's Memory Optimization Settings
#define I2S_BUFFER_SIZE 256 
#define I2S_DMA_BUF_COUNT 2 
#define AUDIO_BUFFER_LEN 16000
#define USE_TFLITE_MODEL true  // Toggle false to use Energy Fallback mode

// =============================================================================
// GLOBALS
// =============================================================================
AD9833 gen(FNC_PIN);
enum SystemState { STATE_JAMMING, STATE_LISTENING };
volatile SystemState currentState = STATE_JAMMING;
volatile bool jammerAllowed = false;

// Dynamic pointer for PSRAM allocation
int16_t* audio_buffer = nullptr; 

// TFLite
constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

unsigned long lastSpeechTime = 0;

void JammerTask(void* pv);
void AITask(void* pv);

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    // --- 1. MEMORY ALLOCATION (The Teammate's Fix) ---
    audio_buffer = (int16_t*)heap_caps_malloc(
        AUDIO_BUFFER_LEN * sizeof(int16_t), 
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!audio_buffer) {
        Serial.println("[MEM] PSRAM failed, using internal heap");
        audio_buffer = (int16_t*)malloc(AUDIO_BUFFER_LEN * sizeof(int16_t));
    } else {
        Serial.println("[MEM] Audio Buffer moved to PSRAM. Internal DRAM freed.");
    }

    // 2. AD9833 Setup
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);
    gen.begin();
    gen.setWave(AD9833_OFF);

    // 3. I2S Setup (Slimmed DMA buffers)
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_DMA_BUF_COUNT,
        .dma_buf_len = I2S_BUFFER_SIZE,
        .use_apll = false
    };
    i2s_pin_config_t pin_cfg = {.bck_io_num=I2S_SCK_PIN, .ws_io_num=I2S_WS_PIN, .data_out_num=-1, .data_in_num=I2S_SD_PIN};
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_cfg);

    // 4. AI Setup
#if USE_TFLITE_MODEL
    const tflite::Model* model = tflite::GetModel(vad_model_data);
    static tflite::MicroErrorReporter error_reporter;
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &error_reporter);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);
#endif

    // 5. Tasks
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096, NULL, 2, NULL, 1);
    xTaskCreatePinnedToCore(AITask, "Brain", 10000, NULL, 1, NULL, 0);
}

void loop() { vTaskDelete(NULL); }

// =============================================================================
// CORE 1: JAMMER SWEEP
// =============================================================================
void JammerTask(void* pv) {
    float currentFreq = 20000.0;
    for (;;) {
        if (jammerAllowed && currentState == STATE_JAMMING) {
            gen.setWave(AD9833_SINE);
            gen.setFrequency(currentFreq, 0);
            currentFreq += 25.0;
            if (currentFreq > 25000.0) currentFreq = 20000.0;
            delayMicroseconds(600); 
        } else {
            gen.setWave(AD9833_OFF);
            vTaskDelay(pdMS_TO_TICKS(20));
        }
    }
}

// =============================================================================
// CORE 0: AI BRAIN
// =============================================================================
void AITask(void* pv) {
    size_t bytesRead;
    for (;;) {
        currentState = STATE_LISTENING;
        neopixelWrite(RGB_LED_PIN, 0, 0, 20); // Blue
        vTaskDelay(pdMS_TO_TICKS(50));
        i2s_zero_dma_buffer(I2S_NUM_0);

        // Record 1 second into PSRAM buffer
        i2s_read(I2S_NUM_0, (void*)audio_buffer, AUDIO_BUFFER_LEN * sizeof(int16_t), &bytesRead, portMAX_DELAY);
        
        currentState = STATE_JAMMING; // Allow jammer while we think

        float speech_prob = 0.0f;

#if USE_TFLITE_MODEL
        // --- TFLITE INFERENCE PATH ---
        // (Includes teammate's quantization loop)
        for (int t = 0; t < kVadTimeFrames; t++) {
            for (int f = 0; f < kVadMfccFeatures; f++) {
                float energy = 0.0001f;
                int start = t * kVadHopLength;
                for(int i=0; i<64; i++) {
                    float s = audio_buffer[(start + i) % 16000] / 32768.0f;
                    energy += s * s;
                }
                float logE = logf(energy * 100.0f);
                int8_t quantized = (int8_t)constrain((logE / kVadInputScale + kVadInputZeroPoint), -128, 127);
                input->data.int8[t * kVadMfccFeatures + f] = quantized;
            }
        }
        if (interpreter->Invoke() == kTfLiteOk) {
            speech_prob = (output->data.int8[0] - kVadOutputZeroPoint) * kVadOutputScale;
        }
#else
        // --- ENERGY FALLBACK PATH ---
        float total_e = 0;
        for(int i=0; i<16000; i++) {
            float s = audio_buffer[i] / 32768.0f;
            total_e += s * s;
        }
        speech_prob = (total_e / 16000.0f > 0.0001f) ? 0.9f : 0.1f;
#endif

        Serial.printf("[VAD] Prob: %.2f\n", speech_prob);

        if (speech_prob > 0.75f) {
            jammerAllowed = true;
            lastSpeechTime = millis();
            neopixelWrite(RGB_LED_PIN, 100, 0, 0); // Red
            vTaskDelay(pdMS_TO_TICKS(5000)); // Jam for 5s
        } else if (millis() - lastSpeechTime > 3000) {
            jammerAllowed = false;
           // neopixelWrite(RGB_LED_PIN, 0, 10, 0); // Green
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }
}
/*void AITask(void* pv) {
    size_t bytesRead;
    for (;;) {
        currentState = STATE_LISTENING;
        neopixelWrite(RGB_LED_PIN, 0, 0, 20); // Blue
        vTaskDelay(pdMS_TO_TICKS(50));
        i2s_zero_dma_buffer(I2S_NUM_0);

        // Record into PSRAM
        i2s_read(I2S_PORT, (void*)audio_buffer, AUDIO_BUFFER_LEN * sizeof(int16_t), &bytesRead, portMAX_DELAY);
        currentState = STATE_JAMMING;

        // --- PRE-PROCESSING & DC OFFSET ---
        long sum = 0;
        for(int i=0; i<16000; i++) sum += audio_buffer[i];
        int16_t mean = sum / 16000;

        float total_avg_energy = 0;
        for (int t = 0; t < kVadTimeFrames; t++) {
            for (int f = 0; f < kVadMfccFeatures; f++) {
                float frame_energy = 1e-9f; 
                int start = t * kVadHopLength;
                
                for(int i=0; i < 64; i++) {
                    float s = (audio_buffer[(start + i) % 16000] - mean) / 32768.0f;
                    frame_energy += s * s;
                }
                
                // Use teammate's logf logic
                float logE = logf(frame_energy + 1e-6f);
                total_avg_energy += logE;

                // QUANTIZATION
                int8_t quantized = (int8_t)constrain((logE / kVadInputScale + kVadInputZeroPoint), -128, 127);
                input->data.int8[t * kVadMfccFeatures + f] = quantized;
            }
        }

        // --- INFERENCE ---
        if (interpreter->Invoke() == kTfLiteOk) {
            float prob = (output->data.int8[0] - kVadOutputZeroPoint) * kVadOutputScale;
            float system_avg_log = total_avg_energy / (kVadTimeFrames * kVadMfccFeatures);

            // DIAGNOSTIC PRINT: Watch these numbers in silence!
            Serial.printf("[DIAG] Avg Log: %.2f | Prob: %.2f\n", system_avg_log, prob);

            // AUTO-GATE: If the Avg Log is too low, force probability to 0
            // Adjust -10.0 based on what you see in the Serial Monitor for silence
            if (system_avg_log < -10.0f) prob = 0.0f; 

            if (prob > 0.85f) { // Higher threshold for more stability
                jammerAllowed = true;
                lastSpeechTime = millis();
                neopixelWrite(RGB_LED_PIN, 100, 0, 0); // Red
                Serial.println(">>> SPEECH DETECTED! <<<");
                vTaskDelay(pdMS_TO_TICKS(JAM_WINDOW_MS)); 
            } else if (millis() - lastSpeechTime > 3000) {
                jammerAllowed = false;
                neopixelWrite(RGB_LED_PIN, 0, 10, 0); // Green
                vTaskDelay(pdMS_TO_TICKS(200));
            }
        }
    }
}*/

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
// HARDWARE PINS
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
// CONFIG
// =============================================================================
// Increased buffers so the mic doesn't drop frames when the AI runs
#define I2S_BUFFER_SIZE     512    
#define I2S_DMA_BUF_COUNT   4      
#define AUDIO_BUFFER_LEN    16000
#define USE_TFLITE_MODEL    true

// 1. Noise Gate: We are checking against the REAL energy, but printing a scaled version.
#define NOISE_GATE_THRESH   0.000001f // Equivalent to an "Energy Score" of 6.0
// 2. AI Threshold
#define ACTIVE_SPEECH_THRESH 0.80f  

// =============================================================================
// GLOBALS
// =============================================================================
AD9833 gen(FNC_PIN);

enum SystemState { STATE_JAMMING, STATE_LISTENING };
volatile SystemState currentState = STATE_JAMMING;
std::atomic<bool> jammerAllowed(false);

int16_t* audio_buffer = nullptr;

// TFLite
constexpr int kArenaSize = 32 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor  = nullptr;
TfLiteTensor* output_tensor = nullptr;

unsigned long lastSpeechTime = 0;

void JammerTask(void* pv);
void AITask(void* pv);

// =============================================================================
// MFCC EXTRACTION (TEAMMATE'S ORIGINAL MATH + OUR NOISE GATE)
// =============================================================================
static bool compute_features() {
    // 1. DC OFFSET REMOVAL (Crucial for time-domain energy accuracy)
    long sum = 0;
    for (int i = 0; i < AUDIO_BUFFER_LEN; i++) sum += audio_buffer[i];
    int16_t mean = sum / AUDIO_BUFFER_LEN;

    // 2. NOISE GATE ENERGY CHECK
    float total_energy = 0.0f;
    for (int i = 0; i < AUDIO_BUFFER_LEN; i++) {
        audio_buffer[i] -= mean; // Center the waveform at 0
        float s = audio_buffer[i] / 32768.0f;
        total_energy += s * s;
    }
    float avg_energy = total_energy / AUDIO_BUFFER_LEN;
    
    // Multiply by 1,000,000 for a human-readable "Energy Score" in the Serial Monitor
    Serial.printf("[DSP] Energy Score: %.2f\n", avg_energy * 1000000.0f);

    if (avg_energy < NOISE_GATE_THRESH) {
        return false; // Room is quiet, skip AI
    }

    // 3. FEATURE EXTRACTION (Original Kaggle logic)
    int frame_length = kVadFftSize;
    int hop = kVadHopLength;

    for (int frame = 0; frame < kVadTimeFrames; frame++) {
        int start = frame * hop;

        for (int band = 0; band < kVadMfccFeatures; band++) {
            float energy = 0.0f;
            int band_start = (band * frame_length) / (kVadMfccFeatures * 2);
            int band_end = ((band + 1) * frame_length) / (kVadMfccFeatures * 2);

            for (int i = band_start; i < band_end && (start + i) < AUDIO_BUFFER_LEN; i++) {
                float sample = audio_buffer[start + i] / 32768.0f;
                energy += sample * sample;
            }

            // Log compression 
            float value = logf(energy + 1e-6f);

            // Quantize directly to tensor
            int q = (int)(value / kVadInputScale + kVadInputZeroPoint);
            input_tensor->data.int8[frame * kVadMfccFeatures + band] = (int8_t)constrain(q, -128, 127);
        }
    }
    return true;
}

// =============================================================================
// SETUP
// =============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    // Memory Allocation (PSRAM Fix)
    audio_buffer = (int16_t*)heap_caps_malloc(AUDIO_BUFFER_LEN * sizeof(int16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!audio_buffer) {
        audio_buffer = (int16_t*)malloc(AUDIO_BUFFER_LEN * sizeof(int16_t));
    }

    // AD9833 Init
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);
    gen.begin();
    gen.setWave(AD9833_OFF);

    // I2S Mic Init
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
    i2s_pin_config_t pin_cfg = {.bck_io_num = I2S_SCK_PIN, .ws_io_num = I2S_WS_PIN, .data_out_num = -1, .data_in_num = I2S_SD_PIN};
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

    // Start Dual Cores
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096,  NULL, 2, NULL, 1);
    xTaskCreatePinnedToCore(AITask,     "Brain",  14000, NULL, 1, NULL, 0);
}

void loop() { vTaskDelete(NULL); }

// =============================================================================
// CORE 1 — JAMMER SWEEP
// =============================================================================
void JammerTask(void* pv) {
    float currentFreq = 20000.0f;
    for (;;) {
        if (jammerAllowed.load() && currentState == STATE_JAMMING) {
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
// CORE 0 — AI BRAIN
// =============================================================================
void AITask(void* pv) {
    size_t bytesRead;

    for (;;) {
        currentState = STATE_LISTENING;
        neopixelWrite(RGB_LED_PIN, 0, 0, 20);   // Blue = listening
        vTaskDelay(pdMS_TO_TICKS(50));
        i2s_zero_dma_buffer(I2S_NUM_0);

        i2s_read(I2S_NUM_0, (void*)audio_buffer, AUDIO_BUFFER_LEN * sizeof(int16_t), &bytesRead, portMAX_DELAY);

        currentState = STATE_JAMMING;
        float speech_prob = 0.0f;

#if USE_TFLITE_MODEL
        if (compute_features()) {
            if (interpreter->Invoke() == kTfLiteOk) {
                speech_prob = (output_tensor->data.int8[0] - kVadOutputZeroPoint) * kVadOutputScale;
            }
        } else {
            speech_prob = 0.0f; 
            Serial.println("[GATE] Silence detected, skipping inference.");
        }
#endif

        if (speech_prob > 0.0f) {
            Serial.printf("[VAD] prob=%.3f\n", speech_prob);
        }

        if (speech_prob > ACTIVE_SPEECH_THRESH) {
            jammerAllowed.store(true);
            lastSpeechTime = millis();
            neopixelWrite(RGB_LED_PIN, 100, 0, 0);  // Red = jamming
            Serial.println(">>> SPEECH DETECTED! JAMMING! <<<");
            vTaskDelay(pdMS_TO_TICKS(5000));
        } else if (millis() - lastSpeechTime > 3000) {
            jammerAllowed.store(false);
            //neopixelWrite(RGB_LED_PIN, 0, 10, 0);   // Green = idle
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }
}