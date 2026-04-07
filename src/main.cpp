/*
 * =============================================================================
 * ESP32-S3 VAD REACTIVE ULTRASONIC JAMMER
 * Senior Design Project - Final Calibrated Version (v2.0)
 * =============================================================================
 */

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
#define SPEECH_THRESHOLD 0.70f       // Lowered for better responsiveness
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
