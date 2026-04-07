#include <Arduino.h>
#include <SPI.h>
#include <AD9833.h>
#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// Include the VAD model data as a header (converted from .tflite using xxd -i)
#include "vad_model_data.h"

// =============================================================================
// HARDWARE PINOUT FOR SIGNAL GEN, MIC, STATUS LED, AND BATTERY MONITOR
// =============================================================================
#define FNC_PIN           10    // AD9833 FSync
#define DAT_PIN           11    // AD9833 SDATA
#define CLK_PIN           12    // AD9833 SCLK
#define LED_PIN           5     // Status LED
#define I2S_WS_PIN        15    // Mic LRCL
#define I2S_SCK_PIN       13    // Mic BCLK
#define I2S_SD_PIN        16    // Mic DOUT
#define BATTERY_PIN       6     // LiPo Voltage

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================
const float startFreq = 20000.0; // 20 kHz is a common lower limit for ultrasonic jammers, and is generally inaudible to most humans.
const float endFreq   = 25000.0; // 25 kHz is the upper limit for most ultrasonic mics and is still mostly inaudible to humans.
const float freqstep  = 10.0; // 10 Hz step for the sweep
const int jamWindowMs = 3000;    // Jam for 3 seconds
const int listenWindowMs = 200;  // Listen for 200ms (the "hole")

// I2S Config for 200ms window
#define I2S_SAMPLE_RATE     16000
#define SAMPLES_200MS       3200 // (16000 * 0.2)

// =============================================================================
// GLOBAL STATE & TASKS FOR DUAL-CORE OPERATION
// =============================================================================
enum SystemState { STATE_JAMMING, STATE_LISTENING }; // Simple state machine for the jammer
volatile SystemState currentState = STATE_JAMMING; // Start in jamming mode
volatile bool jammerAllowed = false; // Flag to control whether jamming is allowed based on VAD results
unsigned long lastSpeechTime = 0; // Timestamp of the last detected speech, used for timeout logic

AD9833 gen(FNC_PIN); // AD9833 instance for signal generation
int16_t audio_buffer[SAMPLES_200MS]; // Buffer to hold 200ms of audio data from the I2S mic

// TFLite Globals
constexpr int kArenaSize = 32 * 1024; // 32 KB arena for TFLite tensors
alignas(16) uint8_t tensor_arena[kArenaSize]; // Aligned arena for TFLite tensors
tflite::MicroInterpreter* interpreter = nullptr; // Global pointer to the TFLite interpreter instance
TfLiteTensor* input = nullptr; // Pointer to the input tensor
TfLiteTensor* output = nullptr; // VAD thresholds and quantization parameters

// Prototypes
void JammerTask(void* pv);
void AITask(void* pv);
void setupI2S();
void extractFeaturesAndInference();

// =============================================================================
// SETUP AND MAIN LOOP
// =============================================================================
void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT); // Status LED for debugging
    
    // 1. AD9833 Setup
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);
    gen.begin();
    gen.reset();

    // 2. I2S Mic Setup
    setupI2S();

    // 3. TFLite Setup
    static tflite::MicroErrorReporter error_reporter; // Must be static to persist after setup() exits
    static tflite::AllOpsResolver resolver; // Registers all ops, can be optimized by only registering needed ops for the VAD model
    const tflite::Model* model = tflite::GetModel(vad_model_data); // Get the model from the included header
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize, &error_reporter); // Must be static to persist after setup() exits
    interpreter = &static_interpreter; // Set the global pointer to the interpreter instance
    interpreter->AllocateTensors(); // Allocate memory for tensors
    input = interpreter->input(0); // Get pointer to the input tensor
    output = interpreter->output(0); // Get pointer to the output tensor

    // 4. Dual-Core Task Creation
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096, NULL, 1, NULL, 1); // Core 1: Signal
    xTaskCreatePinnedToCore(AITask, "AI_Brain", 16384, NULL, 1, NULL, 0); // Core 0: AI
}

void loop() { 
    vTaskDelete(NULL); // Main loop is handled by FreeRTOS tasks
}

// =============================================================================
// CORE 1: CONTINUOUS FREQUENCY SWEEP - Logan Orosco-Murphy's DOMAIN
// =============================================================================
void JammerTask(void* pv) {
    float currentFreq = startFreq; // Start at the lower frequency of the sweep
    
    for (;;) {
        // Jamming logic based on the current state and VAD results
        if (currentState == STATE_JAMMING && jammerAllowed) {
            gen.setWave(AD9833_SINE); // Ensure we're in sine wave mode for jamming
            gen.setFrequency(currentFreq, 0); // Update the frequency for the sweep
            
            currentFreq += freqstep; // Increment the frequency for the next step
            if (currentFreq > endFreq) currentFreq = startFreq; // Loop back to the start frequency
            
            delayMicroseconds(100); // Sweep speed control
        } else {
            // "Quiet Window" logic
            gen.setWave(AD9833_OFF); // Turn off the signal generator during the quiet window
            vTaskDelay(pdMS_TO_TICKS(5)); // Low CPU usage while idling. Helps with timing for the AI task to capture clean audio and keeping battery life high.
        }
    }
}

// =============================================================================
// CORE 0: AI BRAIN (VAD & TIMING) - Eric Raymond's DOMAIN
// =============================================================================
void AITask(void* pv) {
    size_t bytesRead;

    for (;;) {
        // --- PHASE 1: JAMMING PERIOD ---
        vTaskDelay(pdMS_TO_TICKS(jamWindowMs));

        // --- PHASE 2: THE QUIET WINDOW (200ms) ---
        currentState = STATE_LISTENING;
        vTaskDelay(pdMS_TO_TICKS(10)); // Settlement for Amp/Mic
        
        // Capture exactly 200ms of audio
        i2s_read(I2S_NUM_0, &audio_buffer, sizeof(audio_buffer), &bytesRead, portMAX_DELAY);
        
        // Resume jamming immediately after capture
        currentState = STATE_JAMMING;

        // --- PHASE 3: AI INFERENCE (Concurrent with Jamming) ---
        extractFeaturesAndInference();
    }
}

// =============================================================================
// HELPER FUNCTIONS FOR I2S SETUP AND AI INFERENCE
// =============================================================================
void setupI2S() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX), // Master receive mode
        .sample_rate = I2S_SAMPLE_RATE, // 16 kHz is a common sample rate for VAD and is sufficient for capturing human speech while keeping data size manageable.
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, // 16-bit audio for better VAD performance
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Mono input from the mic
        .communication_format = I2S_COMM_FORMAT_STAND_I2S, // Standard I2S format
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Interrupt level 1 for I2S
        .dma_buf_count = 4, // Number of DMA buffers
        .dma_buf_len = 512 // Buffer length in samples, not bytes. 512 samples at 16 bits/channel = 1024 bytes per buffer.
    };
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN, // Set the I2S pins according to the defined constants
        .ws_io_num = I2S_WS_PIN, // Set the I2S pins according to the defined constants
        .data_out_num = I2S_PIN_NO_CHANGE, // Not used for RX
        .data_in_num = I2S_SD_PIN // Set the I2S pins according to the defined constants
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL); // Install I2S driver
    i2s_set_pin(I2S_NUM_0, &pin_config); // Set the I2S pins
}

void extractFeaturesAndInference() {
    // 1. Prep data for TFLite (Teammate's quantization logic)
    for (int i = 0; i < SAMPLES_200MS; i++) {
        // This is a placeholder for MFCC scaling
        float normalized = audio_buffer[i] / 32768.0f;
        int8_t quantized = (int8_t)(normalized / kVadInputScale + kVadInputZeroPoint); // Quantize to int8 using the model's input scale and zero point
        if (i < input->bytes) input->data.int8[i] = quantized; // Ensure we don't write out of bounds
    }

    // 2. Run TFLite Inference
    if (interpreter->Invoke() == kTfLiteOk) {
        // Assuming output is a single int8 value representing speech probability
        float prob = (output->data.int8[0] - kVadOutputZeroPoint) * kVadOutputScale;
        // 3. Decision Logic for Jamming Control
        if (prob > kVadThreshold) {
            jammerAllowed = true;
            lastSpeechTime = millis();
            digitalWrite(LED_PIN, HIGH);
            // Stay in jamming mode for the duration of the jam window after detecting speech
        } else if (millis() - lastSpeechTime > 5000) {
            jammerAllowed = false;
            digitalWrite(LED_PIN, LOW);
        }
        
        Serial.printf("[AI] Speech Prob: %.2f | Jammer: %s\n", prob, jammerAllowed ? "ON" : "OFF"); // Debug output
    }
}
