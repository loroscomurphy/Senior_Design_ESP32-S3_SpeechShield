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
// HARDWARE PINS (Logan)  
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
// CONFIG (Logan)  
// =============================================================================  
#define I2S_BUFFER_SIZE     512  
#define I2S_DMA_BUF_COUNT   4  
#define AUDIO_BUFFER_LEN    16000  
#define USE_TFLITE_MODEL    true  
  
// Noise gate uses actual signal energy.  
#define NOISE_GATE_THRESH   0.000001f  
#define ACTIVE_SPEECH_THRESH 0.80f  
  
// =============================================================================  
// GLOBALS (Logan)  
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
// FFT (Logan)  
// =============================================================================  
static float tw_re[kVadFftSize / 2];  
static float tw_im[kVadFftSize / 2];  
static float hann[kVadFftSize];  
static float fft_buf_re[kVadFftSize];  
static float fft_buf_im[kVadFftSize];  
  
static void fft_init() {  
    for (int k = 0; k < kVadFftSize / 2; ++k) {  
        const float ang = -2.0f * 3.14159265f * k / kVadFftSize;  
        tw_re[k] = cosf(ang);  
        tw_im[k] = sinf(ang);  
    }  
    for (int n = 0; n < kVadFftSize; ++n)  
        hann[n] = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * n / (kVadFftSize - 1)));  
}  
  
static void fft_run() {  
    // Bit-reversal  
    for (int i = 1, j = 0; i < kVadFftSize; ++i) {  
        int bit = kVadFftSize >> 1;  
        for (; j & bit; bit >>= 1) j ^= bit;  
        j ^= bit;  
        if (i < j) {  
            float t = fft_buf_re[i]; fft_buf_re[i] = fft_buf_re[j]; fft_buf_re[j] = t;  
            t = fft_buf_im[i]; fft_buf_im[i] = fft_buf_im[j]; fft_buf_im[j] = t;  
        }  
    }  
  
    // Butterfly passes using precomputed twiddles  
    for (int len = 2; len <= kVadFftSize; len <<= 1) {  
        const int stride = kVadFftSize / len;  
        for (int i = 0; i < kVadFftSize; i += len) {  
            for (int j = 0; j < len / 2; ++j) {  
                const int a = i + j, b = i + j + len / 2;  
                const float wr = tw_re[j * stride], wi = tw_im[j * stride];  
                const float vr = wr * fft_buf_re[b] - wi * fft_buf_im[b];  
                const float vi = wr * fft_buf_im[b] + wi * fft_buf_re[b];  
                fft_buf_re[b] = fft_buf_re[a] - vr;  
                fft_buf_im[b] = fft_buf_im[a] - vi;  
                fft_buf_re[a] += vr;  
                fft_buf_im[a] += vi;  
            }  
        }  
    }  
}  
  
// =============================================================================  
// MFCC + Noise Gate (Logan + Eric)  
// =============================================================================  
static bool compute_features() {  
    // Simple energy check first  
    float total_energy = 0.0f;  
    for (int i = 0; i < AUDIO_BUFFER_LEN; ++i) {  
        float s = static_cast<float>(audio_buffer[i]);  
        total_energy += s * s;  
    }  
    if ((total_energy / AUDIO_BUFFER_LEN) < NOISE_GATE_THRESH) {  
        return false;  
    }  
  
    float mag[257];  
    float log_mel[kVadMfccFeatures];  
  
    auto hz_to_mel = [](float f) { return 2595.0f * log10f(1.0f + f / 700.0f); };  
    auto mel_to_hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };  
  
    float mel_hz[15];  
    const float mel_min = hz_to_mel(0.0f);  
    const float mel_max = hz_to_mel(8000.0f);  
    const float mel_step = (mel_max - mel_min) / 14.0f;  
    for (int i = 0; i < 15; ++i)  
        mel_hz[i] = mel_to_hz(mel_min + mel_step * i);  
  
    const float bin_scale = static_cast<float>(kVadFftSize) / 16000.0f;  
    auto get_bin = [&](float hz) { return static_cast<int>(hz * bin_scale + 0.5f); };  
  
    for (int frame = 0; frame < kVadTimeFrames; ++frame) {  
        const int start = frame * kVadHopLength;  
  
        for (int n = 0; n < kVadFftSize; ++n) {  
            fft_buf_re[n] = static_cast<float>(audio_buffer[start + n]) * hann[n];  
            fft_buf_im[n] = 0.0f;  
        }  
        fft_run();  
  
        for (int k = 0; k <= 256; ++k)  
            mag[k] = sqrtf(fft_buf_re[k] * fft_buf_re[k] + fft_buf_im[k] * fft_buf_im[k]);  
  
        for (int m = 0; m < kVadMfccFeatures; ++m) {  
            int lo  = constrain(get_bin(mel_hz[m]),     0, 256);  
            int mid = constrain(get_bin(mel_hz[m + 1]), 0, 256);  
            int hi  = constrain(get_bin(mel_hz[m + 2]), 0, 256);  
  
            float sum = 0.0f;  
            if (mid > lo)  
                for (int k = lo; k <= mid; ++k)  
                    sum += mag[k] * static_cast<float>(k - lo) / static_cast<float>(mid - lo);  
            if (hi > mid)  
                for (int k = mid + 1; k <= hi; ++k)  
                    sum += mag[k] * static_cast<float>(hi - k) / static_cast<float>(hi - mid);  
  
            log_mel[m] = logf(sum + 1e-6f);  
        }  
  
        const float pi_over_2m = 3.14159265f / (2.0f * kVadMfccFeatures);  
        for (int i = 0; i < kVadMfccFeatures; ++i) {  
            float s = 0.0f;  
            for (int j = 0; j < kVadMfccFeatures; ++j)  
                s += log_mel[j] * cosf(pi_over_2m * i * (2.0f * j + 1.0f));  
  
            int q = static_cast<int>(s / kVadInputScale + kVadInputZeroPoint);  
            input_tensor->data.int8[frame * kVadMfccFeatures + i] = (int8_t)constrain(q, -128, 127);  
        }  
    }  
    return true;  
}  
  
// =============================================================================  
// SETUP (Logan + Eric)  
// =============================================================================  
void setup() {  
    Serial.begin(115200);  
    delay(1000);  
  
    neopixelWrite(RGB_LED_PIN, 0, 0, 0);  
  
    // (Logan) Memory Allocation - prefers PSRAM  
    audio_buffer = (int16_t*)heap_caps_malloc(AUDIO_BUFFER_LEN * sizeof(int16_t),   
                                              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);  
    if (!audio_buffer) {  
        audio_buffer = (int16_t*)malloc(AUDIO_BUFFER_LEN * sizeof(int16_t));  
    }  
  
    if (!audio_buffer) {  
        while(1) delay(1000);  
    }  
  
    SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN);  
    gen.begin();  
    gen.setWave(AD9833_OFF);  
  
    // I2S setup  
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
  
    fft_init();  
  
    // (Eric) TinyML setup  
    if (USE_TFLITE_MODEL) {  
        const tflite::Model* model = tflite::GetModel(vad_model_data);  
        static tflite::MicroErrorReporter error_reporter;  
        static tflite::AllOpsResolver resolver;  
        static tflite::MicroInterpreter static_interpreter(  
            model, resolver, tensor_arena, kArenaSize, &error_reporter);  
          
        interpreter = &static_interpreter;  
        interpreter->AllocateTensors();  
        input_tensor  = interpreter->input(0);  
        output_tensor = interpreter->output(0);  
    }  
  
    xTaskCreatePinnedToCore(JammerTask, "Jammer", 4096, NULL, 2, NULL, 1);  
    xTaskCreatePinnedToCore(AITask,     "Brain",  14000, NULL, 1, NULL, 0);  
}  
  
void loop() {   
    vTaskDelete(NULL);   
}  
  
// =============================================================================  
// CORE 1 - JAMMER (Logan)  
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
// CORE 0 - AI (Logan + Eric)  
// =============================================================================  
void AITask(void* pv) {  
    size_t bytesRead;  
  
    for (;;) {  
        currentState = STATE_LISTENING;  
        vTaskDelay(pdMS_TO_TICKS(50));  
        i2s_zero_dma_buffer(I2S_NUM_0);  
  
        i2s_read(I2S_NUM_0, audio_buffer, AUDIO_BUFFER_LEN * sizeof(int16_t),   
                &bytesRead, portMAX_DELAY);  
  
        currentState = STATE_JAMMING;  
        float speech_prob = 0.0f;  
  
        if (USE_TFLITE_MODEL) {  
            if (compute_features()) {  
                if (interpreter->Invoke() == kTfLiteOk) {  
                    speech_prob = (output_tensor->data.int8[0] - kVadOutputZeroPoint)   
                                  * kVadOutputScale;  
                }  
            }  
        }  
  
        if (speech_prob > ACTIVE_SPEECH_THRESH) {  
            jammerAllowed.store(true);  
            lastSpeechTime = millis();  
        }   
        else if (millis() - lastSpeechTime > 3000) {  
            jammerAllowed.store(false);  
            vTaskDelay(pdMS_TO_TICKS(200));  
        }  
    }  
}  
