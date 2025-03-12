#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/uart.h"
#include "driver/i2s_pdm.h"

#define TAG "MAIN"

#define I2S_NUM (0)
#define I2S_PDM_CLK_GPIO ((gpio_num_t)10U)
#define I2S_PDM_DATA0_GPIO ((gpio_num_t)11U)

#define PDM_RX_FREQ_HZ (16000) // PDM clock frequency

static i2s_chan_handle_t rx_chan; 
static i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);

static int16_t I2S_DATA_BUF[2048] = {0};

void init_i2s_pdm(void)
{
#if SOC_I2S_SUPPORTS_PDM2PCM
    ESP_LOGI(TAG, "I2S PDM RX example (receiving data in PCM format)");
#else
    ESP_LOGI(TAG, "I2S PDM RX example (receiving data in raw PDM format)");
#endif  // SOC_I2S_SUPPORTS_PDM2PCM
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(PDM_RX_FREQ_HZ),
        /* The data bit-width of PDM mode is fixed to 16 */
#if SOC_I2S_SUPPORTS_PDM2PCM
        .slot_cfg = I2S_PDM_RX_SLOT_PCM_FMT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
#else
        // For the target that not support PDM-to-PCM format, we can only receive RAW PDM data format
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
#endif  // SOC_I2S_SUPPORTS_PDM2PCM
        .gpio_cfg = {
            .clk = I2S_PDM_CLK_GPIO,
#if SOC_I2S_PDM_MAX_RX_LINES == 4
            .dins = {
                I2S_PDM_DATA0_GPIO
                //EXAMPLE_PDM_RX_DIN1_IO,
                //EXAMPLE_PDM_RX_DIN2_IO,
                //EXAMPLE_PDM_RX_DIN3_IO,
            },
#else
            #error "Shouldn't be in here..."
#endif
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };

    // TODO: slot omde should be stereo, and slot mask should be all
    pdm_rx_cfg.slot_cfg.slot_mode = I2S_SLOT_MODE_MONO;
    pdm_rx_cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_pdm_rx_mode(rx_chan, &pdm_rx_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));


}

void i2s_pdm_task(void *pvParameter)
{
    init_i2s_pdm();
    while (1) {
        size_t r_bytes = 0;
        if (i2s_channel_read(rx_chan, I2S_DATA_BUF, sizeof(I2S_DATA_BUF), &r_bytes, 1000) == ESP_OK) {
            int16_t *r_buf = (int16_t *)I2S_DATA_BUF;
            // Successfully received data
            printf("Read Task: i2s read %d bytes\n-----------------------------------\n", r_bytes);
            printf("[0] %d [1] %d [2] %d [3] %d\n[4] %d [5] %d [6] %d [7] %d\n\n",
                   r_buf[0], r_buf[1], r_buf[2], r_buf[3], r_buf[4], r_buf[5], r_buf[6], r_buf[7]);
            // Process the received data here
        } else {
            ESP_LOGW(TAG, "No data received within timeout");
        }
    }
}

void blinkTask(void *pvParameter) {
    while (1) {
        ESP_LOGI(TAG, "ESP-IDF Running!");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting ESP-IDF application...");

    // Configure UART0 (TX: GPIO1, RX: GPIO3 by default on ESP32-S3)
    uart_config_t uart_config = {
        .baud_rate = 4000000,  // Set baud rate to 4,000,000
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    // Apply configuration to UART0
    uart_param_config(UART_NUM_0, &uart_config);
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);

    xTaskCreate(blinkTask, "Blink Task", 2048, NULL, 2, NULL);
    xTaskCreate(i2s_pdm_task, "I2S PDM Task", 4096, NULL, 1, NULL);
}
