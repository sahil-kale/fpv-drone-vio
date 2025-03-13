#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/uart.h"
#include "driver/i2s_pdm.h"

#define TAG "MAIN"

#define I2S_NUM (0)
#define I2S_PDM_CLK_GPIO ((gpio_num_t)9U)
#define I2S_PDM_DATA0_GPIO ((gpio_num_t)10U)
#define I2S_PDM_DATA1_GPIO ((gpio_num_t)19U)
#define I2S_PDM_DATA2_GPIO ((gpio_num_t)12U)
#define I2S_PDM_DATA3_GPIO ((gpio_num_t)13U)

#define PDM_RX_FREQ_HZ (16000) // PDM clock frequency

static i2s_chan_handle_t rx_chan; 
static i2s_chan_config_t rx_chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);

constexpr static i2s_pdm_rx_gpio_config_t I2S_PDM_GPIO_CFG = {
    .clk = I2S_PDM_CLK_GPIO,
    .dins = {
        I2S_PDM_DATA0_GPIO,
        //I2S_PDM_DATA1_GPIO,
        //I2S_PDM_DATA2_GPIO,
        //I2S_PDM_DATA3_GPIO,
    },
    .invert_flags = {
        .clk_inv = false,
    },
};

constexpr static size_t NUM_PDM_DIN = sizeof(I2S_PDM_GPIO_CFG.dins) / sizeof(I2S_PDM_GPIO_CFG.dins[0]);

#define I2S_DATA_BUF_SIZE (2048 * 8)
static int16_t I2S_DATA_BUF[I2S_DATA_BUF_SIZE] = {0};
constexpr size_t I2S_DATA_BUF_SIZE_BYTES = sizeof(I2S_DATA_BUF);

#define MAIN_COMM_UART_NUM UART_NUM_0

void init_i2s_pdm(void)
{
    ESP_ERROR_CHECK(i2s_new_channel(&rx_chan_cfg, NULL, &rx_chan));

    i2s_pdm_rx_config_t pdm_rx_cfg = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(PDM_RX_FREQ_HZ),
        /* The data bit-width of PDM mode is fixed to 16 */
        // For the target that not support PDM-to-PCM format, we can only receive RAW PDM data format
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = I2S_PDM_GPIO_CFG,
    };

#if 0
    pdm_rx_cfg.slot_cfg.slot_mask = I2S_PDM_SLOT_BOTH;

    switch (NUM_PDM_DIN) {
        case 1:
            break;
        case 2:
            pdm_rx_cfg.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)(I2S_PDM_RX_LINE1_SLOT_RIGHT | I2S_PDM_RX_LINE1_SLOT_LEFT | I2S_PDM_SLOT_BOTH);
            break;
        case 3:
            pdm_rx_cfg.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)(I2S_PDM_RX_LINE2_SLOT_RIGHT | I2S_PDM_RX_LINE2_SLOT_LEFT | I2S_PDM_RX_LINE1_SLOT_RIGHT | I2S_PDM_RX_LINE1_SLOT_LEFT | I2S_PDM_SLOT_BOTH);
            break;
        case 4:
            pdm_rx_cfg.slot_cfg.slot_mask = (i2s_pdm_slot_mask_t)I2S_PDM_LINE_SLOT_ALL;
            break;
        default:
            ESP_LOGE(TAG, "Invalid number of PDM data input GPIOs");
            return;
    }
    #endif

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
            // just blast the data to UART
            for (size_t i = 0; i < r_bytes / sizeof(int16_t); i++) {
                uart_write_bytes(MAIN_COMM_UART_NUM, (const char *)&r_buf[i], sizeof(int16_t));
            }
            //ESP_LOGI(TAG, "Received %d bytes", r_bytes);
        } else {
            ESP_LOGW(TAG, "No data received within timeout");
        }
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
    uart_param_config(MAIN_COMM_UART_NUM, &uart_config);
    uart_driver_install(MAIN_COMM_UART_NUM, I2S_DATA_BUF_SIZE_BYTES * 3, 0, 0, NULL, 0);


    xTaskCreate(i2s_pdm_task, "I2S PDM Task", 4096, NULL, 1, NULL);
}
