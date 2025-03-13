NUM_CHANNELS = 4

UART_BUFFER_READ_SIZE_BYTES = 2048 * 8 * 2# maybe play with this if running into buffer issues
BASE_SAMPLE_RATE_PCM_HZ = 16000
SAMPLE_RATE_PCM_HZ = BASE_SAMPLE_RATE_PCM_HZ
decode_struct_str = "<hhhhhhhh"
LOG_DIR = "logs"