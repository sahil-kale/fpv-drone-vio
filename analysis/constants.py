NUM_CHANNELS = 2

UART_BUFFER_READ_SIZE_BYTES = 2048 * 8 * 2 # maybe play with this if running into buffer issues
SAMPLE_RATE_PCM_HZ = 16000
decode_struct_str = "<hhhhhhhh"
LOG_DIR = "logs"