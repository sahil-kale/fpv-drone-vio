import serial
import argparse
import struct
import os
import time
from constants import *

def create_log_file():
    """Creates a log file with the appropriate naming convention."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"AUDIO_RECORD_{timestamp}.raw"
    filepath = os.path.join(LOG_DIR, filename)
    return open(filepath, "wb")

def read_serial(port):
    try:
        with serial.Serial(port, baudrate=4000000, timeout=1) as ser:
            ser.reset_input_buffer()  # Flush serial buffer on init
            print(f"Listening on {port}...")
            log_file = create_log_file()
            
            while True:
                data = ser.read(UART_BUFFER_READ_SIZE_BYTES)
                if not data:
                    continue
                
                print(f"Read {len(data)} bytes from {port}")
                
                # write to log file
                log_file.write(data)
    
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read from a serial port continuously and log to a file.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port to read from (default: /dev/ttyACM0)")
    args = parser.parse_args()
    read_serial(args.port)