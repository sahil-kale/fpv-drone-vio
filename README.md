# mte-546-proj

## Setup
Unix/WSL:
- Create a new python virtual environment 
- Run `bash scripts/setup.sh`

Windows:
- Install PlatformIO CLI
- Run `python3 -m pip install -r scripts/requirements.txt`

## Building
- Run `python3 scripts/build.py`
Flags: `--deploy` to deploy to esp32

## Analysis
- Run `python3 analysis/record.py` to record audio from the ESP32
- Run `python3 analysis/plot_raw.py` for plotting or waveform generation (run `--help` for arg options)

## Notes
Fiddling with buffer size and sample rate can help with noise flickering. ESP32 recommends setting up ISR callbacks to help minimize this (possible improvement for u to implement if the popping is bad @dhruv)
