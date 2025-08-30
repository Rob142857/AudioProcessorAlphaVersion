import sys
sys.path.append('.')
from transcribe import get_media_duration
import subprocess
import json

# Check the Windows tada.wav file properties
input_path = r'C:\Windows\Media\tada.wav'
print(f'Checking audio file: {input_path}')

# Get duration
duration = get_media_duration(input_path)
print(f'Duration: {duration} seconds')

# Use ffprobe to get more detailed audio information
try:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print('Audio format information:')
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                print(f'  Codec: {stream.get("codec_name")}')
                print(f'  Sample rate: {stream.get("sample_rate")}')
                print(f'  Channels: {stream.get("channels")}')
                print(f'  Bitrate: {stream.get("bit_rate")}')
                break
        format_info = data.get('format', {})
        print(f'  Duration: {format_info.get("duration")}')
        print(f'  Size: {format_info.get("size")} bytes')
except Exception as e:
    print(f'Could not get detailed audio info: {e}')
