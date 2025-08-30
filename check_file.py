import os
import subprocess
import json

file_path = r'C:/Users/RobertEvans/OneDrive - RME Solutions Technology/_PG Completed Recordings 84-97/1994 MW/0525 The Atmosphere and its significations.mp3'

print('=== Alternative File Analysis ===')

# Try ffprobe to get file info
try:
    result = subprocess.run([
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', file_path
    ], capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        data = json.loads(result.stdout)
        format_info = data.get('format', {})
        duration = float(format_info.get('duration', 0))

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)

        print(f'Duration: {hours:02d}:{minutes:02d}:{seconds:02d} ({duration:.1f} seconds)')
        print(f'Format: {format_info.get("format_name", "unknown")}')
        print(f'Bitrate: {format_info.get("bit_rate", "unknown")}')

        # Check streams
        streams = data.get('streams', [])
        for stream in streams:
            if stream.get('codec_type') == 'audio':
                print(f'Audio codec: {stream.get("codec_name", "unknown")}')
                print(f'Sample rate: {stream.get("sample_rate", "unknown")}')
                print(f'Channels: {stream.get("channels", "unknown")}')
                break
    else:
        print(f'ffprobe failed: {result.stderr}')

except Exception as e:
    print(f'ffprobe error: {e}')

# Check for any output files
downloads = os.path.expanduser('~/Downloads')
output_files = [f for f in os.listdir(downloads) if 'atmosphere' in f.lower() and (f.endswith('.txt') or f.endswith('.docx'))]
print(f'\nOutput files found: {len(output_files)}')
for f in output_files:
    print(f'  {f}')
