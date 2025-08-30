import sys
sys.path.append('.')
import whisper
import tempfile
import subprocess
import os

# Test transcribing the Windows tada.wav file directly
input_path = r'C:\Windows\Media\tada.wav'

print('Testing direct transcription of Windows tada.wav...')
print(f'Input file: {input_path}')

# Load model
model = whisper.load_model('medium', device='cpu')
print(f'Model loaded: {model}')

# Transcribe directly
result = model.transcribe(input_path, language=None,
                         compression_ratio_threshold=float('inf'),
                         logprob_threshold=-1.0,
                         no_speech_threshold=0.1,
                         condition_on_previous_text=False,
                         temperature=0.0)

text = str(result.get('text', '')).strip()
print(f'Direct transcription result: "{text}"')
print(f'Text length: {len(text)} characters')

# Also test with different no_speech_threshold values
print('\nTesting with different no_speech_threshold values:')
for threshold in [0.1, 0.3, 0.5, 0.8]:
    result = model.transcribe(input_path, language=None,
                             compression_ratio_threshold=float('inf'),
                             logprob_threshold=-1.0,
                             no_speech_threshold=threshold,
                             condition_on_previous_text=False,
                             temperature=0.0)
    text = str(result.get('text', '')).strip()
    print(f'  Threshold {threshold}: "{text}" (len: {len(text)})')
