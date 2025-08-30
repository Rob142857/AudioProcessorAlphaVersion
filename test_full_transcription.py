import sys
sys.path.append('.')
from transcribe_aggressive import transcribe_file_aggressive

# Test the actual transcription process
input_path = r'C:\Windows\Media\tada.wav'

print('Testing full transcription process on Windows tada.wav...')
print(f'Input: {input_path}')

try:
    result_path = transcribe_file_aggressive(input_path, model_name="medium")
    print(f'Success! Output: {result_path}')

    # Read the result
    with open(result_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f'Content: "{content}"')
    print(f'Length: {len(content)}')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
