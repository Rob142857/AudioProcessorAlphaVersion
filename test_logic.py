# Test the no speech detection logic
test_text = '.'
print(f'Original text: "{test_text}"')
print(f'After strip(): "{test_text.strip()}"')
print(f'Is empty after strip: {not test_text.strip()}')

# This is the logic from the code
if not test_text.strip():
    result = '[No speech detected]'
else:
    result = test_text.strip()

print(f'Final result: "{result}"')

# The issue is that Whisper is producing valid output but it's minimal
# Let's test with some other examples
test_cases = ['', '   ', '.', 'hello', 'hello world']
for case in test_cases:
    stripped = case.strip()
    is_empty = not stripped
    final = '[No speech detected]' if is_empty else stripped
    print(f'Test case "{case}": stripped="{stripped}", is_empty={is_empty}, final="{final}"')
