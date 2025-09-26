"""
Custom Dictionary Manager
Command-line tool for managing the custom word dictionary
"""

import sys
import argparse
from custom_dictionary import CustomDictionary

def add_word(incorrect: str, correct: str, section: str = "## Custom Additions"):
    """Add a new word substitution to the dictionary."""
    dict_system = CustomDictionary()
    dict_system.save_substitution(incorrect, correct, section)
    print(f"✅ Added: '{incorrect}' -> '{correct}'")

def test_word(word: str):
    """Test if a word would be substituted."""
    dict_system = CustomDictionary()
    test_text = f"This is a test with {word} in it."
    result = dict_system.apply_substitutions(test_text)
    
    if result != test_text:
        print(f"✅ '{word}' would be substituted in: {result}")
    else:
        print(f"ℹ️  '{word}' would not be substituted")

def list_words():
    """List all words in the dictionary."""
    dict_system = CustomDictionary()
    print(f"📚 Dictionary contains {len(dict_system.substitutions)} substitutions:")
    
    for incorrect, correct in dict_system.substitutions:
        print(f"  {incorrect} -> {correct}")

def test_text(text: str):
    """Test custom dictionary on provided text."""
    dict_system = CustomDictionary()
    print(f"Original: {text}")
    result = dict_system.apply_substitutions(text)
    print(f"Result:   {result}")
    
    if result != text:
        print("✅ Substitutions were applied")
    else:
        print("ℹ️  No substitutions applied")

def main():
    parser = argparse.ArgumentParser(
        description="Custom Dictionary Manager for Audio Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dict_manager.py add "gurdief" "Gurdjieff"
  python dict_manager.py add "wyspensky" "Ouspensky" --section "## Proper Names"
  python dict_manager.py test "gurdief"
  python dict_manager.py list
  python dict_manager.py test-text "gurdief was a teacher"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add word command
    add_parser = subparsers.add_parser('add', help='Add a new word substitution')
    add_parser.add_argument('incorrect', help='Word to be replaced')
    add_parser.add_argument('correct', help='Replacement word')
    add_parser.add_argument('--section', default='## Custom Additions', 
                           help='Section to add the word to (default: ## Custom Additions)')
    
    # Test word command
    test_parser = subparsers.add_parser('test', help='Test if a word would be substituted')
    test_parser.add_argument('word', help='Word to test')
    
    # List words command
    subparsers.add_parser('list', help='List all words in the dictionary')
    
    # Test text command
    test_text_parser = subparsers.add_parser('test-text', help='Test dictionary on text')
    test_text_parser.add_argument('text', help='Text to test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'add':
            add_word(args.incorrect, args.correct, args.section)
        elif args.command == 'test':
            test_word(args.word)
        elif args.command == 'list':
            list_words()
        elif args.command == 'test-text':
            test_text(args.text)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()