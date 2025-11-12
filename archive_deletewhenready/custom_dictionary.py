"""
Custom Dictionary System for Audio Transcription
Handles word substitutions from a markdown dictionary file
"""

import re
import os
from typing import Dict, List, Tuple
from pathlib import Path

class CustomDictionary:
    def __init__(self, dictionary_path: str = "custom_dictionary.md"):
        """
        Initialize the custom dictionary system.
        
        Args:
            dictionary_path: Path to the markdown dictionary file
        """
        self.dictionary_path = dictionary_path
        self.substitutions: List[Tuple[str, str]] = []
        self.load_dictionary()
    
    def load_dictionary(self) -> None:
        """Load substitutions from the markdown dictionary file."""
        if not os.path.exists(self.dictionary_path):
            print(f"⚠️  Custom dictionary not found: {self.dictionary_path}")
            return
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract substitution patterns
            substitutions = []
            
            # Look for lines with the pattern "word -> replacement"
            pattern = r'(.+?)\s*->\s*(.+?)$'
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Skip empty lines, comments, headers, and markdown formatting
                if not line or line.startswith('#') or line.startswith('```') or line.startswith('-'):
                    continue
                
                # Match substitution pattern
                match = re.match(pattern, line)
                if match:
                    incorrect = match.group(1).strip()
                    correct = match.group(2).strip()
                    
                    # Skip if either side is empty
                    if incorrect and correct:
                        substitutions.append((incorrect.lower(), correct))
            
            # Sort by length (longest first) to handle overlapping matches correctly
            self.substitutions = sorted(substitutions, key=lambda x: len(x[0]), reverse=True)
            
            print(f"📚 Loaded {len(self.substitutions)} custom word substitutions")
            
        except Exception as e:
            print(f"❌ Error loading custom dictionary: {e}")
            self.substitutions = []
    
    def apply_substitutions(self, text: str) -> str:
        """
        Apply custom word substitutions to text.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with substitutions applied
        """
        if not self.substitutions:
            return text
        
        modified_text = text
        replacements_made = 0
        
        for incorrect, correct in self.substitutions:
            # Enhanced pattern to avoid matching inside contractions
            # Negative lookbehind to avoid matches after apostrophe
            # Negative lookahead to avoid matches before apostrophe
            escaped_word = re.escape(incorrect)
            pattern = r'(?<!\w)(?<!\')' + escaped_word + r'(?!\')(?!\w)'
            
            # Count matches before replacement
            matches = len(re.findall(pattern, modified_text, re.IGNORECASE))
            
            if matches > 0:
                modified_text = re.sub(pattern, lambda m: correct if m.group(0).islower() else correct.capitalize() if m.group(0).istitle() else correct.upper(), modified_text, flags=re.IGNORECASE)
                replacements_made += matches
        
        if replacements_made > 0:
            print(f"📝 Applied {replacements_made} custom word substitutions")
        
        return modified_text
    
    def add_substitution(self, incorrect: str, correct: str) -> None:
        """
        Add a new substitution to the dictionary (in memory only).
        
        Args:
            incorrect: Word to be replaced
            correct: Replacement word
        """
        self.substitutions.append((incorrect.lower(), correct))
        # Resort to maintain longest-first order
        self.substitutions = sorted(self.substitutions, key=lambda x: len(x[0]), reverse=True)
    
    def save_substitution(self, incorrect: str, correct: str, section: str = "## Custom Additions") -> None:
        """
        Add a substitution and save it to the markdown file.
        
        Args:
            incorrect: Word to be replaced
            correct: Replacement word
            section: Section to add the substitution under
        """
        try:
            # Add to memory
            self.add_substitution(incorrect, correct)
            
            # Add to file
            if os.path.exists(self.dictionary_path):
                with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if the section exists
                if section not in content:
                    content += f"\n\n{section}\n```\n{incorrect} -> {correct}\n```\n"
                else:
                    # Find the section and add the substitution
                    section_pattern = f"{section}.*?```(.*?)```"
                    match = re.search(section_pattern, content, re.DOTALL)
                    if match:
                        existing_substitutions = match.group(1)
                        new_substitutions = existing_substitutions + f"{incorrect} -> {correct}\n"
                        content = content.replace(match.group(1), new_substitutions)
                    else:
                        # Section exists but no code block, add one
                        content = content.replace(section, f"{section}\n```\n{incorrect} -> {correct}\n```")
                
                with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Added substitution: '{incorrect}' -> '{correct}' to {self.dictionary_path}")
            
        except Exception as e:
            print(f"❌ Error saving substitution: {e}")
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the dictionary."""
        return {
            "total_substitutions": len(self.substitutions),
            "average_word_length": sum(len(s[0]) for s in self.substitutions) // max(len(self.substitutions), 1)
        }

def test_custom_dictionary():
    """Test the custom dictionary system."""
    print("🧪 Testing Custom Dictionary System")
    
    # Create test dictionary
    dict_system = CustomDictionary()
    
    # Test text
    test_text = """
    Gurdief was a great teacher, as was Wyspensky. They studied the I Ching and 
    concepts from the New Testament. Shakespeare wrote about these themes too.
    The term eschatology appears in religious contexts.
    """
    
    print("Original text:")
    print(test_text)
    
    processed_text = dict_system.apply_substitutions(test_text)
    
    print("\nProcessed text:")
    print(processed_text)
    
    print(f"\nDictionary statistics: {dict_system.get_statistics()}")

if __name__ == "__main__":
    test_custom_dictionary()