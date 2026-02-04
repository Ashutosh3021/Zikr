"""
Text preprocessing utilities for the TTS-STT system.
Implements text normalization, tokenization, and phonemization as specified.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np

from ..utils.logger import get_data_logger
from ..utils.config import DataConfig

logger = get_data_logger()

class TextNormalizer:
    """Text normalization pipeline for TTS and STT systems."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._setup_normalization_rules()
        logger.info("TextNormalizer initialized")
    
    def _setup_normalization_rules(self):
        """Setup normalization rules and patterns."""
        # Number conversion patterns
        self.number_patterns = {
            r'\b(\d{1,3})(,\d{3})*(\.\d+)?\b': self._expand_number,
            r'\b\d+\b': self._number_to_words
        }
        
        # Abbreviation mappings
        self.abbreviations = {
            'dr.': 'doctor',
            'mr.': 'mister',
            'mrs.': 'missus',
            'ms.': 'miss',
            'prof.': 'professor',
            'inc.': 'incorporated',
            'ltd.': 'limited',
            'corp.': 'corporation',
            'co.': 'company',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'i.e.': 'that is',
            'e.g.': 'for example',
            'u.s.': 'united states',
            'u.k.': 'united kingdom',
        }
        
        # Time and date patterns
        self.time_patterns = {
            r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b': self._expand_time,
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b': self._expand_date,
        }
        
        # Currency patterns
        self.currency_patterns = {
            r'\$(\d+(?:\.\d{2})?)': r'\1 dollars',
            r'£(\d+(?:\.\d{2})?)': r'\1 pounds',
            r'€(\d+(?:\.\d{2})?)': r'\1 euros',
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Apply comprehensive text normalization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Normalize Unicode characters
            text = unicodedata.normalize('NFKD', text)
            
            # Remove or replace special characters
            text = self._clean_special_characters(text)
            
            # Expand abbreviations
            text = self._expand_abbreviations(text)
            
            # Handle numbers
            text = self._normalize_numbers(text)
            
            # Handle time and dates
            text = self._normalize_time_dates(text)
            
            # Handle currency
            text = self._normalize_currency(text)
            
            # Clean up whitespace
            text = self._clean_whitespace(text)
            
            logger.debug(f"Text normalized: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            raise
    
    def _clean_special_characters(self, text: str) -> str:
        """Clean or replace special characters."""
        # Remove or replace problematic characters
        replacements = {
            '&': 'and',
            '@': 'at',
            '#': 'hash',
            '%': 'percent',
            '"': '',
            "'": '',
            '...': '.',
            '…': '.',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-alphanumeric characters (except space and basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbr, expansion in self.abbreviations.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(abbr), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize numbers to words."""
        for pattern, handler in self.number_patterns.items():
            text = re.sub(pattern, handler, text)
        return text
    
    def _expand_number(self, match) -> str:
        """Expand a matched number to words."""
        number_str = match.group(0)
        # Remove commas and convert to float
        number_str = number_str.replace(',', '')
        try:
            number = float(number_str)
            return self._number_to_words_helper(number)
        except ValueError:
            return number_str
    
    def _number_to_words(self, match) -> str:
        """Convert simple number to words."""
        number = int(match.group(0))
        return self._number_to_words_helper(number)
    
    def _number_to_words_helper(self, number: Union[int, float]) -> str:
        """Helper function to convert number to words."""
        # This is a simplified implementation
        # In production, you might want to use a dedicated library like num2words
        if isinstance(number, float):
            # Handle decimal numbers
            integer_part = int(number)
            decimal_part = int((number - integer_part) * 100)
            if decimal_part == 0:
                return self._int_to_words(integer_part)
            else:
                return f"{self._int_to_words(integer_part)} point {self._int_to_words(decimal_part)}"
        else:
            return self._int_to_words(number)
    
    def _int_to_words(self, num: int) -> str:
        """Convert integer to words."""
        if num == 0:
            return "zero"
        elif num < 0:
            return f"negative {self._int_to_words(-num)}"
        elif num < 20:
            return ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen"][num]
        elif num < 100:
            tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            return tens[num // 10] + ("" if num % 10 == 0 else f" {self._int_to_words(num % 10)}")
        elif num < 1000:
            return self._int_to_words(num // 100) + " hundred" + ("" if num % 100 == 0 else f" {self._int_to_words(num % 100)}")
        else:
            # Handle larger numbers (thousands, millions, etc.)
            scales = [(1000000, "million"), (1000, "thousand")]
            for scale, name in scales:
                if num >= scale:
                    return self._int_to_words(num // scale) + f" {name}" + ("" if num % scale == 0 else f" {self._int_to_words(num % scale)}")
            return str(num)  # Fallback for very large numbers
    
    def _normalize_time_dates(self, text: str) -> str:
        """Normalize time and date expressions."""
        for pattern, handler in self.time_patterns.items():
            text = re.sub(pattern, handler, text)
        return text
    
    def _expand_time(self, match) -> str:
        """Expand time expression to words."""
        hours, minutes, am_pm = match.groups()
        hours = int(hours)
        minutes = int(minutes)
        
        # Convert 24-hour to 12-hour format
        period = ""
        if am_pm:
            period = f" {am_pm}"
        elif hours >= 12:
            period = " pm"
            if hours > 12:
                hours -= 12
        else:
            period = " am"
            if hours == 0:
                hours = 12
        
        if minutes == 0:
            return f"{self._int_to_words(hours)} o'clock{period}"
        else:
            return f"{self._int_to_words(hours)} {self._int_to_words(minutes)}{period}"
    
    def _expand_date(self, match) -> str:
        """Expand date expression to words."""
        month, day, year = match.groups()
        month_names = ["", "january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november", "december"]
        try:
            month_num = int(month)
            month_name = month_names[month_num] if 1 <= month_num <= 12 else month
            return f"{self._int_to_words(int(day))} {month_name} {self._int_to_words(int(year))}"
        except (ValueError, IndexError):
            return f"{month}/{day}/{year}"
    
    def _normalize_currency(self, text: str) -> str:
        """Normalize currency expressions."""
        for pattern, replacement in self.currency_patterns.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

class Tokenizer:
    """Text tokenization for model input."""
    
    def __init__(self, config: DataConfig, vocab_file: Optional[str] = None):
        self.config = config
        self.vocab_size = config.max_text_length
        self.vocab = {}
        self.inverse_vocab = {}
        
        if vocab_file:
            self.load_vocabulary(vocab_file)
        else:
            self._create_default_vocabulary()
        
        logger.info(f"Tokenizer initialized with vocab size: {len(self.vocab)}")
    
    def _create_default_vocabulary(self):
        """Create a basic character-level vocabulary."""
        # Basic English character vocabulary
        chars = list('abcdefghijklmnopqrstuvwxyz ')
        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        
        for i, token in enumerate(special_tokens + chars):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def load_vocabulary(self, vocab_file: str):
        """Load vocabulary from file."""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                tokens = [line.strip() for line in f if line.strip()]
            
            self.vocab = {}
            self.inverse_vocab = {}
            for i, token in enumerate(tokens):
                self.vocab[token] = i
                self.inverse_vocab[i] = token
                
            logger.info(f"Loaded vocabulary with {len(self.vocab)} tokens from {vocab_file}")
            
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {e}")
            self._create_default_vocabulary()
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        # Normalize text first
        normalizer = TextNormalizer(self.config)
        normalized_text = normalizer.normalize_text(text)
        
        # Convert to tokens
        tokens = []
        for char in normalized_text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])  # Unknown token
        
        # Add start and end tokens
        tokens = [self.vocab['<sos>']] + tokens + [self.vocab['<eos>']]
        
        # Pad or truncate to max length
        if len(tokens) > self.config.max_text_length:
            tokens = tokens[:self.config.max_text_length]
        else:
            tokens.extend([self.vocab['<pad>']] * (self.config.max_text_length - len(tokens)))
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token indices back to text."""
        # Remove padding and special tokens
        special_tokens = {self.vocab.get('<pad>', -1), 
                         self.vocab.get('<sos>', -1), 
                         self.vocab.get('<eos>', -1)}
        
        chars = []
        for token_id in token_ids:
            if token_id not in special_tokens and token_id in self.inverse_vocab:
                char = self.inverse_vocab[token_id]
                if char != '<unk>':
                    chars.append(char)
        
        return ''.join(chars)
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)

class Phonemizer:
    """Grapheme-to-phoneme conversion for better pronunciation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        # This would typically use a library like phonemizer or g2p-en
        # For now, we'll implement a simplified version
        self._setup_simple_g2p()
        logger.info("Phonemizer initialized")
    
    def _setup_simple_g2p(self):
        """Setup simple grapheme-to-phoneme rules."""
        # Basic English pronunciation rules (simplified)
        self.g2p_rules = {
            'a': 'ae', 'e': 'eh', 'i': 'ih', 'o': 'ow', 'u': 'uw',
            'ai': 'ay', 'ea': 'iy', 'ee': 'iy', 'oa': 'ow', 'oo': 'uw',
            'th': 'th', 'sh': 'sh', 'ch': 'ch', 'ph': 'f',
            'qu': 'kw', 'ck': 'k', 'ng': 'ng', 'gh': 'g'
        }
        
        # Word-specific exceptions
        self.exceptions = {
            'the': 'dh ah', 'and': 'ah n d', 'to': 't uw',
            'of': 'ah v', 'in': 'ih n', 'is': 'ih z'
        }
    
    def phonemize(self, text: str) -> str:
        """Convert text to phonemes."""
        # Normalize text first
        normalizer = TextNormalizer(self.config)
        normalized_text = normalizer.normalize_text(text)
        
        # Handle word exceptions
        words = normalized_text.split()
        phoneme_words = []
        
        for word in words:
            if word in self.exceptions:
                phoneme_words.append(self.exceptions[word])
            else:
                phoneme_words.append(self._word_to_phonemes(word))
        
        return ' '.join(phoneme_words)
    
    def _word_to_phonemes(self, word: str) -> str:
        """Convert individual word to phonemes."""
        # Apply rules from right to left (longer patterns first)
        sorted_rules = sorted(self.g2p_rules.items(), key=lambda x: len(x[0]), reverse=True)
        
        result = word
        for grapheme, phoneme in sorted_rules:
            result = result.replace(grapheme, phoneme)
        
        # Handle remaining single characters
        remaining_chars = set(result) - set('aeiou')
        for char in remaining_chars:
            if char in 'bcdfgjklmnpqrstvwxyz':
                result = result.replace(char, char)
        
        return result

# Utility functions
def create_text_preprocessor(config: DataConfig) -> tuple:
    """Factory function to create text preprocessing components."""
    normalizer = TextNormalizer(config)
    tokenizer = Tokenizer(config)
    phonemizer = Phonemizer(config)
    return normalizer, tokenizer, phonemizer

def preprocess_text_pipeline(text: str, config: DataConfig) -> Dict[str, Union[str, List[int]]]:
    """Complete text preprocessing pipeline."""
    normalizer, tokenizer, phonemizer = create_text_preprocessor(config)
    
    normalized_text = normalizer.normalize_text(text)
    token_ids = tokenizer.tokenize(normalized_text)
    phonemes = phonemizer.phonemize(text)
    
    return {
        'original': text,
        'normalized': normalized_text,
        'tokens': token_ids,
        'phonemes': phonemes,
        'token_length': len([t for t in token_ids if t != tokenizer.vocab['<pad>']])
    }