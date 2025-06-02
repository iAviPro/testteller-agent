import pytest
import logging
from data_ingestion.text_splitter import TextSplitter

# Configure logging for tests if necessary (e.g., to see TextSplitter's own logs)
# logging.basicConfig(level=logging.DEBUG) # Optional: for debugging tests themselves

class TestTextSplitter:

    def test_split_text_default_params(self):
        """Test with default chunk_size and overlap."""
        splitter = TextSplitter() # chunk_size=500, overlap=50
        text = "a" * 1000
        chunks = splitter.split_text(text)
        assert len(chunks) == 3 # (500) + (500 starting at 450) + (100 starting at 900, from 500+450=950, remaining 50) -> no, (500) + (500 starting at 450) -> remaining 50. (500) + (50 from 450 to 500, then 450 from 500 to 950) -> (0-500), (450-950), (900-1000)
        # Chunk1: text[0:500]
        # Next start: 500 - 50 = 450
        # Chunk2: text[450:950]
        # Next start: 950 - 50 = 900
        # Chunk3: text[900:1000]
        assert chunks[0] == "a" * 500
        assert chunks[1] == "a" * 500
        assert chunks[2] == "a" * 100
        assert chunks[1].startswith("a" * 50) # Overlap part from chunk1
        assert chunks[2].startswith("a" * 50) # Overlap part from chunk2

    def test_split_text_custom_params(self):
        """Test with custom chunk_size and overlap."""
        splitter = TextSplitter(chunk_size=100, overlap=20)
        text = "b" * 250
        chunks = splitter.split_text(text)
        # Chunk1: text[0:100]
        # Next start: 100 - 20 = 80
        # Chunk2: text[80:180]
        # Next start: 180 - 20 = 160
        # Chunk3: text[160:250] (length 90)
        assert len(chunks) == 3
        assert chunks[0] == "b" * 100
        assert chunks[1] == "b" * 100
        assert chunks[2] == "b" * 90
        assert chunks[1].startswith("b" * 20)
        assert chunks[2].startswith("b" * 20)

    def test_empty_string(self):
        splitter = TextSplitter()
        assert splitter.split_text("") == []

    def test_text_shorter_than_chunk_size(self):
        splitter = TextSplitter(chunk_size=100)
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_text_equal_to_chunk_size(self):
        splitter = TextSplitter(chunk_size=20)
        text = "This text is 20 chr." # Exactly 20 characters
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_text_slightly_longer_than_chunk_size(self):
        splitter = TextSplitter(chunk_size=10, overlap=2)
        text = "abcdefghijklm" # 13 chars
        # C1: abcdefghij (0-10)
        # next_start = 10-2 = 8
        # C2: ijklm (8-13)
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "abcdefghij"
        assert chunks[1] == "ijklm"

    def test_no_overlap(self):
        splitter = TextSplitter(chunk_size=10, overlap=0)
        text = "abcdefghijklmnopqrstuvwxyz" # 26 chars
        chunks = splitter.split_text(text)
        # C1: abcdefghij (0-10)
        # next_start = 10-0 = 10
        # C2: klmnopqrst (10-20)
        # next_start = 20-0 = 20
        # C3: uvwxyz (20-26)
        assert len(chunks) == 3
        assert chunks[0] == "abcdefghij"
        assert chunks[1] == "klmnopqrst"
        assert chunks[2] == "uvwxyz"

    def test_with_overlap(self):
        splitter = TextSplitter(chunk_size=10, overlap=3)
        text = "abcdefghijklmno" # 15 chars
        chunks = splitter.split_text(text)
        # C1: abcdefghij (0-10)
        # next_start = 10-3 = 7
        # C2: hijklmno (7-15) (length 8)
        assert len(chunks) == 2
        assert chunks[0] == "abcdefghij"
        assert chunks[1] == "hijklmno"

    def test_overlap_equal_to_chunk_size_warning_and_behavior(self, caplog):
        """Test when overlap is equal to chunk_size."""
        # This should ideally be prevented by __init__ or handled gracefully.
        # The current __init__ logs a warning.
        # The split_text has a safeguard.
        with caplog.at_level(logging.WARNING, logger="data_ingestion.text_splitter"):
             splitter = TextSplitter(chunk_size=10, overlap=10)
        assert any("Overlap (10) is greater than or equal to chunk_size (10)" in message for message in caplog.messages)
        
        text = "abcdefghijklmnopqrst"
        caplog.clear() # Clear previous logs
        with caplog.at_level(logging.ERROR, logger="data_ingestion.text_splitter"):
            chunks = splitter.split_text(text)
        
        # Expects the safeguard to kick in
        assert len(chunks) == 1 
        assert chunks[0] == "abcdefghij"
        assert any("Overlap (10) is >= chunk_size (10)" in message for message in caplog.messages if "infinite loop" in message)


    def test_overlap_greater_than_chunk_size(self, caplog):
        """Test when overlap is greater than chunk_size."""
        with caplog.at_level(logging.WARNING, logger="data_ingestion.text_splitter"):
            splitter = TextSplitter(chunk_size=10, overlap=12)
        assert any("Overlap (12) is greater than or equal to chunk_size (10)" in message for message in caplog.messages)

        text = "abcdefghijklmnopqrst"
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="data_ingestion.text_splitter"):
            chunks = splitter.split_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == "abcdefghij"
        assert any("Overlap (12) is >= chunk_size (10)" in message for message in caplog.messages if "infinite loop" in message)

    def test_sum_of_chunk_lengths_minus_overlaps(self):
        text = "This is a longer test sentence to verify chunk integrity and overlap handling."
        original_len = len(text)
        
        chunk_size = 20
        overlap = 5
        splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
        chunks = splitter.split_text(text)

        reconstructed_len = 0
        if chunks:
            reconstructed_len += len(chunks[0])
            for i in range(1, len(chunks)):
                # Add length of non-overlapping part of the current chunk
                reconstructed_len += len(chunks[i]) - overlap
        
        # This assertion is tricky because the last chunk might be shorter.
        # A better check is to reconstruct the string.
        reconstructed_text = ""
        if chunks:
            reconstructed_text = chunks[0]
            for i in range(1, len(chunks)):
                # The start of chunk[i] should correspond to text that was 'overlap' characters from the end of chunk[i-1]'s content
                # This means chunk[i] starts with the last 'overlap' characters of the *effective content* covered by chunk[i-1]
                # More simply: chunk[i]'s first 'overlap' characters are a repeat.
                reconstructed_text += chunks[i][overlap:]
        
        assert reconstructed_text == text


    def test_chunk_types(self):
        splitter = TextSplitter()
        text = "A simple string."
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        if chunks: # If not empty
            assert all(isinstance(chunk, str) for chunk in chunks)

    def test_constructor_validation(self):
        """Test __init__ validation."""
        with pytest.raises(ValueError, match="Chunk size must be a positive integer."):
            TextSplitter(chunk_size=0)
        with pytest.raises(ValueError, match="Chunk size must be a positive integer."):
            TextSplitter(chunk_size=-10)
        with pytest.raises(ValueError, match="Overlap must be a non-negative integer."):
            TextSplitter(overlap=-1)

    def test_split_consistency_various_lengths(self):
        """Test with various text lengths relative to chunk_size and overlap."""
        splitter = TextSplitter(chunk_size=100, overlap=10)
        texts_to_test = {
            "short": "abc", # Shorter than chunk_size
            "exact": "a" * 100, # Equal to chunk_size
            "one_overlap": "a" * 150, # chunk1 (0-100), next_start=90, chunk2 (90-150)
            "multiple_overlap": "a" * 250, # c1(0-100), ns=90; c2(90-190), ns=180; c3(180-250)
            "just_under_two_chunks_no_overlap_needed_if_full": "a" * 105 # c1(0-100), ns=90, c2(90-105)
        }

        results = {name: splitter.split_text(texts_to_test[name]) for name in texts_to_test}

        assert len(results["short"]) == 1 and results["short"][0] == texts_to_test["short"]
        assert len(results["exact"]) == 1 and results["exact"][0] == texts_to_test["exact"]
        
        assert len(results["one_overlap"]) == 2
        assert results["one_overlap"][0] == "a" * 100
        assert results["one_overlap"][1] == "a" * (150 - 90) # 60 chars
        assert results["one_overlap"][1].startswith("a" * 10) # Overlap

        assert len(results["multiple_overlap"]) == 3
        assert results["multiple_overlap"][0] == "a" * 100
        assert results["multiple_overlap"][1] == "a" * 100
        assert results["multiple_overlap"][2] == "a" * (250 - 180) # 70 chars
        assert results["multiple_overlap"][1].startswith("a" * 10)
        assert results["multiple_overlap"][2].startswith("a" * 10)

        assert len(results["just_under_two_chunks_no_overlap_needed_if_full"]) == 2
        assert results["just_under_two_chunks_no_overlap_needed_if_full"][0] == "a" * 100
        assert results["just_under_two_chunks_no_overlap_needed_if_full"][1] == "a" * (105 - 90) # 15 chars
        assert results["just_under_two_chunks_no_overlap_needed_if_full"][1].startswith("a" * 10)


    def test_reconstruction_complex(self):
        text = "The quick brown fox jumps over the lazy dog. " * 5
        splitter = TextSplitter(chunk_size=30, overlap=8)
        chunks = splitter.split_text(text)
        
        reconstructed_text = ""
        if chunks:
            reconstructed_text = chunks[0]
            for i in range(1, len(chunks)):
                # This logic assumes the overlap was fully contained in the previous chunk's end.
                # And that the current chunk starts with that overlap.
                # A more robust check would be to verify against original text slices.
                reconstructed_text += chunks[i][splitter.overlap:] 
        
        assert reconstructed_text == text

    def test_split_text_unicode(self):
        """Test with unicode characters."""
        splitter = TextSplitter(chunk_size=5, overlap=1)
        text = "😊😊😊😊😊😊😊😊😊😊" # 10 unicode characters, each might be >1 byte
        # Python strings handle unicode characters as single characters for len() and slicing.
        chunks = splitter.split_text(text)
        # C1: 😊😊😊😊😊 (0-5)
        # next_start = 5-1 = 4
        # C2: 😊😊😊😊😊 (4-9) -> actually (4-10) because text is 10 long. 😊😊😊😊😊😊
        # Oh, min(start + self.chunk_size, text_length)
        # C1: text[0:5] => 😊😊😊😊😊
        # next_start = 5-1 = 4
        # C2: text[4:min(4+5,10)] => text[4:9] => 😊😊😊😊😊 (5th to 9th char)
        # next_start = 9-1 = 8
        # C3: text[8:min(8+5,10)] => text[8:10] => 😊😊 (9th to 10th char)
        
        assert len(chunks) == 3
        assert chunks[0] == "😊😊😊😊😊"
        assert chunks[1] == "😊😊😊😊😊" # Chars 5,6,7,8,9 (using 1-based index for clarity)
        assert chunks[2] == "😊😊"    # Chars 9,10
        
        # Verify overlap content
        assert chunks[1].startswith(text[4:5]) # 5th char
        assert chunks[2].startswith(text[8:9]) # 9th char

        reconstructed_text = ""
        if chunks:
            reconstructed_text = chunks[0]
            for i in range(1, len(chunks)):
                reconstructed_text += chunks[i][splitter.overlap:]
        assert reconstructed_text == text
