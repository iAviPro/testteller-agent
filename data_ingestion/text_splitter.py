import logging

logger = logging.getLogger(__name__)

class TextSplitter:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")
        if overlap < 0:
            raise ValueError("Overlap must be a non-negative integer.")
        if overlap >= chunk_size :
            # This condition could be allowed, but might lead to unexpected behavior or infinite loops
            # if not handled carefully in split_text. For now, let's make it a warning or stricter check.
            logger.warning(
                f"Overlap ({overlap}) is greater than or equal to chunk_size ({chunk_size}). "
                "This may lead to redundant or empty chunks."
            )
            # Depending on desired strictness, could raise ValueError here too.
            # raise ValueError("Overlap cannot be greater than or equal to chunk size.")

        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.debug(f"TextSplitter initialized with chunk_size={chunk_size}, overlap={overlap}")

    def split_text(self, text: str) -> list[str]:
        if not text:
            logger.debug("Input text is empty, returning empty list of chunks.")
            return []
        
        chunks = []
        start = 0
        text_length = len(text)

        if text_length <= self.chunk_size:
            logger.debug(f"Text length ({text_length}) is less than or equal to chunk_size ({self.chunk_size}). Returning as single chunk.")
            return [text]

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            logger.debug(f"Created chunk: '{chunk[:50]}...' (length {len(chunk)}) from start={start} to end={end}")

            if end == text_length:
                logger.debug("Reached end of text.")
                break
            
            start_before_overlap_check = start
            start = end - self.overlap
            
            # Prevent infinite loops if overlap is too large or chunk_size too small relative to overlap
            if start <= start_before_overlap_check and self.overlap > 0 : 
                # This can happen if chunk_size is not much larger than overlap,
                # or if overlap pushes start backwards or keeps it the same.
                # Example: chunk_size=10, overlap=10. start=0, end=10. next_start=10-10=0. Loop.
                # Example: chunk_size=10, overlap=8. text="abcdefghijklm".
                #   1. "abcdefghij" (start=0, end=10) -> next_start = 10-8 = 2
                #   2. "cdefghijkl" (start=2, end=12) -> next_start = 12-8 = 4
                #   3. "efghijklm"  (start=4, end=13) -> next_start = 13-8 = 5. end=13. break.
                # What if start doesn't advance sufficiently?
                # If chunk_size = 5, overlap = 4. text = "abcdefghi"
                # 1. "abcde" (0:5), next_start = 5-4=1
                # 2. "bcdef" (1:6), next_start = 6-4=2
                # 3. "cdefg" (2:7), next_start = 7-4=3
                # 4. "defgh" (3:8), next_start = 8-4=4
                # 5. "efghi" (4:9), next_start = 9-4=5. end=9. break.
                # The logic seems okay if overlap < chunk_size.
                # The problem arises if start doesn't advance. start = end - overlap.
                # We need end - overlap > previous_start for progress.
                # previous_start + chunk_size - overlap > previous_start
                # chunk_size - overlap > 0  => chunk_size > overlap
                if self.overlap >= self.chunk_size: # Stricter check than just start <= start_before_overlap_check
                     logger.error(
                        f"Overlap ({self.overlap}) is >= chunk_size ({self.chunk_size}), "
                        "which would cause an infinite loop. Breaking split. "
                        f"Problematic chunk: '{chunk[:50]}...'. Current start: {start_before_overlap_check}"
                    )
                     # This indicates a setup error or a text that can't be split with these settings.
                     # Depending on desired behavior, could return current chunks or raise error.
                     # For now, returning what we have to avoid infinite loop.
                     break 
            
            logger.debug(f"Next chunk will start at: {start}")

        logger.info(f"Split text of length {text_length} into {len(chunks)} chunks.")
        return chunks

# Example Usage (optional, can be removed or kept for module-level testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Setup basic logging for direct script run
    splitter = TextSplitter(chunk_size=10, overlap=3)
    sample_text = "abcdefghijklmnopqr" # 18 chars
    chunks = splitter.split_text(sample_text)
    logger.info(f"Sample text: '{sample_text}'")
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: '{chunk}'")

    splitter_no_overlap = TextSplitter(chunk_size=10, overlap=0)
    chunks_no_overlap = splitter_no_overlap.split_text(sample_text)
    logger.info(f"Sample text (no overlap): '{sample_text}'")
    for i, chunk in enumerate(chunks_no_overlap):
        logger.info(f"Chunk {i+1}: '{chunk}'")

    splitter_short_text = TextSplitter(chunk_size=100, overlap=10)
    short_text = "This is short."
    chunks_short = splitter_short_text.split_text(short_text)
    logger.info(f"Short text: '{short_text}'")
    for i, chunk in enumerate(chunks_short):
        logger.info(f"Chunk {i+1}: '{chunk}'")
    
    splitter_problematic = TextSplitter(chunk_size=5, overlap=5)
    problem_text = "abcdefghijkl"
    chunks_problem = splitter_problematic.split_text(problem_text)
    logger.info(f"Problematic text: '{problem_text}'")
    for i, chunk in enumerate(chunks_problem):
        logger.info(f"Chunk {i+1}: '{chunk}'")
