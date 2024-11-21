class ContextRetrievalConfig:

    def __init__(self, chunk_size, max_chunks, threshold, result_format):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        self.threshold = threshold
        self.result_format = result_format