from enum import Enum


class ConfigurationOption(Enum):
    SYSTEM_MESSAGE = 'system_message'
    USER_MESSAGE = 'user_message'
    OUTPUT_FORMAT = 'output_format'
    THRESHOLD = 'threshold'
    MAX_CHUNKS = 'max_chunks'
    CHUNK_SIZE = 'chunk_size'
    RESULT_FORMAT = 'result_format'
