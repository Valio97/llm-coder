from prompt_components import PromptComponents
from context_retrieval_config import ContextRetrievalConfig


class ToolConfig:

    def __init__(self,
                 tool_mode,
                 messages: PromptComponents,
                 concept_input,
                 files_array,
                 save_path,
                 context_retrieval_config: ContextRetrievalConfig):
        self.tool_mode = tool_mode
        self.messages = messages
        self.concept_input = concept_input
        self.files_array = files_array
        self.save_path = save_path
        self.context_retrieval_config = context_retrieval_config
