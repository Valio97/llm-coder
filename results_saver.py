import os
from datetime import datetime
from pathlib import Path


class ResultsSaver:

    def __init__(self, file, save_path, llm_response, concept):
        self.file = file
        self.save_path = save_path
        self.llm_response = llm_response
        self.concept = concept

    def save_results_fully_automated(self):
        # Create the filename based on the concept name and LLM results
        # Replace spaces with underscores for the filename
        filename = f"{self.concept.replace(' ', '_')}_llm_results.txt"

        # Prepare the entry to be saved
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current date and time
        entry = f"{date_str} - {get_file_name(self.file)} - {self.llm_response}\n"

        full_file_path = os.path.join(self.save_path, filename)

        # Open the file in append mode
        with open(full_file_path, 'a') as file:
            # If the file does not exist, write the header (optional)
            # Write the entry to the file
            file.write(entry)

        print(f"Results saved to {full_file_path}")

    @staticmethod
    def save_results_relevant_chunks(messages, results_with_scores, save_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{messages.concept.replace(' ', '_')}_relevant_context_{timestamp}.txt"

        full_file_path = os.path.join(save_path, filename)

        # Write the results to a new file
        with open(full_file_path, 'w') as file:
            file.write(f"Relevant Context for Concept: {messages.concept}\n")
            file.write("=" * 50 + "\n")

            if results_with_scores:
                for result, score in results_with_scores:
                    content = result.page_content
                    file.write(f"Document:\n{content}\nScore: {score}\n")
                    file.write("-" * 50 + "\n")
            else:
                file.write("No relevant documents found above the similarity threshold.\n")


def get_file_name(file_path):
    # Create a Path object and return the name of the file
    return Path(file_path).name
