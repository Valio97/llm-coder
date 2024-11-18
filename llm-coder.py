import os
import tkinter as tk
import re
import tiktoken
from datetime import datetime
from tkinter import filedialog, messagebox
from tkinter import ttk

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from configuration_option import ConfigurationOption
from configuration_section import ConfigurationSection
from file_utils import FileUtils
from messages import Messages
from results_saver import ResultsSaver
from configuration import Configuration

# The API key is read from the environment variable OPENAI_API_KEY
api_key = os.getenv('OPENAI_API_KEY')
background_color = "#dff5e0"


def main(file_path, save_path, tool_mode, messages, threshold, max_chunks, result_format, chunk_size):
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.")
        return

    files_array = file_path.split(", ")

    if tool_mode == '1':
        fully_automated_coding(messages, files_array, save_path, threshold, max_chunks, chunk_size)
    else:
        relevant_context_retrieval(messages, files_array, save_path, threshold, max_chunks, result_format, chunk_size)


def fully_automated_coding(messages, files_array, save_path, threshold, max_chunks, chunk_size):
    # Prepare the LLM and the Q&A chain
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for single_file in files_array:
        prompt = construct_prompt(messages)
        raw_text = ''
        if single_file.endswith('.txt'):
            with open(single_file, "r", encoding='utf-8', errors='replace') as file:
                raw_text = file.read()
        elif single_file.endswith('.pdf'):
            raw_text = FileUtils.extract_text_from_pdf_file(single_file)

        tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Tokenize the text
        tokens = tokenizer.encode(raw_text)

        # Count the number of tokens
        token_count = len(tokens)

        if token_count > 20000:
            embeddings = OpenAIEmbeddings()
            text_chunks = FileUtils.split_text(raw_text, chunk_size)
            vector_store = FAISS.from_texts(text_chunks, embeddings)
            threshold_value = float(threshold)
            results_with_scores = vector_store.similarity_search_with_relevance_scores(
                prompt,
                k=max_chunks,
                threshold=threshold_value)
            filtered_results = [doc for doc, score in results_with_scores if score >= threshold_value]

            combined_chunks = refactor_documents(filtered_results)
            prompt = f"{prompt}\n\nHere is the text for your analysis:\n\n{combined_chunks}"
            print(prompt)
            response = llm.invoke(prompt)

        else:
            prompt = f"{prompt}\n\nHere is the text for your analysis:\n\n{raw_text}"
            print(prompt)

            response = llm.invoke(prompt)

        result_saver = ResultsSaver(single_file, save_path, response.content, messages.concept)
        result_saver.save_results_fully_automated(timestamp)


def relevant_context_retrieval(messages, files_array, save_path, threshold, max_chunks, result_format, chunk_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embeddings = OpenAIEmbeddings()
    threshold_value = float(threshold)

    for single_file in files_array:
        raw_text = ''
        if single_file.endswith('.txt'):
            with open(single_file, "r", encoding='utf-8', errors='replace') as file:
                raw_text = file.read()
        elif single_file.endswith('.pdf'):
            raw_text = FileUtils.extract_text_from_pdf_file(single_file)
        text_chunks = FileUtils.split_text(raw_text, chunk_size)
        print(text_chunks.__sizeof__())
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        results_with_scores = vector_store.similarity_search_with_relevance_scores(
            Configuration.get_concept_description(messages.concept),
            k=max_chunks,
            threshold=threshold_value)
        filtered_results = [(doc, score) for doc, score in results_with_scores if score >= threshold_value]

        if result_format == "txt":
            ResultsSaver.save_results_relevant_chunks_in_txt_file(messages, single_file, filtered_results, save_path,
                                                                  timestamp)
        elif result_format == "csv":
            ResultsSaver.save_results_relevant_chunks_in_csv_file(messages, single_file, filtered_results, save_path,
                                                                  timestamp)


def construct_prompt(messages):
    prompt = (f"{messages.system_message}\n{messages.user_message}\n "
              f"Please provide the output in the following format: \n{messages.output_format}")
    return prompt


def refactor_documents(documents):
    page_contents = []

    for doc in documents:
        if hasattr(doc, 'page_content'):
            page_contents.append(doc.page_content)

    return '\n\n'.join(page_contents)


class App:
    def __init__(self, root_1):
        self.config_window = None
        self.format_var = (
            tk.StringVar(value=Configuration.get_context_retrieval_option(ConfigurationOption.RESULT_FORMAT)))
        self.chunk_size_var = (
            tk.IntVar(value=Configuration.get_context_retrieval_option(ConfigurationOption.CHUNK_SIZE)))
        self.threshold_var = (
            tk.DoubleVar(value=Configuration.get_context_retrieval_option(ConfigurationOption.THRESHOLD)))
        self.max_chunks_var = (
            tk.IntVar(value=Configuration.get_context_retrieval_option(ConfigurationOption.MAX_CHUNKS)))
        self.add_concept_window = None
        self.concept_description_entry = None
        self.concept_name_entry = None
        self.concepts = None
        self.concept_listbox = None
        self.concept_window = None
        self.root = root_1
        self.root.title("LLM-Coder")

        # Menu
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # Configuration menu
        config_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Settings", menu=config_menu)
        config_menu.add_command(label="Configuration", command=self.open_configuration)

        # Set the size of the window
        self.root.geometry("650x530")  # Width x Height

        # Make the window non-resizable
        self.root.resizable(False, False)
        self.root.configure(bg=background_color)

        # Mode selection
        self.mode = tk.StringVar(value="1")
        tk.Label(root, text="Select Mode:", bg=background_color).grid(row=0, column=0, padx=5, pady=(20, 5))
        (tk.Radiobutton(root, text="Automated deductive coding", variable=self.mode, value="1",
                        bg=background_color, command=self.toggle_fields)
         .grid(row=0, column=1, pady=(20, 5), sticky="w"))
        (tk.Radiobutton(root, text="Relevant context retrieval", variable=self.mode, value="2", bg=background_color,
                        command=self.toggle_fields)
         .grid(row=1, column=1, sticky="w"))

        # Concept input dropdown
        tk.Label(root, text="Concept:", bg=background_color).grid(row=2, column=0, padx=5, pady=(20, 5))
        self.concept_combobox = ttk.Combobox(root, values=list(Configuration.read_concepts().keys()), state="readonly",
                                             width=47)  # Create a readonly combobox
        self.concept_combobox.grid(row=2, column=1, columnspan=1, padx=5, pady=(20, 5))

        # Button to view concepts
        self.edit_button = tk.Button(root, text="Edit Concepts", command=self.view_concepts)
        self.edit_button.grid(row=2, column=2, padx=5, pady=(20, 5))

        # System message
        tk.Label(root, text="System Message:", bg=background_color).grid(row=3, column=0, padx=5, pady=(20, 5))
        self.system_message = tk.Text(root, height=5, width=50, wrap=tk.WORD)
        self.system_message.grid(row=3, column=1, columnspan=1, padx=5, pady=(20, 5))
        self.system_message.insert(tk.END, Configuration.get_prompt_part(ConfigurationOption.SYSTEM_MESSAGE))

        (tk.Label(root, text="Info:\nDescribe the role of\nthe LLM", fg="red", bg=background_color)
         .grid(row=3, column=2, padx=5, pady=(20, 5)))

        # User message
        tk.Label(root, text="User Message:", bg=background_color).grid(row=4, column=0, padx=5, pady=5)
        self.user_message = tk.Text(root, height=5, width=50, wrap=tk.WORD)
        self.user_message.grid(row=4, column=1, columnspan=1, padx=5, pady=5)
        self.user_message.insert(tk.END, Configuration.get_prompt_part(ConfigurationOption.USER_MESSAGE))

        (tk.Label(root, text="Info:\nExplain the task and\ndescribe the codebook", fg="red", bg=background_color)
         .grid(row=4, column=2, padx=5, pady=(20, 5)))

        # Output format
        tk.Label(root, text="Output Format:", bg=background_color).grid(row=5, column=0, padx=5, pady=5)
        self.output_format = tk.Text(root, height=3, width=50, wrap=tk.WORD)
        self.output_format.grid(row=5, column=1, columnspan=1, padx=5, pady=5)
        self.output_format.insert(tk.END, Configuration.get_prompt_part(ConfigurationOption.OUTPUT_FORMAT))

        # File path input
        tk.Label(root, text="Files for Analysis:", bg=background_color).grid(row=6, column=0, padx=5, pady=(20, 5))
        self.file_path_entry = tk.Entry(root, width=50)
        self.file_path_entry.grid(row=6, column=1, padx=5, pady=(20, 5))
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=6, column=2, padx=5, pady=(20, 5))

        # Save destination input
        tk.Label(root, text="Save Results To:", bg=background_color).grid(row=7, column=0, padx=5, pady=5)
        self.save_path_entry = tk.Entry(root, width=50)
        self.save_path_entry.grid(row=7, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_save_path).grid(row=7, column=2, padx=5, pady=5)

        # Run button
        tk.Button(root, text="Run", command=self.run_main).grid(row=9, column=0, columnspan=4, pady=10)

        for i in range(4):
            root.grid_columnconfigure(i, weight=1)
        self.toggle_fields()

    def open_configuration(self):
        self.config_window = tk.Toplevel(self.root)
        self.config_window.title("Configuration")
        self.config_window.geometry("570x230")

        # Maximum number of chunks entry
        (tk.Label(self.config_window, text="Maximum number of text chunks that can be retrieved:")
         .grid(row=0, column=0, padx=10, pady=10, sticky="w"))
        max_chunks_entry = tk.Entry(self.config_window, textvariable=self.max_chunks_var, width=10)
        max_chunks_entry.grid(row=0, column=1, padx=10, pady=10)

        (tk.Label(self.config_window, text="Maximum number of characters for a single chunk:")
         .grid(row=1, column=0, padx=10, pady=10, sticky="w"))
        chunk_size_entry = tk.Entry(self.config_window, textvariable=self.chunk_size_var, width=10)
        chunk_size_entry.grid(row=1, column=1, padx=10, pady=10)

        # Threshold entry
        (tk.Label(self.config_window, text="Minimal threshold for the similarity search algorithm [0,1]:")
         .grid(row=2, column=0, padx=10, pady=10, sticky="w"))
        threshold_entry = tk.Entry(self.config_window, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=2, column=1, padx=10, pady=10)

        # Result format dropdown
        (tk.Label(self.config_window, text="Result format (applicable only to the relevant context retrieval feature):")
         .grid(row=3, column=0, padx=10, pady=10, sticky="w"))
        format_dropdown = ttk.Combobox(self.config_window, textvariable=self.format_var, values=["txt", "csv"],
                                       state="readonly")
        format_dropdown.grid(row=3, column=1, padx=10, pady=10)

        # Save button
        save_button = tk.Button(self.config_window, text="Save", command=self.save_configuration)
        save_button.grid(row=6, column=0, columnspan=2, pady=20)

    def save_configuration(self):
        max_chunks = str(self.max_chunks_var.get())
        chunk_size = str(self.chunk_size_var.get())
        threshold = str(self.threshold_var.get())
        result_format = self.format_var.get()

        Configuration.update_configuration(ConfigurationSection.OTHER, ConfigurationOption.MAX_CHUNKS, max_chunks)
        Configuration.update_configuration(ConfigurationSection.OTHER, ConfigurationOption.CHUNK_SIZE, chunk_size)
        Configuration.update_configuration(ConfigurationSection.OTHER, ConfigurationOption.THRESHOLD, threshold)
        Configuration.update_configuration(ConfigurationSection.OTHER, ConfigurationOption.RESULT_FORMAT, result_format)

        self.config_window.destroy()

    def toggle_fields(self):
        is_mode_2 = self.mode.get() == "2"

        self.concept_combobox.config(state="readonly" if is_mode_2 else "disabled")
        self.edit_button.config(state="normal" if is_mode_2 else "disabled")

        text_fields = [self.system_message, self.user_message, self.output_format]
        for field in text_fields:
            field.config(state="disabled" if is_mode_2 else "normal")
        text_fields = [self.system_message, self.user_message, self.output_format]

        enabled_fg = "black"
        enabled_bg = "white"
        disabled_fg = "gray"
        disabled_bg = "#F0F0F0"
        for field in text_fields:
            if is_mode_2:
                field.config(state="disabled", fg=disabled_fg, bg=disabled_bg)
            else:
                field.config(state="normal", fg=enabled_fg, bg=enabled_bg)

    def browse_file(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Text and PDF files", "*.txt *.pdf")])

        if not file_paths:
            directory = filedialog.askdirectory()
            if directory:
                file_paths = [
                    os.path.join(directory, f) for f in os.listdir(directory)
                    if f.endswith('.txt') or f.endswith('.pdf')
                ]

        if file_paths:
            file_paths_str = ', '.join(file_paths)

            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_paths_str)

    def browse_save_path(self):
        save_path = filedialog.askdirectory()
        if save_path:
            self.save_path_entry.delete(0, tk.END)
            self.save_path_entry.insert(0, save_path)

    def reset_fields(self):
        """Clears the file path and concept fields while leaving mode selection intact."""

    def run_main(self):
        system_message = self.system_message.get('1.0', 'end').strip()
        user_message = self.user_message.get('1.0', 'end').strip()
        output_format = self.output_format.get('1.0', 'end').strip()

        file_path = self.file_path_entry.get()
        save_path = self.save_path_entry.get()
        concept_input = self.concept_combobox.get().strip()
        tool_mode = self.mode.get()
        threshold = self.threshold_var.get()
        max_chunks = self.max_chunks_var.get()
        result_format = self.format_var.get()
        chunk_size = self.chunk_size_var.get()

        self.reset_fields()

        try:
            messages = Messages(concept_input, system_message, user_message, output_format)
            main(file_path, save_path, tool_mode, messages, threshold, max_chunks, result_format, chunk_size)
            Configuration.update_whole_prompt(system_message, user_message, output_format)
            messagebox.showinfo("Success", "Process completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def view_concepts(self):
        self.concept_window = tk.Toplevel(self.root)
        self.concept_window.title("Edit Concepts")

        self.concept_window.configure(bg=background_color)

        self.concept_window.grab_set()

        self.concept_listbox = tk.Listbox(self.concept_window, width=50, height=15)
        self.concept_listbox.pack(padx=10, pady=10)

        self.concepts = Configuration.read_concepts()
        for concept in self.concepts:
            self.concept_listbox.insert(tk.END, concept)

        tk.Button(self.concept_window, text="View Definition", command=self.view_single_concept, width=12).pack(
            pady=(0, 5))
        tk.Button(self.concept_window, text="Add Concept", command=self.add_concept, width=12).pack(pady=(0, 5))
        tk.Button(self.concept_window, text="Delete Concept", command=self.delete_concept, width=12).pack(pady=(0, 5))
        tk.Button(self.concept_window, text="Close", command=self.concept_window.destroy).pack(pady=(0, 5))

    def view_single_concept(self):
        try:
            selected_index = self.concept_listbox.curselection()[0]
            selected_concept = self.concept_listbox.get(selected_index)
            description = self.concepts[selected_concept]
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select a concept to view.")
            return

        view_window = tk.Toplevel(self.concept_window)
        view_window.title("View Concept")

        concept_name_label = tk.Label(view_window, text="Concept Name:", font=("Helvetica", 10, "bold"))
        concept_name_label.pack(anchor="w", padx=10, pady=(10, 0))

        concept_name_text = tk.Label(view_window, text=selected_concept, font=("Helvetica", 10))
        concept_name_text.pack(anchor="w", padx=20, pady=(0, 10))

        description_label = tk.Label(view_window, text="Concept Definition:", font=("Helvetica", 10, "bold"))
        description_label.pack(anchor="w", padx=10, pady=(0, 5))

        description_text = tk.Label(view_window, text=description, font=("Helvetica", 10), wraplength=400,
                                    justify="left")
        description_text.pack(anchor="w", padx=20, pady=(0, 10))

    def add_concept(self):
        self.add_concept_window = tk.Toplevel(self.root)
        self.add_concept_window.title("Add Concept")

        self.add_concept_window.configure(bg=background_color)

        self.add_concept_window.grab_set()

        (tk.Label(self.add_concept_window, text="Enter concept name:", bg=background_color)
         .grid(row=0, column=0, padx=10, pady=10))
        self.concept_name_entry = tk.Entry(self.add_concept_window, width=50)
        self.concept_name_entry.grid(row=0, column=1, padx=10, pady=10)

        (tk.Label(self.add_concept_window, text="Enter concept definition:", bg=background_color)
         .grid(row=1, column=0, padx=10, pady=10))
        self.concept_description_entry = tk.Text(self.add_concept_window, height=10, width=40)  # Larger text field
        self.concept_description_entry.grid(row=1, column=1, padx=10, pady=10)

        # Add and Cancel buttons
        tk.Button(self.add_concept_window, text="Add", command=self.submit_concept).grid(row=2, column=0,
                                                                                         columnspan=2,
                                                                                         padx=10,
                                                                                         pady=10)

    def submit_concept(self):
        concept_name = self.concept_name_entry.get().strip()
        concept_description = self.concept_description_entry.get("1.0", tk.END).strip()

        if not concept_name or not concept_description:
            messagebox.showerror("Error", "Both fields must be filled out.")
        else:
            self.save_new_concept(concept_name, concept_description)
            self.add_concept_window.destroy()

    def save_new_concept(self, new_concept, description):
        if new_concept and new_concept not in Configuration.read_concepts():
            self.concept_listbox.insert(tk.END, new_concept)
            self.concepts = Configuration.read_concepts()
            self.concepts.update({new_concept: description})
            Configuration.save_concepts(self.concepts)
            self.concept_combobox['values'] = list(self.concepts.keys())
            self.concept_combobox.set('')  # Clear current selection
        elif new_concept in Configuration.read_concepts():
            messagebox.showwarning("Warning", "Concept already exists!")

    def delete_concept(self):
        selected_indices = self.concept_listbox.curselection()

        if selected_indices:
            for index in selected_indices[::-1]:
                selected_concept = self.concept_listbox.get(index)
                del self.concepts[selected_concept]
                self.concept_listbox.delete(index)

            Configuration.save_concepts(self.concepts)
            self.concept_combobox['values'] = list(self.concepts.keys())
            self.concept_combobox.set('')
            messagebox.showinfo("Success", "Selected concept(s) deleted successfully.")
        else:
            messagebox.showwarning("Selection Error", "Please select a concept to delete.")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
