import configparser
import os
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from file_utils import FileUtils
from results_saver import ResultsSaver
from messages import Messages

# The API key is read from the environment variable OPENAI_API_KEY
api_key = os.getenv('OPENAI_API_KEY')
background_color = "#dff5e0"


# Function to read from the properties file and construct the prompt
def get_concept_description(concept):
    config_file = 'prompt_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    section = 'Concepts'
    if config.has_section(section) and config.has_option(section, concept):
        return config.get(section, concept)
    else:
        return None


def construct_prompt(messages):
    # Combine the parts into the final prompt
    prompt = (f"{messages.system_message}\n{messages.user_message}\n "
              f"The concept '{messages.concept}' is described as follows: {get_concept_description(messages.concept)}\n"
              f"Please provide the output in the following format: \n{messages.output_format}")

    if messages.explanations:
        prompt = (f"{prompt} Please provide a brief explanation of the results next to the result"
                  f" separated by a semicolon")

    return prompt


def refactor_documents(documents):
    # Create a list to store the page content
    page_contents = []

    # Loop through each document and extract the 'page_content'
    for doc in documents:
        if hasattr(doc, 'page_content'):  # Check if 'page_content' exists
            page_contents.append(doc.page_content)  # Extract the 'page_content'

    # Combine all page contents into a single string (you can add formatting if necessary)
    return '\n'.join(page_contents)


def semi_automated_coding(messages, vector_store, save_path, threshold):
    threshold_value = float(threshold)
    results_with_scores = vector_store.similarity_search_with_relevance_scores(
        get_concept_description(messages.concept),
        k=100,
        threshold=threshold_value)
    filtered_results = [(doc, score) for doc, score in results_with_scores if score >= threshold_value]

    ResultsSaver.save_results_relevant_chunks(messages, filtered_results, save_path)


def main(file, save_path, tool_mode, messages, threshold):
    if not api_key:
        print("OPENAI_API_KEY environment variable is not set.")
        return

    if not os.path.isfile(file):
        print("Not a valid path")
        return

    if not FileUtils.is_pdf_file(file):
        print("The file is not a PDF!")
        return

    # Get the text from the pdf
    raw_text = FileUtils.extract_text_from_file(file)

    # Create text chunks
    text_chunks = FileUtils.split_text(raw_text)

    # Get the OpenAI embeddings. This object also uses the OPENAI_API_KEY
    # A request is sent to OpenAI to get the embeddings.
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)

    if tool_mode == '1':
        fully_automated_coding(messages, vector_store, file, save_path)
    else:
        semi_automated_coding(messages, vector_store, save_path, threshold)


def fully_automated_coding(messages, vector_store, file, save_path):
    # Prepare the LLM and the Q&A chain
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o")
    chain = load_qa_chain(llm, chain_type="stuff")

    # Call the function to construct the prompt
    prompt = construct_prompt(messages)

    # Retrieve the "k" most similar chunks
    documents = vector_store.similarity_search(prompt, k=40)

    prompt = f"{prompt}\n'Here are the relevant chunks from the text:'{refactor_documents(documents)}"
    print(prompt)

    # Send the request and get a response
    response = chain.invoke({"question": prompt, "input_documents": documents})

    result_saver = ResultsSaver(file, save_path, response.get("output_text"), messages.concept)
    result_saver.save_results_fully_automated()


def read_concepts():
    config_file = 'prompt_config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)

    # Get all concepts from the 'Concepts' section
    return {key: config.get('Concepts', key) for key in config.options('Concepts')}  # Return the list of concept names


def save_concepts(concepts_dict):
    """Save the given concepts dictionary to the prompt_config.ini file."""
    config = configparser.ConfigParser()

    # Load the existing configuration file or create a new one if it doesn't exist
    config.read("prompt_config.ini")

    # Clear existing Concepts section or create it if it doesn't exist
    if "Concepts" in config:
        config.remove_section("Concepts")
    config.add_section("Concepts")

    # Add each concept to the Concepts section from the provided dictionary
    for concept, description in concepts_dict.items():
        config.set("Concepts", concept, description)

    # Write the updated configuration back to the file
    with open("prompt_config.ini", "w") as configfile:
        config.write(configfile)

    # Confirmation message
    print("Concepts saved to prompt_config.ini successfully.")


# Tkinter GUI setup
def get_system_message():
    # Load the .ini file
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    # Get the system_message value
    try:
        system_message = config.get('Prompt', 'system_message')
        return system_message
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        # Handle missing section or option
        return "Error: Unable to retrieve system message."


def get_user_message():
    # Load the .ini file
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    # Get the system_message value
    try:
        system_message = config.get('Prompt', 'user_message')
        return system_message
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        # Handle missing section or option
        return "Error: Unable to retrieve system message."


def get_output_format():
    # Load the .ini file
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    # Get the system_message value
    try:
        system_message = config.get('Prompt', 'output_format')
        return system_message
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        # Handle missing section or option
        return "Error: Unable to retrieve system message."


def save_new_messages(system_message, user_message, output_format):
    # Load the .ini file
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    if not config.has_section('Prompt'):  # Replace 'SectionName' with your actual section name
        config.add_section('Prompt')  # Create the section if it doesn't exist

        # Replace the old message with the new one
    config.set('Prompt', 'system_message', system_message)
    config.set('Prompt', 'user_message', user_message)
    config.set('Prompt', 'output_format', output_format)

    # Write changes to the .ini file
    with open('prompt_config.ini', 'w') as configfile:  # Replace with your actual .ini file name
        config.write(configfile)

    print("System message updated successfully.")


def get_threshold_value():
    # Load the .ini file
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    # Get the system_message value
    try:
        threshold = config.get('Other', 'threshold')
        return float(threshold)
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        # Handle missing section or option
        return "Error: Unable to retrieve system message."


def save_threshold(threshold):
    config = configparser.ConfigParser()

    config.read("prompt_config.ini")

    if not config.has_section('Other'):  # Replace 'SectionName' with your actual section name
        config.add_section('Other')  # Create the section if it doesn't exist

        # Replace the old message with the new one
    config.set('Other', 'threshold', threshold)

    # Write changes to the .ini file
    with open('prompt_config.ini', 'w') as configfile:  # Replace with your actual .ini file name
        config.write(configfile)

    print("Threshold updated successfully.")


class App:
    def __init__(self, root_1):
        self.add_concept_window = None
        self.concept_description_entry = None
        self.concept_name_entry = None
        self.concepts = None
        self.concept_listbox = None
        self.concept_window = None
        self.root = root_1
        self.root.title("LLM-Coder")

        # Set the size of the window
        self.root.geometry("650x530")  # Width x Height

        # Make the window non-resizable
        self.root.resizable(False, False)
        self.root.configure(bg=background_color)

        # Mode selection
        self.mode = tk.StringVar(value="1")
        tk.Label(root, text="Select Mode:", bg=background_color).grid(row=0, column=0, padx=5, pady=(20, 5))
        (tk.Radiobutton(root, text="Fully-automated deductive coding", variable=self.mode, value="1",
                        bg=background_color, command=self.toggle_fields)
         .grid(row=0, column=1, pady=(20, 5), sticky="w"))
        (tk.Radiobutton(root, text="Relevant context retrieval                                                      "
                                   "      Threshold:", variable=self.mode, value="2", bg=background_color,
                        command=self.toggle_fields)
         .grid(row=1, column=1, sticky="w"))

        threshold = tk.DoubleVar(value=get_threshold_value())  # Variable to hold float value
        self.threshold_entry = tk.Entry(root, textvariable=threshold, width=10)
        self.threshold_entry.grid(row=1, column=2, padx=5, sticky="w")

        # Concept input dropdown
        tk.Label(root, text="Topic:", bg=background_color).grid(row=2, column=0, padx=5, pady=(20, 5))
        self.concept_combobox = ttk.Combobox(root, values=list(read_concepts().keys()), state="readonly",
                                             width=47)  # Create a readonly combobox
        self.concept_combobox.grid(row=2, column=1, columnspan=1, padx=5, pady=(20, 5))

        # Button to view concepts
        self.edit_button = tk.Button(root, text="Edit Concepts", command=self.view_concepts)
        self.edit_button.grid(row=2, column=2, padx=5, pady=(20, 5))

        # System message
        tk.Label(root, text="System Message:", bg=background_color).grid(row=3, column=0, padx=5, pady=(20, 5))
        self.system_message = tk.Text(root, height=5, width=50, wrap=tk.WORD)
        self.system_message.grid(row=3, column=1, columnspan=1, padx=5, pady=(20, 5))
        self.system_message.insert(tk.END, get_system_message())

        (tk.Label(root, text="Info:\nExplain the role of\nthe LLM", fg="red", bg=background_color)
         .grid(row=3, column=2, padx=5, pady=(20, 5)))

        # User message
        tk.Label(root, text="User Message:", bg=background_color).grid(row=4, column=0, padx=5, pady=5)
        self.user_message = tk.Text(root, height=5, width=50, wrap=tk.WORD)
        self.user_message.grid(row=4, column=1, columnspan=1, padx=5, pady=5)
        self.user_message.insert(tk.END, get_user_message())

        (tk.Label(root, text="Info:\nExplain the task", fg="red", bg=background_color)
         .grid(row=4, column=2, padx=5, pady=(20, 5)))

        # Output format
        tk.Label(root, text="Output Format:", bg=background_color).grid(row=5, column=0, padx=5, pady=5)
        self.output_format = tk.Text(root, height=3, width=50, wrap=tk.WORD)
        self.output_format.grid(row=5, column=1, columnspan=1, padx=5, pady=5)
        self.output_format.insert(tk.END, get_output_format())

        # Checkbox for Result Explanation
        self.result_explanation_var = tk.BooleanVar()  # Variable to hold the checkbox state
        self.result_explanation_checkbox = tk.Checkbutton(
            root,
            text="Provide explanations\n for the result",
            variable=self.result_explanation_var,
            bg=background_color
        )
        self.result_explanation_checkbox.grid(row=5, column=2, columnspan=1, padx=5, pady=5)

        # File path input
        tk.Label(root, text="File Path:", bg=background_color).grid(row=6, column=0, padx=5, pady=(20, 5))
        self.file_path_entry = tk.Entry(root, width=50)
        self.file_path_entry.grid(row=6, column=1, padx=5, pady=(20, 5))
        tk.Button(root, text="Browse", command=self.browse_file).grid(row=6, column=2, padx=5, pady=(20, 5))

        # Save destination input
        tk.Label(root, text="Save Destination:", bg=background_color).grid(row=7, column=0, padx=5, pady=5)
        self.save_path_entry = tk.Entry(root, width=50)
        self.save_path_entry.grid(row=7, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_save_path).grid(row=7, column=2, padx=5, pady=5)

        # Run button
        tk.Button(root, text="Run", command=self.run_main).grid(row=9, column=0, columnspan=4, pady=10)

        for i in range(4):  # Assuming there are 4 columns
            root.grid_columnconfigure(i, weight=1)
        self.toggle_fields()

    def toggle_fields(self):
        is_mode_2 = self.mode.get() == "2"

        # Mode-specific toggling
        self.threshold_entry.config(state="normal" if is_mode_2 else "disabled")
        self.concept_combobox.config(state="readonly" if is_mode_2 else "disabled")
        self.edit_button.config(state="normal" if is_mode_2 else "disabled")
        self.result_explanation_checkbox.config(state="disabled" if is_mode_2 else "normal")

        # Text fields to toggle
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
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)

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
        explanations = self.result_explanation_var.get()
        threshold = self.threshold_entry.get()

        self.reset_fields()

        # Validate inputs
        if not file_path or not concept_input:
            messagebox.showerror("Input Error", "Please provide both file path and concept.")
            return

        # Run the main function
        try:
            messages = Messages(concept_input, system_message, user_message, output_format, explanations)
            main(file_path, save_path, tool_mode, messages, threshold)
            save_new_messages(system_message, user_message, output_format)
            save_threshold(threshold)
            messagebox.showinfo("Success", "Process completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def view_concepts(self):
        """Open a new window to view and manage concepts."""
        self.concept_window = tk.Toplevel(self.root)
        self.concept_window.title("Edit Concepts")

        self.concept_window.configure(bg=background_color)

        # Make the main window inactive
        self.concept_window.grab_set()

        self.concept_listbox = tk.Listbox(self.concept_window, width=50, height=15)
        self.concept_listbox.pack(padx=10, pady=10)

        # Load concepts into the listbox
        self.concepts = read_concepts()
        for concept in self.concepts:
            self.concept_listbox.insert(tk.END, concept)

        # Buttons to add and delete concepts
        tk.Button(self.concept_window, text="View Definition", command=self.view_single_concept, width=12).pack(
            pady=(0, 5))
        tk.Button(self.concept_window, text="Add Concept", command=self.add_concept, width=12).pack(pady=(0, 5))
        tk.Button(self.concept_window, text="Delete Concept", command=self.delete_concept, width=12).pack(pady=(0, 5))
        tk.Button(self.concept_window, text="Close", command=self.concept_window.destroy).pack(pady=(0, 5))

    def view_single_concept(self):
        """Open a window to show the selected concept's name and description."""
        # Get the selected item from the Listbox
        try:
            selected_index = self.concept_listbox.curselection()[0]
            selected_concept = self.concept_listbox.get(selected_index)
            description = self.concepts[selected_concept]
        except IndexError:
            messagebox.showwarning("Selection Error", "Please select a concept to view.")
            return

        # Create a new window for viewing the concept details
        view_window = tk.Toplevel(self.concept_window)
        view_window.title("View Concept")

        # Display the concept name in bold
        concept_name_label = tk.Label(view_window, text="Concept Name:", font=("Helvetica", 10, "bold"))
        concept_name_label.pack(anchor="w", padx=10, pady=(10, 0))

        concept_name_text = tk.Label(view_window, text=selected_concept, font=("Helvetica", 10))
        concept_name_text.pack(anchor="w", padx=20, pady=(0, 10))

        # Display the description in bold and then the actual text
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

        # Concept name label and entry
        (tk.Label(self.add_concept_window, text="Enter concept name:", bg=background_color)
         .grid(row=0, column=0, padx=10, pady=10))
        self.concept_name_entry = tk.Entry(self.add_concept_window, width=50)
        self.concept_name_entry.grid(row=0, column=1, padx=10, pady=10)

        # Concept description label and larger text entry
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
        """Submit the concept if both fields are filled out."""
        concept_name = self.concept_name_entry.get().strip()
        concept_description = self.concept_description_entry.get("1.0", tk.END).strip()

        if not concept_name or not concept_description:
            messagebox.showerror("Error", "Both fields must be filled out.")
        else:
            self.save_new_concept(concept_name, concept_description)
            self.add_concept_window.destroy()  # Close the dialog after submission

    def save_new_concept(self, new_concept, description):
        if new_concept and new_concept not in read_concepts():
            self.concept_listbox.insert(tk.END, new_concept)
            self.concepts = read_concepts()
            self.concepts.update({new_concept: description})
            save_concepts(self.concepts)
            self.concept_combobox['values'] = list(self.concepts.keys())
            self.concept_combobox.set('')  # Clear current selection
        elif new_concept in read_concepts():
            messagebox.showwarning("Warning", "Concept already exists!")

    def delete_concept(self):
        """Delete the selected concept from the Listbox and update self.concepts."""
        selected_indices = self.concept_listbox.curselection()

        if selected_indices:
            for index in selected_indices[::-1]:  # Delete from the end to avoid index shift
                selected_concept = self.concept_listbox.get(index)  # Get the selected concept
                del self.concepts[selected_concept]  # Remove it from the dictionary
                self.concept_listbox.delete(index)  # Remove it from the Listbox

            # Save the updated concepts dictionary
            save_concepts(self.concepts)
            self.concept_combobox['values'] = list(self.concepts.keys())
            self.concept_combobox.set('')  # Clear current selection
            # Update the config file
            messagebox.showinfo("Success", "Selected concept(s) deleted successfully.")
        else:
            messagebox.showwarning("Selection Error", "Please select a concept to delete.")


# Run the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
