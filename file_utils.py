from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


class FileUtils:

    @staticmethod
    def is_pdf_file(file):
        return file.lower().endswith('.pdf')

    @staticmethod
    def split_text(text):
        # chunk_size is the maximum chunk size that will be split if splitting is possible.
        # If not possible the chunk will end at the next possible place.
        # If a chunk is bigger than the size defined here, a message will be shown in the console.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=False
        )
        return text_splitter.split_text(text)

    # Gets the text from each page and combines them together
    @staticmethod
    def extract_text_from_file(file):
        pdfreader = PdfReader(file)

        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        return raw_text
