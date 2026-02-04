from werkzeug.datastructures import FileStorage
import pymupdf


class ParsePDF:
    """Parse text from PDF file"""

    def __new__(cls, file: str|FileStorage) -> str:
        """
        Args:
            file (str|FileStorage): File path or file storage to parse

        Returns:
            str: Extracted text string
        """

        if isinstance(file, str):
            return cls.__parse_from_path(file)
        elif isinstance(file, FileStorage):
            return cls.__parse_from_file(file) 
    
    @staticmethod
    def __parse_from_path(file_path: str):
        with pymupdf.open(file_path) as doc:
            return '\n\n'.join(page.get_text() for page in doc) #type: ignore
        
    @staticmethod
    def __parse_from_file(file_storage: FileStorage):
        file_storage.stream.seek(0)
        pdf_bytes = file_storage.stream.read()

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            return '\n\n'.join(page.get_text() for page in doc) #type: ignore