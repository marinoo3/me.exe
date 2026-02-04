from markdown import Markdown
from io import StringIO



def unmark_element(element, stream=None):
        if stream is None:
            stream = StringIO()
        if element.text:
            stream.write(element.text)
        for sub in element:
            unmark_element(sub, stream)
        if element.tail:
            stream.write(element.tail)
        return stream.getvalue()

Markdown.output_formats["plain"] = unmark_element #type: ignore



class ParseMD:
    """Parse text from PDF file"""

    def __new__(cls, file: str) -> str:
        """
        Args:
            file (str): File path to parse

        Returns:
            str: Extracted text string
        """
        md = Markdown(output_format="plain")
        md.stripTopLevelTags = False
        return cls.__parse_from_path(md, file)
    
    @staticmethod
    def __parse_from_path(md: Markdown, file_path: str):
        with open(file_path) as doc:
            return md.convert(doc.read())