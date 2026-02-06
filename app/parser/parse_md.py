from markdown import Markdown
from io import StringIO
import emoji



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
    """Parse text from Markdown"""

    @staticmethod
    def __create_md_parser() -> Markdown:
        """
        Create a MD parser

        Returns:
            Markdown: MD parser
        """
        md = Markdown(output_format="plain") #type: ignore
        md.stripTopLevelTags = False
        return md
    
    @staticmethod
    def __remove_emojis(text: str) -> str:
        return emoji.replace_emoji(text, replace="")
    
    @staticmethod
    def from_string(text: str, remove_emojis=False) -> str:
        """
        Parse text from text string
        
        Args:
            text (str): MD formatted string to parse

        Returns:
            str: Extracted text string
        """
        md = ParseMD.__create_md_parser()
        content = md.convert(text)

        if remove_emojis:
            content = ParseMD.__remove_emojis(content)

        return content
    
    @staticmethod
    def from_path(file_path: str, remove_emojis=False) -> str:
        """
        Parse text from .md file

        Args:
            file (str): File path to parse

        Returns:
            str: Extracted text string
        """
        md = ParseMD.__create_md_parser()
        with open(file_path) as doc:
            content = md.convert(doc.read())

            if remove_emojis:
                content = ParseMD.__remove_emojis(content)

            return content