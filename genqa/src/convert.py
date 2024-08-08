import argparse
import io
import logging
import os
from typing import  Optional, Union

import mammoth
import markdownify
import pymupdf
import pymupdf4llm
from cleantext import clean

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DocToMarkdown:
    """A class to convert various document formats to Markdown.

    This class provides methods to convert DOCX, HTML, and PDF files or content
    to Markdown format using various libraries including PyMuPDF.

    Attributes:
        supported_formats (dict): A dictionary mapping file extensions to
            their respective conversion methods.
    """

    def __init__(self):
        """Initializes the DocToMarkdown with supported file formats."""
        self.supported_formats = {
            ".docx": self._convert_docx,
            ".html": self._convert_html,
            ".pdf": self._convert_pdf,
        }

    def convert(
        self, input_data: Union[str, bytes], file_type: Optional[str] = None
    ) -> str:
        """Converts the input to Markdown.

        Args:
            input_data: Path to the file or the content itself.
            file_type: File extension (e.g., '.docx', '.html', '.pdf') or None.

        Returns:
            A string containing the Markdown representation of the input.

        Raises:
            ValueError: If the file type is unsupported or not specified when needed.
        """
        if isinstance(input_data, str) and os.path.isfile(input_data):
            file_type = os.path.splitext(input_data)[1].lower()
            return self.supported_formats[file_type](input_data)
        else:
            if file_type is None:
                raise ValueError(
                    "File type must be specified when input is not a file path"
                )
            if file_type not in self.supported_formats:
                raise ValueError(f"Unsupported file type: {file_type}")
            return self.supported_formats[file_type](input_data)

    def _convert_docx(self, input_data: Union[str, bytes]) -> str:
        """Converts DOCX content to Markdown.

        Args:
            input_data: The DOCX file path or content as bytes.

        Returns:
            A string containing the Markdown representation of the DOCX content.
        """
        if isinstance(input_data, str):
            with open(input_data, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
        else:
            result = mammoth.convert_to_html(io.BytesIO(input_data))

        html = result.value
        return self._html_to_markdown(html)

    def _convert_html(self, input_data: Union[str, bytes]) -> str:
        """Converts HTML content to Markdown.

        Args:
            input_data: The HTML file path, content as string, or content as bytes.

        Returns:
            A string containing the Markdown representation of the HTML content.
        """
        if isinstance(input_data, str):
            if os.path.isfile(input_data):
                with open(input_data, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = input_data
        else:
            content = input_data.decode("utf-8")
        return self._html_to_markdown(content)

    def _convert_pdf(self, input_data: Union[str, bytes]) -> str:
        """Converts PDF content to Markdown.

        Args:
            input_data: The PDF file path or content as bytes.

        Returns:
            A string containing the Markdown representation of the PDF content.
        """
        if isinstance(input_data, str):
            doc = pymupdf.open(input_data)
        else:
            doc = pymupdf.open(stream=input_data, filetype="pdf")

        return pymupdf4llm.to_markdown(doc)

    def _html_to_markdown(self, html_content: str) -> str:
        """Converts HTML to Markdown.

        Args:
            html_content: The HTML content as a string.

        Returns:
            A string containing the Markdown representation of the HTML content.
        """
        converter = markdownify.MarkdownConverter(strip=["img"])
        markdown_text = converter.convert(html_content)
        cleaned_text = clean(markdown_text, lower=False)
        return cleaned_text


def main():
    parser = argparse.ArgumentParser(description="Convert documents to Markdown.")
    parser.add_argument("input_file", help="Path to the input file.")
    args = parser.parse_args()

    converter = DocToMarkdown()
    input_file = args.input_file
    file_type = os.path.splitext(input_file)[1].lower()

    try:
        markdown_text = converter.convert(input_file, file_type)
        print(markdown_text)
    except ValueError as e:
        logging.error(e)


if __name__ == "__main__":
    main()
