"""Document loaders for various file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
import re
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from pydantic import BaseModel


class Document(BaseModel):
    """Represents a loaded document."""

    content: str
    metadata: dict[str, str]


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """
        Load a document from a file path.

        Args:
            file_path: Path to the file to load

        Returns:
            Loaded document with content and metadata
        """
        pass


class EPUBLoader(DocumentLoader):
    """Loader for epub documents."""

    def load(self, file_path: Path) -> Document:
        """
        Load a eoub document.

        Args:
            file_path: Path to the epub file

        Returns:
            Document with extracted text and metadata
        """
        book = epub.read_epub(str(file_path))

        # Extract text from all pages
        text_parts = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                # Use BeautifulSoup to strip HTML tags
                raw_html = item.get_content().decode('utf-8', errors='replace')
                soup = BeautifulSoup(raw_html, "html.parser")
                text = soup.get_text()
                if text.strip():
                    text_parts.append(text)

        content = "\n\n".join(text_parts)
        #I take out any multiple spaces
        content = re.sub(r'\s+', ' ', content).strip() 

        # Extract metadata
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "doc_id": file_path.stem,
            "file_type": "epub",  
        }

        # Add epub metadata if available
        title = book.get_metadata('DC', 'title')
        if title:
            metadata["title"] = title[0][0]
            
        creator = book.get_metadata('DC', 'creator')
        if creator:
            metadata["author"] = creator[0][0]

        return Document(content=content, metadata=metadata)


class TextLoader(DocumentLoader):
    """Loader for plain text documents."""

    def load(self, file_path: Path) -> Document:
        """
        Load a text document.

        Args:
            file_path: Path to the text file

        Returns:
            Document with content and metadata
        """
        content = file_path.read_text(encoding="utf-8")

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "text",
        }

        return Document(content=content, metadata=metadata)


def get_loader(file_path: Path) -> DocumentLoader:
    """
    Get the appropriate loader for a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Appropriate document loader

    Raises:
        ValueError: If file type is not supported
    """
    suffix = file_path.suffix.lower()

    if suffix == ".epub":
        return EPUBLoader()
    elif suffix in [".txt", ".md"]:
        return TextLoader()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
 

