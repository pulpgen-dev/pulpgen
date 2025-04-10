# exporter.py
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import unicodedata
import os
from html import unescape # Use unescape for markdown conversion

from rich.console import Console

console = Console() # Use a local console instance if needed for messages

# --- Helper Functions (Copied/Adapted from pulpgen.py) ---

def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    (Adapted from Django's slugify utility)
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    value = re.sub(r"[-\s]+", "-", value).strip("-_")
    # Limit length to avoid excessively long folder/file names
    return value[:50]


def _get_sorted_chapters(book_root):
    """Helper to get chapters sorted numerically by ID."""
    if book_root is None:
        return []
    chapters_raw = book_root.findall(".//chapter")
    # Handle chapters that might be missing 'id' or have non-integer IDs during sorting
    def get_sort_key(chap):
        chap_id_str = chap.get("id", "0")
        try:
            return int(chap_id_str)
        except ValueError:
            return float('inf') # Put chapters with invalid IDs at the end
    return sorted(chapters_raw, key=get_sort_key)

# --- Export Functions ---

def export_single_markdown(book_root, output_path: Path):
    """Exports the entire book content to a single Markdown file."""
    if book_root is None:
        console.print("[red]Error: Book data is missing, cannot export.[/red]")
        return False

    try:
        markdown_content = []

        # Title
        title = book_root.findtext("title", "Untitled Book")
        markdown_content.append(f"# {unescape(title)}\n")

        # Synopsis
        synopsis = book_root.findtext("synopsis")
        if synopsis:
            markdown_content.append(f"## Synopsis\n\n{unescape(synopsis.strip())}\n")

        # Optional: Add Characters (simple list for now)
        characters = book_root.findall(".//character")
        if characters:
            markdown_content.append("## Characters\n")
            for char in characters:
                 name = unescape(char.findtext("name", "N/A"))
                 desc = unescape(char.findtext("description", "N/A"))
                 markdown_content.append(f"*   **{name}**: {desc}")
            markdown_content.append("\n")


        # Chapters
        markdown_content.append("## Chapters\n")
        chapters = _get_sorted_chapters(book_root)
        if not chapters:
             markdown_content.append("*No chapters found.*\n")
        else:
            for chap in chapters:
                chap_id = chap.get("id", "N/A")
                chap_num_elem = chap.find("number")
                chap_num = chap_num_elem.text if chap_num_elem is not None else chap_id # Fallback to ID if number tag missing
                chap_title = chap.findtext("title", "Untitled Chapter")

                markdown_content.append(f"### Chapter {unescape(chap_num)}: {unescape(chap_title)}\n")

                content = chap.find("content")
                if content is not None:
                    paragraphs = content.findall(".//paragraph")
                    if paragraphs:
                        for para in paragraphs:
                            para_text = (para.text or "").strip()
                            if para_text: # Only add non-empty paragraphs
                                markdown_content.append(f"{unescape(para_text)}\n") # Double newline handled by join
                    else:
                        markdown_content.append("*[Chapter content not available]*\n")
                else:
                    markdown_content.append("*[Chapter content structure missing]*\n")

        # Join content with double newlines between major sections/paragraphs
        full_markdown = "\n".join(markdown_content)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)

        console.print(f"[green]Successfully exported single Markdown file to:[/green] [cyan]{output_path.resolve()}[/cyan]")
        return True

    except Exception as e:
        console.print(f"[bold red]Error exporting single Markdown file to {output_path}: {e}[/bold red]")
        # Optionally print traceback if needed for debugging
        # console.print_exception(show_locals=False)
        return False


def export_markdown_per_chapter(book_root, output_parent_dir: Path, book_title_slug: str):
    """Exports each chapter to its own Markdown file within a dedicated directory."""
    if book_root is None:
        console.print("[red]Error: Book data is missing, cannot export.[/red]")
        return False

    # Define the specific output directory for this export type
    export_dir_name = f"{book_title_slug}-chapters-markdown"
    export_path = output_parent_dir / export_dir_name

    try:
        export_path.mkdir(parents=True, exist_ok=True)
        console.print(f"Exporting chapters to directory: [cyan]{export_path.resolve()}[/cyan]")

        chapters = _get_sorted_chapters(book_root)

        if not chapters:
            console.print("[yellow]No chapters found in the book to export.[/yellow]")
            # Create an empty README or indicator file?
            (export_path / "_EMPTY_NO_CHAPTERS_FOUND.txt").touch()
            return True # Technically successful, just nothing to export

        files_exported_count = 0
        for chap in chapters:
            chap_id = chap.get("id", "N/A")
            chap_num_elem = chap.find("number")
            chap_num = chap_num_elem.text if chap_num_elem is not None else chap_id
            chap_title = chap.findtext("title", "Untitled Chapter")
            chap_title_slug = slugify(chap_title if chap_title != "Untitled Chapter" else f"chapter-{chap_num}")

            # Try to format chapter number nicely for filename
            try:
                chap_num_int = int(chap_num)
                filename = f"{chap_num_int:02d}-{chap_title_slug}.md"
            except ValueError:
                filename = f"{chap_num}-{chap_title_slug}.md" # Fallback if number isn't integer

            chapter_file_path = export_path / filename

            chapter_markdown = []
            chapter_markdown.append(f"# Chapter {unescape(chap_num)}: {unescape(chap_title)}\n")

            content = chap.find("content")
            if content is not None:
                paragraphs = content.findall(".//paragraph")
                if paragraphs:
                    for para in paragraphs:
                        para_text = (para.text or "").strip()
                        if para_text:
                            chapter_markdown.append(f"{unescape(para_text)}\n")
                else:
                    chapter_markdown.append("*[Chapter content not available]*\n")
            else:
                chapter_markdown.append("*[Chapter content structure missing]*\n")

            full_chapter_markdown = "\n".join(chapter_markdown)

            try:
                with open(chapter_file_path, "w", encoding="utf-8") as f:
                    f.write(full_chapter_markdown)
                files_exported_count += 1
            except Exception as file_e:
                console.print(f"[red]Error writing chapter file {filename}: {file_e}[/red]")
                # Continue trying to export other chapters

        console.print(f"[green]Successfully exported {files_exported_count}/{len(chapters)} chapters.[/green]")
        return True

    except OSError as dir_e:
         console.print(f"[bold red]Error creating export directory {export_path}: {dir_e}[/bold red]")
         return False
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during per-chapter export: {e}[/bold red]")
        # console.print_exception(show_locals=False)
        return False