# -*- coding: utf-8 -*-
import google.generativeai as genai
from google.generativeai import types

# Import specific exceptions for better handling
from google.api_core import exceptions as google_api_exceptions

import xml.etree.ElementTree as ET
from xml.dom import minidom  # For pretty printing XML
import os
import argparse

# import json # Logging removed
from datetime import datetime
import uuid
from pathlib import Path
import time
import copy
import re
import json
import getpass
import unicodedata  # For slugify
import math
import sys  # For word count display and exiting

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from html import escape  # For HTML generation

import markdown_exporter

# --- Configuration ---
WRITING_MODEL_NAME = "gemini-2.5-pro-exp-03-25"
WRITING_MODEL_CONFIG = types.GenerationConfig(
    temperature=1,
    max_output_tokens=65536,  # Standard max for 1.5 Pro
    top_p=0.95,
    top_k=40,
    # stop_sequences=["</patch>"] # Stop sequence might be useful here
)
# LOG_FILE = "novel_writer_log.jsonl" # Logging removed
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT_FOR_FOLDER = "%Y%m%d"
# --- API Retry Configuration ---
MAX_API_RETRIES = 3
API_RETRY_BACKOFF_FACTOR = 2  # Base seconds for backoff (e.g., 2s, 4s, 8s)

# --- Rich Console Setup ---
console = Console()

# --- Helper Functions ---

# Logging functions removed
# def setup_logging(): ...
# def log_interaction(log_data): ...


def _count_words(text):
    """Simple word counter."""
    if not text or not text.strip():
        return 0
    return len(text.split())


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
    # Limit length to avoid excessively long folder names
    return value[:50]


def pretty_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    try:
        rough_string = ET.tostring(elem, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    except Exception as e:
        console.print(f"[red]Error pretty-printing XML: {e}[/red]")
        # Fallback to basic tostring
        return ET.tostring(elem, encoding="unicode")


def clean_llm_xml_output(xml_string):
    """Attempts to clean potential markdown/text surrounding LLM XML output."""
    if not isinstance(xml_string, str):
        return ""  # Handle None or other types
    # Basic cleaning: find the first '<' and last '>'
    start = xml_string.find("<")
    end = xml_string.rfind(">")
    if start != -1 and end != -1:
        cleaned = xml_string[start : end + 1].strip()
        # Remove potential markdown code fences
        cleaned = re.sub(r"^```xml\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned
    return xml_string  # Return original if no tags found


def parse_xml_string(xml_string, expected_root_tag="patch", attempt_clean=True):
    """
    Safely parses an XML string, optionally cleaning it first.
    expected_root_tag: The root tag name expected ('patch', 'book', etc.).
                       If 'patch', may attempt to wrap fragments.
    """
    if not xml_string:
        console.print("[bold red]Error: Received empty XML string from LLM.[/bold red]")
        return None

    if attempt_clean:
        xml_string = clean_llm_xml_output(xml_string)

    if not xml_string.strip():
        console.print("[bold red]Error: XML string is empty after cleaning.[/bold red]")
        return None

    try:
        # Attempt to wrap with <patch> ONLY if expecting a patch and it's missing
        if expected_root_tag == "patch" and not xml_string.strip().startswith(
            "<patch>"
        ):
            # Check if it looks like chapter content fragments
            if "<chapter" in xml_string and "</chapter>" in xml_string:
                console.print(
                    "[yellow]Warning: LLM output seems to be missing root <patch> tag for chapters, attempting to wrap.[/yellow]"
                )
                xml_string = f"<patch>{xml_string}</patch>"
            # Add more heuristics if needed for other patch types

        return ET.fromstring(xml_string)
    except ET.ParseError as e:
        console.print(
            f"[bold red]Error parsing XML (expecting <{expected_root_tag}>):[/bold red] {e}"
        )
        # Extract line/column if available
        error_details = str(e)
        match = re.search(r"line (\d+), column (\d+)", error_details)
        line_num, col_num = -1, -1
        if match:
            line_num, col_num = int(match.group(1)), int(match.group(2))

        console.print("[yellow]Attempted to parse:[/yellow]")
        if line_num > 0:
            lines = xml_string.splitlines()
            context_start = max(0, line_num - 3)
            context_end = min(len(lines), line_num + 2)
            for i in range(context_start, context_end):
                prefix = ">> " if (i + 1) == line_num else "   "
                console.print(f"[dim]{prefix}{lines[i]}[/dim]")
            if col_num > 0:
                console.print(f"[dim]   {' ' * (col_num - 1)}^ Error near here[/dim]")
        else:
            console.print(
                f"[dim]{xml_string[:1000]}...[/dim]"
            )  # Fallback if line number unknown

        return None


def find_chapter(book_root, chapter_id):
    """Finds a chapter element by its id attribute."""
    if book_root is None:
        return None
    return book_root.find(f".//chapter[@id='{chapter_id}']")


def find_paragraph(chapter_element, paragraph_id):
    """Finds a paragraph element by its id attribute within a chapter."""
    if chapter_element is None:
        return None
    content_element = chapter_element.find("content")
    if content_element is None:
        return None
    return content_element.find(f".//paragraph[@id='{paragraph_id}']")


def get_next_patch_number(book_dir):
    """Finds the next available patch number based on existing files."""
    if not book_dir or not book_dir.is_dir():
        return 1
    try:
        patch_files = sorted(
            [
                p
                for p in book_dir.glob("patch-*.xml")
                if p.stem.split("-")[-1].isdigit()
            ],
            key=lambda p: int(p.stem.split("-")[-1]),
        )
        if not patch_files:
            return 1
        last_patch = patch_files[-1].stem
        last_num = int(last_patch.split("-")[-1])
        return last_num + 1
    except (IndexError, ValueError, FileNotFoundError):
        # Fallback: count files if parsing fails
        return len(list(book_dir.glob("patch-*.xml"))) + 1


# --- NovelWriter Class ---


class NovelWriter:
    # Modified __init__ to handle new folder naming scheme and file input for idea
    def __init__(
        self, resume_folder_name=None, initial_prompt_file=None
    ):  # Added initial_prompt_file parameter
        self.api_key = self._get_api_key()
        if not self.api_key:
            console.print(
                "[bold red]API Key is required to initialize the client. Exiting.[/bold red]"
            )
            raise ValueError("Missing API Key")
        self.client = None
        self.book_root = None
        self.book_dir = None
        self.book_id = None  # This will be the UUID part
        self.book_title_slug = "untitled"  # Default
        self.patch_counter = 0
        self.chapters_generated_in_session = set()
        self.total_word_count = 0  # Initialize word count

        self._init_client()  # Init client early

        if resume_folder_name:
            # --- Resuming Book ---
            console.print(
                Panel(f"Resuming Project: {resume_folder_name}", style="bold blue")
            )
            self.book_dir = Path(resume_folder_name)
            if not self.book_dir.is_dir():
                console.print(
                    f"[bold red]Error: Book directory '{resume_folder_name}' not found.[/bold red]"
                )
                raise FileNotFoundError(
                    f"Book directory '{resume_folder_name}' not found."
                )

            # Try parsing folder name: YYYYMMDD-slug-uuid
            parts = resume_folder_name.split("-", 2)
            if len(parts) == 3:
                self.book_title_slug = parts[1]
                self.book_id = parts[2]  # UUID is the book_id
            else:
                # Fallback if pattern doesn't match (e.g., old format)
                console.print(
                    f"[yellow]Warning: Could not parse folder name '{resume_folder_name}' into standard format. Using fallback ID.[/yellow]"
                )
                self.book_id = (
                    resume_folder_name  # Use full name as ID if parsing fails
                )

            console.print(f"Loading state from: [cyan]{self.book_dir.name}[/cyan]")
            self._load_book_state()  # Load state using self.book_dir

        else:
            # --- New Book Creation ---
            console.print(Panel("Starting a New Novel Project", style="bold green"))

            # --- Get Initial Idea/Prompt ---
            idea = ""
            # Check if a prompt file was provided via command line
            if initial_prompt_file:
                prompt_path = Path(initial_prompt_file)
                if prompt_path.is_file():
                    try:
                        idea = prompt_path.read_text(encoding="utf-8")
                        console.print(
                            f"[green]✓ Read book idea from file: {prompt_path}[/green]"
                        )
                        console.print(f"[dim]Content preview: {idea[:200]}...[/dim]")
                    except Exception as e:
                        console.print(
                            f"[bold red]Error reading prompt file '{prompt_path}': {e}. Please provide prompt interactively.[/bold red]"
                        )
                        # Fallback to interactive prompt if file read fails
                        initial_prompt_file = None  # Clear the flag so we ask below
                else:
                    console.print(
                        f"[bold red]Error: Prompt file '{prompt_path}' not found. Please provide prompt interactively.[/bold red]"
                    )
                    # Fallback to interactive prompt if file not found
                    initial_prompt_file = None  # Clear the flag so we ask below

            # Ask interactively if no file was provided OR if the file failed to load
            if not initial_prompt_file:
                idea = Prompt.ask("[yellow]Enter your book idea/description[/yellow]")

            # Handle empty idea (whether from file or prompt)
            if not idea or not idea.strip():
                console.print(
                    "[red]No valid idea provided. Using a generic placeholder.[/red]"
                )
                idea = "A default book idea for generating a title."

            # --- Generate initial title & Setup ---
            temp_uuid = str(uuid.uuid4())[:8]
            initial_outline_data = self._generate_minimal_outline(idea)

            book_title = initial_outline_data.get(
                "title", f"Untitled Novel {temp_uuid}"
            )
            self.book_title_slug = slugify(book_title) if book_title else "untitled"

            today_date_str = datetime.now().strftime(DATE_FORMAT_FOR_FOLDER)
            self.book_id = temp_uuid  # Keep UUID as the core unique ID
            folder_name = f"{today_date_str}-{self.book_title_slug}-{self.book_id}"
            self.book_dir = Path(folder_name)

            try:
                self.book_dir.mkdir(parents=True, exist_ok=True)
                console.print(
                    f"Created project directory: [cyan]{self.book_dir.resolve()}[/cyan]"
                )
            except OSError as e:
                console.print(
                    f"[bold red]Error creating directory {self.book_dir}: {e}[/bold red]"
                )
                raise

            self.book_root = ET.Element("book")
            ET.SubElement(self.book_root, "title").text = book_title
            ET.SubElement(self.book_root, "synopsis").text = initial_outline_data.get(
                "synopsis", "Synopsis not yet generated."
            )
            ET.SubElement(
                self.book_root, "initial_idea"
            ).text = idea  # Store the idea used
            ET.SubElement(self.book_root, "characters")
            ET.SubElement(self.book_root, "chapters")

            self._save_book_state("outline.xml")
            self.patch_counter = 0

            console.print(
                f"Started new book: [cyan]{book_title}[/cyan] (ID: {self.book_id})"
            )
            # Note: The full outline generation (characters, chapter summaries) happens in Step 1 of run()

        # setup_logging() # Logging removed

    # New helper to get just title/synopsis without full structure generation yet
    def _generate_minimal_outline(self, idea):
        """Attempts to generate just the title and synopsis using the LLM."""
        console.print("[cyan]Generating initial title and synopsis...[/cyan]")
        prompt = f"""
Based on the following book idea/description, please generate ONLY a compelling title and a brief (1-2 sentence) synopsis.

Idea/Description:
---
{idea}
---

Output format (Strictly JSON):
{{
  "title": "Your Generated Title",
  "synopsis": "Your generated brief synopsis."
}}

Do not include any other text, explanations, or markdown. Just the JSON object.
"""
        # Use a simpler/faster model maybe? Or just the main one.
        # For now, stick to the main model but with simpler request.
        response_json_str = self._get_llm_response(
            prompt, "Generating Title/Synopsis", allow_stream=False
        )  # Don't need streaming for JSON

        if response_json_str:
            try:
                # Clean potential markdown fences around JSON
                cleaned_json_str = re.sub(
                    r"^```json\s*",
                    "",
                    response_json_str.strip(),
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                cleaned_json_str = re.sub(r"\s*```$", "", cleaned_json_str)
                data = json.loads(cleaned_json_str)
                if isinstance(data, dict) and "title" in data and "synopsis" in data:
                    console.print("[green]✓ Title and Synopsis generated.[/green]")
                    return {
                        "title": data.get("title"),
                        "synopsis": data.get("synopsis"),
                    }
                else:
                    console.print(
                        "[yellow]Warning: LLM response was not valid JSON with title/synopsis keys.[/yellow]"
                    )
            except json.JSONDecodeError as e:
                console.print(
                    f"[yellow]Warning: Failed to decode LLM response as JSON: {e}[/yellow]"
                )
                console.print(f"[dim]Received: {response_json_str[:200]}[/dim]")
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Error processing title/synopsis response: {e}[/yellow]"
                )

        console.print(
            "[yellow]Could not generate initial title/synopsis automatically. Using placeholders.[/yellow]"
        )
        return {
            "title": f"Untitled Novel ({str(uuid.uuid4())[:4]})",
            "synopsis": "Synopsis pending.",
        }

    def _get_api_key(self):
        """Gets Gemini API key from environment variables or prompts the user."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print(
                "[yellow]GEMINI_API_KEY not found in environment variables or .env file.[/yellow]"
            )
            try:
                api_key = getpass.getpass("Enter your Gemini API Key: ")
            except (
                EOFError
            ):  # Handle environments where getpass isn't available (e.g., some CI/CD)
                console.print("[red]Could not read API key from input.[/red]")
                return None
        return api_key

    def _init_client(self):
        """Initializes the Gemini client."""
        if not self.api_key:
            # Should have been caught in __init__, but double-check
            console.print(
                "[bold red]Cannot initialize Gemini client without API Key.[/bold red]"
            )
            return False  # Indicate failure

        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(
                model_name=WRITING_MODEL_NAME,
                generation_config=WRITING_MODEL_CONFIG,
            )
            # Simple test call (optional, might consume quota)
            # self.client.count_tokens("test")
            console.print(
                f"[green]Successfully initialized Gemini client with model '{WRITING_MODEL_NAME}'[/green]"
            )
            return True
        except Exception as e:
            console.print(
                "[bold red]Fatal Error: Could not initialize Gemini client.[/bold red]"
            )
            console.print(f"Error details: {e}")
            # console.print_exception(show_locals=False) # Show traceback if needed
            self.client = None  # Ensure client is None on failure
            return False

    def _load_book_state(self):
        """Loads the latest book state from XML files in the book directory."""
        if not self.book_dir or not self.book_dir.is_dir():
            console.print(
                "[bold red]Error: Book directory not set or invalid for loading.[/bold red]"
            )
            return  # Cannot load

        outline_file = self.book_dir / "outline.xml"
        final_file = self.book_dir / "final.xml"  # Check for final first

        # Determine the base file to load (final > outline)
        load_file = None
        if final_file.exists():
            load_file = final_file
            console.print(f"Found final state file: [cyan]{load_file.name}[/cyan]")
        elif outline_file.exists():
            load_file = outline_file
            console.print(f"Found outline file: [cyan]{load_file.name}[/cyan]")
        else:
            # If neither exists, check for patches based on outline
            load_file = outline_file  # Assume outline *should* exist, even if we load patches over it

        # Ensure patches are sorted numerically correctly
        patch_files = sorted(
            [
                p
                for p in self.book_dir.glob("patch-*.xml")
                if p.stem.split("-")[-1].isdigit()
            ],
            key=lambda p: int(p.stem.split("-")[-1]),
        )

        latest_file = load_file  # Start with the base file
        if patch_files:
            latest_patch_file = patch_files[-1]
            # Decide if patch is newer than final.xml (simplistic: always load latest patch if exists)
            # A more robust check would compare timestamps, but loading latest patch is usually desired for resuming work
            if final_file.exists():
                console.print(
                    f"[yellow]Warning: Found both final.xml and patches. Loading latest patch ({latest_patch_file.name}) as current state.[/yellow]"
                )
            latest_file = latest_patch_file
            try:
                self.patch_counter = int(latest_file.stem.split("-")[-1])
            except (ValueError, IndexError):
                console.print(
                    f"[yellow]Warning: Could not parse patch number from {latest_file.name}. Estimating counter.[/yellow]"
                )
                self.patch_counter = len(patch_files)  # Estimate based on count
        elif load_file == outline_file:  # No patches, loaded outline
            self.patch_counter = 0
        # else: loaded final.xml, patch counter determined by latest patch or 0 if none

        if not latest_file or not latest_file.exists():
            console.print(
                f"[yellow]Warning: No state file found ({latest_file.name if latest_file else 'outline.xml/final.xml'}). Book may be empty or corrupted.[/yellow]"
            )
            # Initialize empty structure to prevent errors later
            self.book_root = ET.Element("book")
            ET.SubElement(
                self.book_root, "title"
            ).text = f"{self.book_title_slug} (Loaded Empty)"
            ET.SubElement(self.book_root, "synopsis").text = "Synopsis not found."
            ET.SubElement(
                self.book_root, "initial_idea"
            ).text = "Initial idea not found."
            ET.SubElement(self.book_root, "characters")
            ET.SubElement(self.book_root, "chapters")
            self.patch_counter = 0
            return

        try:
            tree = ET.parse(latest_file)
            self.book_root = tree.getroot()
            # Update title slug from loaded data if needed
            loaded_title = self.book_root.findtext("title")
            if loaded_title:
                self.book_title_slug = slugify(loaded_title)

            console.print(f"Loaded state from: [cyan]{latest_file.name}[/cyan]")
            # Ensure patch counter reflects the loaded file if it was a patch
            if latest_file.name.startswith("patch-"):
                try:
                    self.patch_counter = int(latest_file.stem.split("-")[-1])
                except:
                    pass  # Keep previous estimation if parsing fails here too
            # If final was loaded, find the actual highest patch number to continue from there
            elif latest_file == final_file and patch_files:
                try:
                    self.patch_counter = int(patch_files[-1].stem.split("-")[-1])
                except:
                    self.patch_counter = len(patch_files)  # Fallback if parse fails
            elif latest_file == outline_file and not patch_files:
                self.patch_counter = 0  # Ensure counter is 0 if only outline exists

        except ET.ParseError as e:
            console.print(
                f"[bold red]Error parsing XML file {latest_file}: {e}[/bold red]"
            )
            console.print("Cannot continue. Please check the XML file integrity.")
            self.book_root = None  # Indicate failure
            raise  # Re-raise to stop execution potentially
        except Exception as e:
            console.print(
                f"[bold red]Error loading book state from {latest_file}: {e}[/bold red]"
            )
            self.book_root = None  # Indicate failure
            raise

    def _save_book_state(self, filename):
        """Saves the current book state (ElementTree root) to an XML file."""
        if self.book_root is None:
            console.print(
                "[bold red]Error: Cannot save state, book data is missing.[/bold red]"
            )
            return False  # Indicate failure
        if not self.book_dir:
            console.print(
                "[bold red]Error: Cannot save state, book directory not set.[/bold red]"
            )
            return False  # Indicate failure

        filepath = self.book_dir / filename
        try:
            # Ensure directory exists right before saving
            self.book_dir.mkdir(parents=True, exist_ok=True)

            xml_str = pretty_xml(self.book_root)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(xml_str)
            console.print(
                f"[green]Book state saved to:[/green] [cyan]{filepath.name}[/cyan]"
            )
            return True  # Indicate success
        except Exception as e:
            console.print(
                f"[bold red]Error saving book state to {filepath}: {e}[/bold red]"
            )
            return False  # Indicate failure

    # --- Feature: Enhanced Error Handling & Retry ---
    def _get_llm_response(
        self, prompt_content, task_description="Generating content", allow_stream=True
    ):
        """
        Sends prompt to LLM, handles streaming/non-streaming, logs interaction (removed),
        and handles API errors with retries and backoff.
        """
        if self.client is None:
            console.print(
                "[bold red]Error: Gemini client not initialized. Cannot make API call.[/bold red]"
            )
            return None

        retries = 0
        while retries < MAX_API_RETRIES:
            full_response = ""
            try:
                console.print(
                    Panel(
                        f"[yellow]Sending request to Gemini ({task_description})... (Attempt {retries + 1}/{MAX_API_RETRIES})[/yellow]",
                        border_style="dim",
                    )
                )

                if allow_stream:
                    # --- Streaming ---
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,  # Keeps spinner clean
                    ) as progress:
                        task = progress.add_task(
                            description="[cyan]Gemini is thinking...", total=None
                        )
                        response_stream = self.client.generate_content(
                            contents=prompt_content, stream=True
                        )

                        console.print(
                            f"[cyan]>>> Gemini Response ({task_description}):[/cyan]"
                        )
                        first_chunk = True
                        response_completed_normally = (
                            False  # Track if stream finished ok
                        )
                        for chunk in response_stream:
                            if first_chunk:
                                progress.update(
                                    task, description="[cyan]Receiving response..."
                                )
                                first_chunk = False

                            # Check for blocking reasons in each chunk's feedback
                            if (
                                hasattr(
                                    chunk, "prompt_feedback"
                                )  # Check if attribute exists
                                and chunk.prompt_feedback
                                and chunk.prompt_feedback.block_reason
                            ):
                                reason = chunk.prompt_feedback.block_reason
                                ratings = (
                                    chunk.prompt_feedback.safety_ratings
                                    if chunk.prompt_feedback
                                    else []
                                )
                                ratings_str = "\n".join(
                                    [
                                        f"  - {r.category.name}: {r.probability.name}"
                                        for r in ratings
                                    ]
                                )
                                msg = f"Content generation blocked during streaming.\nReason: {reason.name if reason else 'Unknown'}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                                console.print(
                                    f"\n[bold red]API Safety Error: {msg}[/bold red]"
                                )
                                # This is likely unrecoverable by simple retry, treat as fatal for this attempt
                                raise types.BlockedPromptException(
                                    msg
                                )  # Re-raise for outer catch

                            # Append text safely
                            try:
                                # Check if chunk has 'text' attribute before accessing
                                if hasattr(chunk, "text"):
                                    chunk_text = chunk.text
                                    print(chunk_text, end="", flush=True)
                                    full_response += chunk_text
                                elif hasattr(
                                    chunk, "parts"
                                ):  # Handle potential 'parts' structure
                                    for part in chunk.parts:
                                        if hasattr(part, "text"):
                                            chunk_text = part.text
                                            print(chunk_text, end="", flush=True)
                                            full_response += chunk_text
                                # else: Unknown chunk structure, skip

                            except ValueError as ve:
                                # This might happen if the stream is abruptly terminated or blocked without explicit reason yet
                                console.print(
                                    f"\n[yellow]Warning: Received potentially invalid chunk data: {ve}[/yellow]"
                                )
                                # Check overall response feedback *after* the loop if needed, but blocking check above is better
                                continue  # Try to get next chunk

                        print()  # Newline after streaming finishes
                        response_completed_normally = True  # Mark as finished ok

                        # Check final response feedback after stream completion (important!)
                        # Accessing internal _response might be fragile, prefer official ways if available
                        final_feedback = None
                        if hasattr(
                            response_stream, "prompt_feedback"
                        ):  # Check the stream object itself first
                            final_feedback = response_stream.prompt_feedback
                        elif hasattr(response_stream, "_response") and hasattr(
                            response_stream._response, "prompt_feedback"
                        ):  # Fallback
                            final_feedback = response_stream._response.prompt_feedback

                        if final_feedback and final_feedback.block_reason:
                            reason = final_feedback.block_reason
                            ratings = final_feedback.safety_ratings
                            ratings_str = "\n".join(
                                [
                                    f"  - {r.category.name}: {r.probability.name}"
                                    for r in ratings
                                ]
                            )
                            msg = f"Content generation blocked (final check).\nReason: {reason.name if reason else 'Unknown'}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                            console.print(
                                f"\n[bold red]API Safety Error: {msg}[/bold red]"
                            )
                            raise types.BlockedPromptException(msg)

                else:
                    # --- Non-Streaming ---
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        transient=True,
                    ) as progress:
                        progress.add_task(
                            description="[cyan]Gemini is thinking...", total=None
                        )
                        response = self.client.generate_content(
                            contents=prompt_content
                        )  # Non-streaming call

                    # Check for blocking (more straightforward in non-streaming)
                    if (
                        hasattr(response, "prompt_feedback")  # Check attribute exists
                        and response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        reason = response.prompt_feedback.block_reason
                        ratings = (
                            response.prompt_feedback.safety_ratings
                            if response.prompt_feedback
                            else []
                        )
                        ratings_str = "\n".join(
                            [
                                f"  - {r.category.name}: {r.probability.name}"
                                for r in ratings
                            ]
                        )
                        msg = f"Content generation blocked.\nReason: {reason.name if reason else 'Unknown'}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                        console.print(f"[bold red]API Safety Error: {msg}[/bold red]")
                        raise types.BlockedPromptException(msg)  # Raise for outer catch

                    # Extract text safely
                    try:
                        # Check if response has 'text' attribute
                        if hasattr(response, "text"):
                            full_response = response.text
                        elif hasattr(
                            response, "parts"
                        ):  # Handle 'parts' structure if text is missing
                            full_response = "".join(
                                part.text
                                for part in response.parts
                                if hasattr(part, "text")
                            )
                        else:
                            # If neither text nor parts, check candidates for finish reason
                            finish_reason_msg = "Unknown reason"
                            if hasattr(response, "candidates") and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, "finish_reason"):
                                    finish_reason_msg = candidate.finish_reason.name
                                # Check safety ratings on the candidate if available
                                if (
                                    hasattr(candidate, "safety_ratings")
                                    and candidate.safety_ratings
                                ):
                                    ratings = candidate.safety_ratings
                                    ratings_str = "\n".join(
                                        [
                                            f"  - {r.category.name}: {r.probability.name}"
                                            for r in ratings
                                        ]
                                    )
                                    finish_reason_msg += (
                                        f"\nSafety Ratings:\n{ratings_str or 'N/A'}"
                                    )

                            if finish_reason_msg == "SAFETY":
                                raise types.BlockedPromptException(
                                    "Content generation likely blocked due to safety (Finish Reason: SAFETY)."
                                )
                            else:
                                raise ValueError(
                                    f"Response object does not contain expected 'text' or 'parts' attribute. Finish Reason: {finish_reason_msg}"
                                )

                        console.print(
                            f"[cyan]>>> Gemini Response ({task_description}):[/cyan]"
                        )
                        console.print(
                            f"[dim]{full_response[:1000]}{'...' if len(full_response) > 1000 else ''}[/dim]"
                        )
                    except ValueError as ve:
                        # Handle case where response might be blocked but didn't have block_reason set explicitly?
                        # Or other extraction issues
                        console.print(
                            f"[bold red]Error retrieving text from non-streaming response: {ve}[/bold red]"
                        )
                        console.print(
                            f"[dim]Response object (summary): {response}[/dim]"
                        )

                        # Re-raise or handle as appropriate for retry logic
                        raise ValueError(
                            f"Could not extract text from response: {ve}"
                        ) from ve

                # --- Success Case ---
                console.print(
                    Panel(
                        "[green]✓ Gemini response received successfully.[/green]",
                        border_style="dim",
                    )
                )
                return full_response  # Successful, exit the retry loop

            # --- Exception Handling & Retry Logic ---
            except (
                types.BlockedPromptException,
                # types.StopCandidateException, # This might indicate normal stop, not necessarily an error
            ) as safety_error:
                # Safety errors are usually not recoverable by retrying the same prompt
                console.print(f"[bold red]Safety Error: {safety_error}[/bold red]")
                console.print(
                    "[yellow]This type of error usually requires changing the prompt or safety settings. Retrying might not help.[/yellow]"
                )
                # Ask if user wants to retry anyway? Or just abort? For now, abort this attempt.
                if not Confirm.ask(
                    f"[yellow]Safety error encountered. Try sending the request again anyway? (Attempt {retries + 1}/{MAX_API_RETRIES})[/yellow]",
                    default=False,
                ):
                    console.print(
                        "[red]Aborting API call due to safety error and user choice.[/red]"
                    )
                    return None
                # If they choose to retry, increment count and continue loop (backoff happens below)

            except google_api_exceptions.ResourceExhausted as rate_limit_error:
                error_msg = f"API Rate Limit Error: {rate_limit_error}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry makes sense here

            except google_api_exceptions.DeadlineExceeded as timeout_error:
                error_msg = f"API Timeout Error: {timeout_error}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help

            except google_api_exceptions.GoogleAPICallError as api_error:
                error_msg = f"API Call Error: {type(api_error).__name__} - {api_error}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help for transient issues (e.g., 503 Service Unavailable)

            except ET.ParseError as xml_error:
                # This happens *after* getting a response, but indicates a bad response likely due to API issues
                error_msg = f"XML Parsing Error after receiving response: {xml_error}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print(
                    "[yellow]This often means the API response was incomplete or malformed (possibly due to interruption, blocking, or rate limits).[/yellow]"
                )
                console.print(
                    f"[dim]Partial response received before error:\n{full_response[:500]}...[/dim]"
                )
                # Retry might get a complete response

            except ValueError as val_error:  # Catch extraction errors raised above
                error_msg = f"Data Extraction Error: {val_error}"
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help if it was transient

            except Exception as e:
                error_msg = (
                    f"Unexpected Error during API call: {type(e).__name__} - {e}"
                )
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print_exception(show_locals=False, word_wrap=True)
                # Retry might help for some transient errors

            # --- Retry Action ---
            retries += 1
            if retries < MAX_API_RETRIES:
                wait_time = API_RETRY_BACKOFF_FACTOR * (
                    2 ** (retries - 1)
                )  # Exponential backoff
                console.print(
                    f"[yellow]Waiting {wait_time:.1f} seconds before retrying...[/yellow]"
                )
                time.sleep(wait_time)
                # Ask user if they want to proceed with the retry
                if not Confirm.ask(
                    f"[yellow]Error encountered. Proceed with retry attempt {retries + 1}/{MAX_API_RETRIES}? [/yellow]",
                    default=True,
                ):
                    console.print(
                        "[red]Aborting API call after error due to user choice.[/red]"
                    )
                    return None  # User cancelled retry
                # Continue loop
            else:
                console.print(
                    f"[bold red]Maximum retries ({MAX_API_RETRIES}) reached. Failed to get a valid response.[/bold red]"
                )
                return None  # Max retries exceeded

        # Should not be reached if logic is correct, but as a fallback:
        console.print("[bold red]Exited LLM response function unexpectedly.[/bold red]")
        return None

    def _apply_patch(self, patch_xml_str):
        """Applies a patch XML string to the current book_root."""
        # Expecting a 'patch' root tag here
        patch_root = parse_xml_string(patch_xml_str, expected_root_tag="patch")
        if patch_root is None or patch_root.tag != "patch":
            console.print(
                "[bold red]Error: Invalid patch XML received or failed to parse. Cannot apply patch.[/bold red]"
            )
            # Optionally log invalid patch string (Logging removed)
            return False

        applied_changes = False
        # Make a deep copy ONLY if we intend to revert on failure, otherwise modify in place
        # original_book_root = copy.deepcopy(self.book_root) # Removed for potentially better memory usage if reverts aren't strictly needed

        try:
            # --- Apply Chapter Patches ---
            chapters_patched = []
            for chapter_patch in patch_root.findall("chapter"):
                chapter_id = chapter_patch.get("id")
                if not chapter_id:
                    console.print(
                        "[yellow]Warning: Chapter in patch missing 'id'. Skipping.[/yellow]"
                    )
                    continue

                target_chapter = find_chapter(self.book_root, chapter_id)
                if target_chapter is None:
                    console.print(
                        f"[yellow]Warning: Chapter ID '{chapter_id}' in patch not found in book. Skipping.[/yellow]"
                    )
                    continue

                new_content = chapter_patch.find("content")
                content_patch = chapter_patch.find("content-patch")

                if new_content is not None:
                    old_content = target_chapter.find("content")
                    if old_content is not None:
                        target_chapter.remove(old_content)
                    else:  # Ensure content tag exists even if replacing
                        # We'll add the new_content element itself later
                        pass

                    # Ensure new paragraphs have IDs and Text
                    paras_in_new_content = new_content.findall(".//paragraph")
                    valid_paras_added = 0
                    for i, para in enumerate(paras_in_new_content):
                        # Generate ID if missing or empty (use 1-based index)
                        para_id = para.get("id")
                        if not para_id or not para_id.strip():
                            para_id = str(i + 1)
                            para.set("id", para_id)
                            console.print(
                                f"[yellow]Warning: Paragraph {i + 1} in chapter {chapter_id} patch missing/empty ID. Assigned '{para_id}'.[/yellow]"
                            )

                        # Also ensure paragraph has some text or it's useless
                        if not para.text or not para.text.strip():
                            para.text = "[Empty Paragraph]"  # Add placeholder if empty
                            console.print(
                                f"[yellow]Warning: Paragraph {para.get('id')} in chapter {chapter_id} patch was empty. Added placeholder text.[/yellow]"
                            )
                        valid_paras_added += 1

                    if valid_paras_added > 0:
                        target_chapter.append(
                            copy.deepcopy(new_content)
                        )  # Deepcopy sub-elements being added
                        console.print(
                            f"[green]Applied full content patch to Chapter {chapter_id} ({valid_paras_added} paragraphs).[/green]"
                        )
                        chapters_patched.append(chapter_id)
                        applied_changes = True
                    else:
                        console.print(
                            f"[yellow]Warning: Full content patch for Chapter {chapter_id} contained no valid paragraphs. Not applied.[/yellow]"
                        )
                        # Add an empty content tag back if it was removed
                        if old_content is None:
                            ET.SubElement(target_chapter, "content")

                elif content_patch is not None:
                    target_content = target_chapter.find("content")
                    if target_content is None:
                        console.print(
                            f"[yellow]Warning: Cannot apply paragraph patch to Chapter {chapter_id} (no <content>). Creating <content> tag.[/yellow]"
                        )
                        # Create content tag if missing, maybe shouldn't happen but safer
                        target_content = ET.SubElement(target_chapter, "content")

                    paras_patched_count = 0
                    for para_patch in content_patch.findall("paragraph"):
                        para_id = para_patch.get("id")
                        if para_id is None or not para_id.strip():
                            console.print(
                                f"[yellow]Warning: Paragraph patch for Chapter {chapter_id} missing or empty 'id'. Skipping.[/yellow]"
                            )
                            continue
                        # Ensure text exists
                        if para_patch.text is None or not para_patch.text.strip():
                            console.print(
                                f"[yellow]Warning: Paragraph patch for Chapter {chapter_id}, Paragraph {para_id} has empty text. Skipping patch.[/yellow]"
                            )
                            continue

                        target_para = find_paragraph(
                            target_chapter, para_id
                        )  # Searches within target_content implicitly
                        if target_para is not None:
                            target_para.text = para_patch.text
                            # Update attributes if needed (example: status)
                            for key, value in para_patch.attrib.items():
                                if key != "id":  # Don't overwrite id
                                    target_para.set(key, value)
                            paras_patched_count += 1
                            applied_changes = True
                        else:
                            # Option: Append the paragraph if it doesn't exist?
                            # console.print(f"[yellow]Warning: Paragraph ID '{para_id}' in patch for Chapter {chapter_id} not found. Appending instead.[/yellow]")
                            # new_para = copy.deepcopy(para_patch) # Deepcopy patch element
                            # target_content.append(new_para)
                            # paras_patched_count += 1
                            # applied_changes = True
                            # OR just warn and skip:
                            console.print(
                                f"[yellow]Warning: Paragraph ID '{para_id}' in patch for Chapter {chapter_id} not found. Skipping.[/yellow]"
                            )

                    if paras_patched_count > 0:
                        console.print(
                            f"[green]Applied paragraph patch to {paras_patched_count} paragraph(s) in Chapter {chapter_id}.[/green]"
                        )
                        if chapter_id not in chapters_patched:
                            chapters_patched.append(chapter_id)

            # --- Apply Top-Level Patches ---
            # Title
            new_title_elem = patch_root.find("title")
            if new_title_elem is not None:  # Check element exists
                title_elem = self.book_root.find("title")
                new_title_text = new_title_elem.text or ""  # Handle empty tag
                if title_elem is not None:
                    title_elem.text = new_title_text
                else:
                    # Insert title at the beginning if missing
                    title_elem = ET.Element("title")  # Create element first
                    title_elem.text = new_title_text
                    self.book_root.insert(0, title_elem)
                console.print("[green]Applied title patch.[/green]")
                self.book_title_slug = slugify(new_title_text)  # Update slug
                applied_changes = True
            # Synopsis
            new_synopsis_elem = patch_root.find("synopsis")
            if new_synopsis_elem is not None:
                synopsis_elem = self.book_root.find("synopsis")
                new_synopsis_text = new_synopsis_elem.text or ""
                if synopsis_elem is not None:
                    synopsis_elem.text = new_synopsis_text
                else:
                    # Insert synopsis after title (index 1) if missing
                    synopsis_elem = ET.Element("synopsis")
                    synopsis_elem.text = new_synopsis_text
                    # Find index carefully
                    title_index = -1
                    for i, child in enumerate(self.book_root):
                        if child.tag == "title":
                            title_index = i
                            break
                    self.book_root.insert(
                        title_index + 1 if title_index != -1 else 0, synopsis_elem
                    )
                console.print("[green]Applied synopsis patch.[/green]")
                applied_changes = True
            # Characters (Full Replace - be cautious)
            new_characters_elem = patch_root.find("characters")
            if new_characters_elem is not None:
                old_characters = self.book_root.find("characters")
                if old_characters is not None:
                    self.book_root.remove(old_characters)
                # Insert characters typically after synopsis (find appropriate index)
                synopsis_index = -1
                title_index = -1
                initial_idea_index = -1
                for i, child in enumerate(self.book_root):
                    if child.tag == "title":
                        title_index = i
                    if child.tag == "initial_idea":
                        initial_idea_index = i  # Consider new idea tag
                    if child.tag == "synopsis":
                        synopsis_index = i
                    # No break, find all relevant tags first

                insert_index = 0
                if synopsis_index != -1:
                    insert_index = synopsis_index + 1
                elif initial_idea_index != -1:
                    insert_index = initial_idea_index + 1
                elif title_index != -1:
                    insert_index = title_index + 1

                self.book_root.insert(insert_index, copy.deepcopy(new_characters_elem))
                console.print("[green]Applied full character list patch.[/green]")
                applied_changes = True

            # Mark chapters as generated if content was added/changed
            for chap_id in chapters_patched:
                self.chapters_generated_in_session.add(chap_id)

        except Exception as e:
            console.print(f"[bold red]Error applying patch: {e}[/bold red]")
            console.print_exception(show_locals=False)  # Show traceback
            # console.print("[yellow]Reverting to state before patch attempt - Disabled[/yellow]")
            # self.book_root = original_book_root # Revert on error (disabled)
            # Log the failed patch attempt (Logging removed)
            return False

        if not applied_changes:
            console.print(
                "[yellow]Patch processed, but no applicable changes were found or applied.[/yellow]"
            )

        return applied_changes

    def _display_summary(self):
        """Displays a summary of the current book state, including word counts."""
        if self.book_root is None:
            console.print("[red]Cannot display summary, book data is not loaded.[/red]")
            return

        title = self.book_root.findtext("title", "N/A")
        synopsis = self.book_root.findtext("synopsis", "N/A")
        initial_idea = self.book_root.findtext("initial_idea", "")  # Get the idea
        characters = self.book_root.findall(".//character")
        chapters = self.book_root.findall(
            ".//chapter"
        )  # Find all, sort later if needed

        # --- Calculate Word Counts ---
        total_wc = 0
        chapter_word_counts = {}
        if chapters:
            # Sort chapters by ID numerically before processing counts
            sorted_chapters_for_wc = sorted(chapters, key=lambda c: int(c.get("id", 0)))
            for chap in sorted_chapters_for_wc:
                chap_id = chap.get("id")
                content = chap.find("content")
                chap_wc = 0
                if content is not None:
                    paragraphs = content.findall(".//paragraph")
                    for para in paragraphs:
                        chap_wc += _count_words(para.text)
                chapter_word_counts[chap_id] = chap_wc
                total_wc += chap_wc
        self.total_word_count = total_wc  # Store total word count

        # --- Display ---
        console.print(
            Panel(
                f"Book Summary: [cyan]{title}[/cyan]\n"
                f"[bold]Total Word Count:[/bold] [bright_magenta]{self.total_word_count:,}[/bright_magenta]",  # Added word count
                title=f"Current Status ({self.book_dir.name if self.book_dir else 'No Project'})",
                border_style="blue",
            )
        )
        console.print(f"[bold]Synopsis:[/bold]\n{synopsis}\n")

        # Display initial idea snippet if it exists
        if initial_idea:
            console.print(
                f"[bold]Initial Idea/Description:[/bold]\n[dim]{initial_idea[:300]}{'...' if len(initial_idea) > 300 else ''}[/dim]\n"
            )

        # Characters Table
        if characters:
            char_table = Table(title="Characters")
            char_table.add_column("ID", style="dim")
            char_table.add_column("Name")
            char_table.add_column("Description")
            for char in characters:
                char_table.add_row(
                    char.get("id", "N/A"),
                    char.findtext("name", "N/A"),
                    (char.findtext("description", "N/A") or "")[:100] + "...",
                )
            console.print(char_table)
        else:
            console.print("[dim]No characters defined yet.[/dim]")

        # Chapters Table
        if chapters:
            chap_table = Table(title=f"Chapters ({len(chapters)} total)")
            chap_table.add_column("ID", style="dim")
            chap_table.add_column("Number")
            chap_table.add_column("Title")
            chap_table.add_column("Summary", justify="center")
            chap_table.add_column("Content", justify="center")
            chap_table.add_column(
                "Word Count", style="magenta", justify="right"
            )  # New column

            # Sort chapters by ID numerically before display
            sorted_chapters_display = sorted(
                chapters, key=lambda c: int(c.get("id", 0))
            )

            for chap in sorted_chapters_display:
                chap_id = chap.get("id", "N/A")
                num = chap.findtext("number", "N/A")
                chap_title = chap.findtext("title", "N/A")
                summary = chap.findtext("summary", "").strip()
                content = chap.find("content")
                paragraphs = (
                    content.findall(".//paragraph") if content is not None else []
                )
                # Check if paragraphs actually have non-empty text
                has_real_content = any(p.text and p.text.strip() for p in paragraphs)
                chap_wc = chapter_word_counts.get(chap_id, 0)  # Get pre-calculated wc

                summary_status = "[green]✓[/green]" if summary else "[red]✗[/red]"
                content_status = (
                    "[green]✓[/green]" if has_real_content else "[red]✗[/red]"
                )

                if chap_id in self.chapters_generated_in_session:
                    content_status += " [cyan](new)[/cyan]"  # Mark newly generated

                chap_table.add_row(
                    chap_id,
                    num,
                    chap_title,
                    summary_status,
                    content_status,
                    f"{chap_wc:,}",  # Display wc
                )
            console.print(chap_table)
        else:
            console.print("[dim]No chapters defined yet.[/dim]")

    # --- Step 1: Outline Generation ---
    def generate_outline(self):
        """Guides the user and LLM to generate the full book outline (characters, chapter summaries)."""
        console.print(Panel("Step 1: Generating Full Book Outline", style="bold blue"))
        if self.book_root is None:
            console.print(
                "[red]Cannot generate outline, book data structure is missing.[/red]"
            )
            return False  # Indicate failure

        # Check if chapters element exists and has children with summaries
        chapters_element = self.book_root.find(".//chapters")
        has_chapters = chapters_element is not None and len(chapters_element) > 0
        has_summaries = chapters_element is not None and any(
            c.findtext("summary", "").strip()
            for c in chapters_element.findall("chapter")
        )
        has_characters = self.book_root.find(".//characters") is not None and bool(
            self.book_root.findall(".//character")
        )

        if has_chapters and has_summaries and has_characters:
            if not Confirm.ask(
                "[yellow]Full outline (chapters with summaries, characters) seems to exist. Regenerate? (This will replace existing structure)[/yellow]",
                default=False,
            ):
                console.print("Skipping full outline generation.")
                return True  # Indicate success (no action needed)

        # Get required info (use existing title/idea if available)
        current_title = self.book_root.findtext("title", "Unknown Title")
        current_idea = self.book_root.findtext(
            "initial_idea",
            self.book_root.findtext("synopsis", "No description available"),
        )  # Use idea or synopsis

        num_chapters = IntPrompt.ask(
            "[yellow]Approximate number of chapters for the full outline? (e.g., 20)[/yellow]",
            default=20,
        )

        # Create the prompt using current state (include idea/synopsis)
        current_book_xml_for_prompt = ET.tostring(
            self.book_root, encoding="unicode"
        )  # Already includes title, synopsis, idea

        prompt = f"""
You are a creative assistant expanding an initial book concept into a full outline.
The current minimal state of the book is:
```xml
{current_book_xml_for_prompt}
```

Based on the title, synopsis, and initial idea (if present), please generate the missing or incomplete parts of the outline, specifically:
1.  A detailed `<characters>` section with multiple `<character>` elements (ID, name, description). Define the main characters and their arcs briefly. Use CamelCase or simple alphanumeric IDs (e.g., 'mainHero', 'villain').
2.  A detailed `<chapters>` section containing approximately {num_chapters} `<chapter>` elements. For each chapter:
    *   Ensure it has a unique sequential string `id` (e.g., '1', '2', ...).
    *   Include `<number>` (matching the id), `<title>`, and a detailed `<summary>` (150-200 words) outlining key events and progression.
    *   Keep the `<content>` tag EMPTY.
    *   Ensure the summaries form a coherent narrative arc (setup, rising action, climax, falling action, resolution).

Output ONLY the complete `<book>` XML structure, merging the generated details with the existing title/synopsis/initial_idea. Do not include any text outside the `<book>` tags. Ensure IDs are correct.
"""
        response_xml_str = self._get_llm_response(prompt, "Generating full outline")

        if response_xml_str:
            # Expecting 'book' root tag here
            new_book_root = parse_xml_string(response_xml_str, expected_root_tag="book")

            if new_book_root is not None and new_book_root.tag == "book":
                # --- Validation and Cleanup ---
                validation_passed = True
                chapters_elem = new_book_root.find("chapters")
                if chapters_elem is None:
                    console.print(
                        "[bold red]Generated XML is missing <chapters> tag. Outline invalid.[/bold red]"
                    )
                    validation_passed = False
                else:
                    # Check chapters have required elements + Empty content
                    # Re-number chapters sequentially based on order in XML response
                    chapter_list = chapters_elem.findall("chapter")
                    if not chapter_list:
                        console.print(
                            "[bold red]Generated XML <chapters> tag contains no <chapter> elements. Outline invalid.[/bold red]"
                        )
                        validation_passed = False
                    else:
                        for i, chapter in enumerate(chapter_list):
                            is_valid_chap = True
                            expected_id = str(i + 1)
                            current_id = chapter.get("id")

                            # Correct ID and Number
                            if current_id != expected_id:
                                console.print(
                                    f"[yellow]Warning: Chapter {i + 1} has incorrect ID '{current_id}'. Correcting to '{expected_id}'.[/yellow]"
                                )
                                chapter.set("id", expected_id)
                            num_elem = chapter.find("number")
                            if num_elem is None:
                                num_elem = ET.SubElement(chapter, "number")
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} missing <number>. Adding.[/yellow]"
                                )
                            if num_elem.text != expected_id:
                                if num_elem.text:
                                    console.print(
                                        f"[yellow]Warning: Chapter {expected_id} <number> mismatch ('{num_elem.text}'). Correcting.[/yellow]"
                                    )
                                num_elem.text = expected_id

                            # Check Title
                            if chapter.find("title") is None:
                                ET.SubElement(
                                    chapter, "title"
                                ).text = f"Chapter {expected_id} Title Placeholder"
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} missing title. Added placeholder.[/yellow]"
                                )
                                is_valid_chap = False
                            elif not (chapter.findtext("title", "").strip()):
                                chapter.find(
                                    "title"
                                ).text = f"Chapter {expected_id} Title Placeholder"  # Ensure non-empty
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} has empty title. Added placeholder.[/yellow]"
                                )
                                is_valid_chap = False

                            # Check Summary
                            if chapter.find("summary") is None:
                                ET.SubElement(
                                    chapter, "summary"
                                ).text = f"Summary for chapter {expected_id} needed."
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} missing summary. Added placeholder.[/yellow]"
                                )
                                is_valid_chap = False
                            elif not (chapter.findtext("summary", "").strip()):
                                chapter.find(
                                    "summary"
                                ).text = f"Summary for chapter {expected_id} needed."  # Ensure non-empty
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} has empty summary. Added placeholder.[/yellow]"
                                )
                                is_valid_chap = False

                            # Ensure Content is empty
                            content_elem = chapter.find("content")
                            if content_elem is None:
                                ET.SubElement(
                                    chapter, "content"
                                )  # Add empty content tag
                            elif len(content_elem) > 0 or (
                                content_elem.text and content_elem.text.strip()
                            ):
                                console.print(
                                    f"[yellow]Warning: Chapter {expected_id} <content> tag is not empty. Clearing it.[/yellow]"
                                )
                                content_elem.clear()  # Remove children and text
                                content_elem.text = None

                            if not is_valid_chap:
                                validation_passed = False  # Mark as potentially problematic if placeholders added

                # Characters validation
                chars_elem = new_book_root.find("characters")
                if chars_elem is None:
                    console.print(
                        "[yellow]Warning: Generated XML is missing <characters> tag. Creating empty one.[/yellow]"
                    )
                    # Insert it in a reasonable place if possible
                    synopsis_index = -1
                    title_index = -1
                    initial_idea_index = -1
                    for i, child in enumerate(new_book_root):
                        if child.tag == "title":
                            title_index = i
                        if child.tag == "initial_idea":
                            initial_idea_index = i
                        if child.tag == "synopsis":
                            synopsis_index = i
                    insert_index = 0
                    if synopsis_index != -1:
                        insert_index = synopsis_index + 1
                    elif initial_idea_index != -1:
                        insert_index = initial_idea_index + 1
                    elif title_index != -1:
                        insert_index = title_index + 1
                    chars_elem = ET.Element("characters")
                    new_book_root.insert(insert_index, chars_elem)
                    validation_passed = False  # Missing characters is an issue

                else:
                    used_char_ids = set()
                    valid_chars_found = False
                    for char in chars_elem.findall("character"):
                        valid_chars_found = True
                        char_id = char.get("id")
                        name_elem = char.find("name")
                        name = (
                            name_elem.text
                            if name_elem is not None
                            and name_elem.text
                            and name_elem.text.strip()
                            else None  # Mark as None if missing or empty
                        )

                        if not name:  # Generate name if missing
                            name = f"Character_{uuid.uuid4().hex[:4]}"
                            if name_elem is not None:
                                name_elem.text = name
                            else:
                                ET.SubElement(char, "name").text = name
                            console.print(
                                f"[yellow]Warning: Character missing name. Assigned '{name}'.[/yellow]"
                            )

                        if (
                            not char_id
                            or not char_id.strip()
                            or char_id in used_char_ids
                        ):
                            base_id = (
                                slugify(name, allow_unicode=False).replace("-", "")
                                or f"Char{uuid.uuid4().hex[:4]}"
                            )
                            new_id = base_id
                            counter = 1
                            while new_id in used_char_ids:
                                new_id = f"{base_id}{counter}"
                                counter += 1
                            console.print(
                                f"[yellow]Warning: Character '{name}' missing/duplicate ID '{char_id}'. Setting unique ID to '{new_id}'.[/yellow]"
                            )
                            char.set("id", new_id)
                            char_id = new_id  # Use the new one
                        used_char_ids.add(char_id)

                        # Check description
                        desc_elem = char.find("description")
                        if desc_elem is None:
                            ET.SubElement(
                                char, "description"
                            ).text = "Description needed."
                        elif not desc_elem.text or not desc_elem.text.strip():
                            desc_elem.text = "Description needed."

                    if not valid_chars_found:
                        console.print(
                            "[yellow]Warning: Generated <characters> tag has no <character> elements.[/yellow]"
                        )
                        validation_passed = (
                            False  # Empty characters section is an issue
                        )

                # Ensure Title/Synopsis/Idea exist (copy from original if LLM removed them)
                for tag_info in [("title", 0), ("synopsis", 1), ("initial_idea", 2)]:
                    tag, default_pos = tag_info
                    if new_book_root.find(tag) is None:
                        original_elem = self.book_root.find(tag)
                        insert_elem = None
                        if original_elem is not None:
                            console.print(
                                f"[yellow]Warning: LLM removed <{tag}>. Restoring from original.[/yellow]"
                            )
                            insert_elem = copy.deepcopy(original_elem)
                        else:
                            # Add placeholder if it was missing originally too
                            console.print(
                                f"[yellow]Warning: LLM removed <{tag}> and it was missing originally. Adding placeholder.[/yellow]"
                            )
                            insert_elem = ET.Element(tag)
                            insert_elem.text = (
                                f"{tag.replace('_', ' ').title()} needed."
                            )

                        if insert_elem is not None:
                            # Try to insert at a reasonable position
                            current_tags = [child.tag for child in new_book_root]
                            insert_at = default_pos
                            # Adjust position based on what's already there
                            if tag == "synopsis" and "title" in current_tags:
                                insert_at = current_tags.index("title") + 1
                            elif tag == "initial_idea":
                                if "synopsis" in current_tags:
                                    insert_at = current_tags.index("synopsis") + 1
                                elif "title" in current_tags:
                                    insert_at = current_tags.index("title") + 1
                            # Ensure insert position is valid
                            insert_at = min(insert_at, len(new_book_root))
                            new_book_root.insert(insert_at, insert_elem)
                        if tag in [
                            "title",
                            "synopsis",
                        ]:  # Missing essential tags is an issue
                            validation_passed = False

                # --- Commit Changes ---
                if validation_passed:
                    self.book_root = (
                        new_book_root  # Replace current root with the validated one
                    )
                    save_ok = self._save_book_state(
                        "outline.xml"
                    )  # Save as the definitive outline
                    if save_ok:
                        console.print(
                            "[green]Full outline generated/updated and saved as outline.xml[/green]"
                        )
                        # Update internal state if title changed
                        self.book_title_slug = slugify(
                            self.book_root.findtext("title", "untitled")
                        )
                        self._display_summary()  # Display updated summary
                        self.patch_counter = (
                            0  # Reset patch counter as this is the new base
                        )
                        self.chapters_generated_in_session.clear()
                        return True  # Indicate success
                    else:
                        console.print(
                            "[bold red]Outline generated but failed to save state.[/bold red]"
                        )
                        return False  # Indicate failure to save

                else:
                    console.print(
                        "[bold red]Outline generation completed, but validation issues occurred (check warnings). Outline saved, but review recommended.[/bold red]"
                    )
                    # Still save the potentially problematic outline, but warn user
                    self.book_root = new_book_root
                    save_ok = self._save_book_state("outline.xml")
                    self.patch_counter = 0
                    self.chapters_generated_in_session.clear()
                    # Even with validation errors, we consider the *step* completed, but maybe return False if critical?
                    # For now, let's return True to allow proceeding, but user is warned.
                    self._display_summary()  # Display potentially problematic summary
                    return save_ok  # Return True if saved, False otherwise

            else:
                console.print(
                    "[bold red]Failed to parse the generated outline XML or root tag was not <book>. Outline not saved.[/bold red]"
                )
                return False
        else:
            console.print(
                "[bold red]Failed to get a valid response from the LLM for the outline.[/bold red]"
            )
            return False

    # --- Step 2: Chapter Content Generation (Batch Mode) ---
    def _select_chapters_to_generate(self, batch_size=5):
        """Selects the next batch of chapters to generate using a spread/fill strategy."""
        if self.book_root is None:
            return []

        all_chapters = sorted(
            self.book_root.findall(".//chapter"),
            key=lambda c: int(c.get("id", 0)),  # Sort by numeric ID
        )
        if not all_chapters:
            return []

        total_chapters = len(all_chapters)
        chapters_with_content_ids = set()
        for chap in all_chapters:
            content = chap.find("content")
            # Check if content tag exists and has any paragraph children with actual text
            if content is not None:
                paragraphs = content.findall(".//paragraph")
                if any(p.text and p.text.strip() for p in paragraphs):
                    chapters_with_content_ids.add(chap.get("id"))

        chapters_needing_content = [
            chap
            for chap in all_chapters
            if chap.get("id") not in chapters_with_content_ids
        ]

        if not chapters_needing_content:
            return []  # All done

        selected_chapters = []
        if not chapters_with_content_ids:  # Initial spread if nothing generated yet
            console.print("[cyan]Selecting initial spread of chapters...[/cyan]")
            indices_to_pick = set()
            if total_chapters > 0:
                indices_to_pick.add(0)  # First
            if total_chapters > 1:
                indices_to_pick.add(total_chapters - 1)  # Last

            # Add intermediate chapters (using integer division for steps)
            if batch_size > 2 and total_chapters >= batch_size:
                steps = batch_size - 1
                for i in range(1, steps):
                    # Calculate index, ensuring it's within bounds and not start/end
                    index = min(
                        total_chapters - 1,
                        max(0, int(round(i * total_chapters / steps))),
                    )
                    if index != 0 and index != total_chapters - 1:
                        indices_to_pick.add(index)
            elif (
                total_chapters > 2
            ):  # Add middle if possible and different from start/end
                mid_index = total_chapters // 2
                if mid_index != 0 and mid_index != total_chapters - 1:
                    indices_to_pick.add(mid_index)

            # Ensure we don't exceed batch size, prioritize unique indices
            potential_indices = sorted(list(indices_to_pick))
            actual_indices = potential_indices[:batch_size]

            # Retrieve the actual chapter elements for the selected indices
            potential_chapters = [
                all_chapters[i] for i in actual_indices if 0 <= i < total_chapters
            ]
            # Final check: only include chapters that actually *need* content according to our definition
            selected_chapters = [
                ch
                for ch in potential_chapters
                if ch.get("id") not in chapters_with_content_ids
            ]

        else:  # Fill gaps
            console.print("[cyan]Selecting next batch to fill gaps...[/cyan]")
            gaps = []
            current_gap = []
            for chap in all_chapters:
                if chap.get("id") not in chapters_with_content_ids:
                    current_gap.append(chap)
                else:
                    if current_gap:
                        gaps.append(current_gap)
                        current_gap = []
            if current_gap:  # Add the last gap if it exists
                gaps.append(current_gap)

            # Simple strategy: flatten gaps and take first N
            chapters_from_gaps = [chap for gap in gaps for chap in gap]
            selected_chapters = chapters_from_gaps[:batch_size]

        # Ensure selected_chapters only contains unique chapters (should be guaranteed by logic above, but safe)
        seen_ids = set()
        unique_selected = []
        for chap in selected_chapters:
            chap_id = chap.get("id")
            if chap_id not in seen_ids:
                unique_selected.append(chap)
                seen_ids.add(chap_id)

        return unique_selected

    def generate_chapters(self):
        """Generates content for batches of chapters."""
        console.print(
            Panel("Step 2: Generating Chapter Content (Batch Mode)", style="bold blue")
        )
        if self.book_root is None:
            console.print("[red]Cannot generate chapters, book data missing.[/red]")
            return False  # Indicate failure (or inability to run)

        self.chapters_generated_in_session.clear()  # Reset for this run
        chapters_were_generated = False  # Flag to track if any content was attempted

        while True:
            chapters_to_generate = self._select_chapters_to_generate(
                batch_size=5
            )  # Increased batch size example

            if not chapters_to_generate:
                # Double check if any are *really* missing content based on the precise definition
                truly_needs_generation = False
                all_chapters_check = self.book_root.findall(".//chapter")
                for c in all_chapters_check:
                    content = c.find("content")
                    if content is None:
                        truly_needs_generation = True
                        break
                    paragraphs = content.findall(".//paragraph")
                    if not paragraphs or not any(
                        p.text and p.text.strip() for p in paragraphs
                    ):
                        truly_needs_generation = True
                        break

                if truly_needs_generation:
                    console.print(
                        "[yellow]Could not select more chapters via strategy, but some still need content. Manual check needed? Exiting generation.[/yellow]"
                    )
                elif (
                    chapters_were_generated
                ):  # Only print success if we actually did something
                    console.print(
                        "[bold green]\nAll chapters appear to have content now![/bold green]"
                    )
                else:  # If no chapters needed generation from the start
                    console.print(
                        "[cyan]No chapters required content generation in this run.[/cyan]"
                    )
                break  # Exit generation loop

            chapters_were_generated = True  # Mark that we are attempting generation
            chapter_ids_str = ", ".join(
                [c.get("id", "N/A") for c in chapters_to_generate]
            )
            console.print(
                f"\nAttempting to generate content for {len(chapters_to_generate)} chapters (IDs: {chapter_ids_str})..."
            )

            chapter_details_prompt = ""
            for chapter_element in chapters_to_generate:
                chap_id = chapter_element.get("id")
                chap_num = chapter_element.findtext("number", "N/A")
                chap_title = chapter_element.findtext("title", "N/A")
                chap_summary = chapter_element.findtext("summary", "N/A")
                chapter_details_prompt += f'- Chapter {chap_num} (ID: {chap_id}): "{chap_title}"\n  Summary: {chap_summary}\n'

            current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

            # Adjusted word count goal
            prompt = f"""
You are a novelist continuing the story based on the full book context provided below.
Your task is to write the full prose content for the following {len(chapters_to_generate)} chapters:
{chapter_details_prompt}

Guidelines:
- Write detailed and engaging prose for each chapter, aiming for a substantial length appropriate for a novel chapter (e.g., 1500-4000 words *per chapter*). Adjust length based on the chapter's summary and importance within the narrative arc.
- Maintain consistency with the established plot, characters (personalities, motivations, relationships), tone, and writing style evident in the rest of the book context (including summaries of unwritten chapters and content of written ones).
- Ensure the events of these chapters align with their summaries and logically connect preceding/succeeding chapters. Pay attention to how these chapters function together within the narrative arc.
- For EACH chapter you generate content for, structure it within `<content>` tags, divided into paragraphs using `<paragraph>` tags. Each paragraph tag MUST have a unique sequential `id` attribute within that chapter (e.g., `<paragraph id="1">...</paragraph>`, `<paragraph id="2">...</paragraph>`, etc.). Start paragraph IDs from "1" for each chapter. Ensure paragraphs contain meaningful text.
- Output ONLY an XML `<patch>` structure containing the `<chapter>` elements for the requested chapters. Each `<chapter>` element must have the correct `id` attribute and contain the fully generated `<content>` tag with its `<paragraph>` children. DO NOT include chapters that were not requested in this batch.

Example Output Format:
<patch>
    <chapter id="[ID of first requested chapter]">
        <content>
            <paragraph id="1">First paragraph text...</paragraph>
            <paragraph id="2">Second paragraph text...</paragraph>
            ...
        </content>
    </chapter>
    <chapter id="[ID of second requested chapter]">
        <content>
            <paragraph id="1">Another first paragraph...</paragraph>
            <paragraph id="2">...</paragraph>
            ...
        </content>
    </chapter>
    <!-- ... include all requested chapters ... -->
</patch>

Full Book Context (including outline and any previously generated chapters):
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML containing the content for the requested chapters now. Ensure all specified chapters are included in the response patch with correct IDs and paragraph structure. Ensure generated paragraphs have sequential IDs starting from 1 for each chapter and contain text.
"""

            response_patch_xml_str = self._get_llm_response(
                prompt,
                f"Writing Chapters {chapter_ids_str}",
                allow_stream=True,  # Streaming preferred for long content
            )

            # Check if response was obtained (handles None return from _get_llm_response)
            if response_patch_xml_str is None:
                console.print(
                    f"[bold red]Failed to get response from LLM for batch starting with Chapter {chapters_to_generate[0].get('id')}.[/bold red]"
                )
                # _get_llm_response handles the confirmation to retry/abort. If it returns None, it failed definitively.
                if not Confirm.ask(
                    "[yellow]Failed to get content for this batch. Continue trying the NEXT batch? [/yellow]",
                    default=True,
                ):
                    console.print("Aborting chapter generation.")
                    return False  # Indicate failure to continue
                else:
                    continue  # Skip apply/save for this failed batch, try next selection

            # --- Apply and Save the patch ---
            next_patch_num = get_next_patch_number(self.book_dir)
            if self._apply_patch(response_patch_xml_str):
                self.patch_counter = next_patch_num
                patch_filename = f"patch-{self.patch_counter:02d}.xml"
                if self._save_book_state(patch_filename):  # Save the patch file
                    # Also generate versioned and final HTML
                    html_filename = f"version-{self.patch_counter:02d}.html"
                    self._generate_html_output(html_filename)
                    self._generate_html_output(
                        "latest.html"
                    )  # Always update a 'latest' html file
                    console.print(
                        f"[green]Successfully generated, applied, and saved patch {patch_filename}. HTML updated.[/green]"
                    )
                    self._display_summary()  # Display summary showing new chapters
                else:
                    console.print(
                        f"[bold red]Failed to save state file {patch_filename} after applying patch. HTML not updated.[/bold red]"
                    )
                    # Consider reverting apply_patch? Or just warn and continue? For now, warn.
                    if not Confirm.ask(
                        "[yellow]Failed to save state. Continue trying the NEXT batch? [/yellow]",
                        default=True,
                    ):
                        console.print(
                            "Aborting chapter generation due to save failure."
                        )
                        return False
            else:
                console.print(
                    f"[bold red]Failed to apply generated content patch for batch starting with Chapter {chapters_to_generate[0].get('id')}. State not saved for this batch.[/bold red]"
                )
                # Ask to retry or skip batch?
                if not Confirm.ask(
                    "[yellow]Failed to apply content patch. Continue trying the NEXT batch? [/yellow]",
                    default=True,
                ):
                    console.print("Aborting chapter generation.")
                    return False  # Indicate failure to continue
                # else: continue loop to try next batch

            time.sleep(1)  # Small delay between batches

        return True  # Indicate chapter generation phase completed (even if nothing was generated or some batches failed)

    # --- Editing Step Helper Methods ---

    def _handle_patch_result(self, success, operation_desc="Operation"):
        """Handles saving and HTML generation after a patch attempt."""
        if success:
            next_patch_num = get_next_patch_number(self.book_dir)
            self.patch_counter = next_patch_num
            patch_filename = f"patch-{self.patch_counter:02d}.xml"
            if self._save_book_state(patch_filename):
                html_filename = f"version-{self.patch_counter:02d}.html"
                self._generate_html_output(html_filename)
                self._generate_html_output("latest.html")  # Keep latest HTML up-to-date
                console.print(
                    f"[green]{operation_desc} successful. Patch saved as {patch_filename}. HTML updated.[/green]"
                )
                return True
            else:
                console.print(
                    f"[bold red]{operation_desc} applied, but failed to save state file {patch_filename}. HTML not updated.[/bold red]"
                )
                return False  # Indicate save failure
        else:
            console.print(
                f"[bold red]{operation_desc} failed to apply patch. State not saved.[/bold red]"
            )
            return False  # Indicate apply failure

    def _get_chapter_selection(
        self,
        prompt_text="Enter chapter ID(s) to modify (comma-separated)",
        allow_multiple=True,
    ):
        """Prompts user for chapter IDs and validates them."""
        if self.book_root is None:
            return []
        all_chapter_ids = {
            chap.get("id") for chap in self.book_root.findall(".//chapter")
        }
        if not all_chapter_ids:
            console.print("[yellow]No chapters found in the book.[/yellow]")
            return []

        while True:
            raw_input = Prompt.ask(
                f"{prompt_text} (Available: {', '.join(sorted(all_chapter_ids, key=int))})"
            )
            selected_ids_str = [s.strip() for s in raw_input.split(",") if s.strip()]

            if not selected_ids_str:
                console.print("[red]No chapter ID entered. Please try again.[/red]")
                continue

            if not allow_multiple and len(selected_ids_str) > 1:
                console.print(
                    "[red]Only one chapter ID is allowed for this operation. Please enter a single ID.[/red]"
                )
                continue

            selected_chapters = []
            invalid_ids = []
            valid_ids = set()
            for chap_id_str in selected_ids_str:
                if chap_id_str in all_chapter_ids:
                    if chap_id_str not in valid_ids:  # Avoid duplicates
                        chapter_elem = find_chapter(self.book_root, chap_id_str)
                        if chapter_elem is not None:
                            selected_chapters.append(chapter_elem)
                            valid_ids.add(chap_id_str)
                        else:
                            # Should not happen if ID is in all_chapter_ids, but safety check
                            invalid_ids.append(chap_id_str + " (internal error)")
                    # else: skip duplicate
                else:
                    invalid_ids.append(chap_id_str)

            if invalid_ids:
                console.print(
                    f"[red]Invalid or duplicate chapter IDs entered: {', '.join(invalid_ids)}. Please try again.[/red]"
                )
                continue

            return selected_chapters  # Return list of validated chapter elements

    def _edit_make_longer(self):
        """Handler for making chapters longer."""
        console.print(Panel("Edit Option: Make Chapter(s) Longer", style="cyan"))
        selected_chapters = self._get_chapter_selection(
            prompt_text="Enter chapter ID(s) to make longer (comma-separated)",
            allow_multiple=True,
        )
        if not selected_chapters:
            return  # User cancelled or no chapters

        target_word_count = IntPrompt.ask(
            "[yellow]Enter target word count per chapter[/yellow]",
            default=max(
                3000,
                math.ceil(
                    self.total_word_count
                    / len(self.book_root.findall(".//chapter"))
                    * 1.5
                )
                if self.total_word_count > 0
                else 3000,
            ),  # Suggest 50% increase or 3000
        )
        if target_word_count <= 0:
            console.print("[red]Invalid word count.[/red]")
            return

        chapter_details_prompt = ""
        chapter_ids = []
        for chapter_element in selected_chapters:
            chap_id = chapter_element.get("id")
            chap_num = chapter_element.findtext("number", "N/A")
            chap_title = chapter_element.findtext("title", "N/A")
            chap_summary = chapter_element.findtext("summary", "N/A")
            chapter_ids.append(chap_id)
            # Include existing content for context? Yes, better for expansion.
            existing_content_xml = ""
            content_elem = chapter_element.find("content")
            if content_elem is not None:
                existing_content_xml = ET.tostring(content_elem, encoding="unicode")

            chapter_details_prompt += f"""
<chapter id="{chap_id}">
  <number>{chap_num}</number>
  <title>{escape(chap_title)}</title>
  <summary>{escape(chap_summary)}</summary>
  <!-- Existing Content -->
  {existing_content_xml}
</chapter>
"""

        current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")
        prompt = f"""
You are a novelist tasked with expanding specific chapters of a manuscript.
Your goal is to rewrite the *entire content* for the chapter(s) listed below, significantly increasing their length to approximately {target_word_count:,} words *each*, while enriching the detail, description, dialogue, and internal monologue.

Chapters to Expand (including their original content for context):
---
{chapter_details_prompt}
---

Guidelines:
- Rewrite the full `<content>` for EACH specified chapter ({", ".join(chapter_ids)}).
- Target word count: Approximately {target_word_count:,} words per chapter.
- Elaborate on existing scenes, add descriptive details, deepen character thoughts/feelings, expand dialogues, and potentially add small connecting scenes *within* the chapter's scope if necessary to reach the target length naturally.
- Maintain absolute consistency with the overall plot, established characters, tone, and style provided in the full book context below. The expansion should feel seamless.
- Ensure the rewritten chapter still fulfills the purpose outlined in its original summary.
- Structure the rewritten content within `<content>` tags, using sequentially numbered `<paragraph id="...">` tags starting from 1 for each chapter. Ensure paragraphs contain text.
- Output ONLY the XML `<patch>` structure containing the rewritten `<chapter>` elements (with their new, full `<content>`). Do not include chapters not specified.

Full Book Context:
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML now.
"""
        suggested_patch_xml_str = self._get_llm_response(
            prompt, f"Expanding Chapters {', '.join(chapter_ids)}", allow_stream=True
        )
        if suggested_patch_xml_str:
            apply_success = self._apply_patch(suggested_patch_xml_str)
            self._handle_patch_result(
                apply_success, f"Make Longer (Ch {', '.join(chapter_ids)})"
            )

    def _edit_rewrite_chapter(self, blackout=False):
        """Handler for rewriting a chapter, optionally with blackout."""
        mode = "Fresh Rewrite" if blackout else "Rewrite"
        console.print(Panel(f"Edit Option: {mode} Chapter", style="cyan"))
        selected_chapters = self._get_chapter_selection(
            prompt_text=f"Enter the chapter ID to {mode.lower()}",
            allow_multiple=False,  # Only one chapter at a time for rewrite
        )
        if not selected_chapters:
            return
        target_chapter = selected_chapters[0]
        chapter_id = target_chapter.get("id")

        instructions = Prompt.ask(
            "[yellow]Enter specific instructions for the rewrite[/yellow]"
        )
        if not instructions.strip():
            console.print("[red]No instructions provided. Aborting rewrite.[/red]")
            return

        # --- Prepare Context ---
        current_book_xml_str = ""
        if blackout:
            console.print(
                "[yellow]Preparing context with target chapter content removed (fresh rewrite)...[/yellow]"
            )
            temp_book_root = copy.deepcopy(self.book_root)
            target_chapter_copy = find_chapter(temp_book_root, chapter_id)
            if target_chapter_copy is not None:
                content_elem_copy = target_chapter_copy.find("content")
                if content_elem_copy is not None:
                    content_elem_copy.clear()
                    content_elem_copy.text = None  # Explicitly clear text too
                    console.print(
                        f"[dim]Content of chapter {chapter_id} removed for prompt context.[/dim]"
                    )
                else:
                    console.print(
                        f"[yellow]Warning: Chapter {chapter_id} has no <content> tag in temporary copy.[/yellow]"
                    )
            else:
                # Should not happen given validation, but safety check
                console.print(
                    f"[red]Error: Could not find chapter {chapter_id} in temporary copy for fresh rewrite. Aborting.[/red]"
                )
                return
            current_book_xml_str = ET.tostring(temp_book_root, encoding="unicode")
        else:
            # Standard rewrite: use full context
            current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

        # --- Prepare Prompt ---
        chap_num = target_chapter.findtext("number", "N/A")
        chap_title = target_chapter.findtext("title", "N/A")
        chap_summary = target_chapter.findtext("summary", "N/A")

        rewrite_context_info = f"""
Chapter to Rewrite:
- ID: {chapter_id}
- Number: {chap_num}
- Title: {escape(chap_title)}
- Summary: {escape(chap_summary)}
"""
        if not blackout:  # Include original content if not blackout
            original_content_xml = ""
            content_elem = target_chapter.find("content")
            if content_elem is not None:
                original_content_xml = ET.tostring(content_elem, encoding="unicode")
            rewrite_context_info += (
                f"<!-- Original Content -->\n{original_content_xml}\n"
            )

        prompt = f"""
You are a novelist rewriting a specific chapter based on user instructions.
Your task is to rewrite the *entire content* for Chapter {chapter_id}.

{rewrite_context_info}

User's Rewrite Instructions:
---
{escape(instructions)}
---
{"*Note: The original content of this chapter was intentionally removed from the context below to encourage a fresh perspective based on the summary and instructions.*" if blackout else ""}

Guidelines:
- Rewrite the full `<content>` for Chapter {chapter_id} according to the instructions.
- Ensure the rewritten chapter aligns with its summary and maintains consistency with the overall plot, characters, tone, and style provided in the full book context below.
- Structure the rewritten content within `<content>` tags, using sequentially numbered `<paragraph id="...">` tags starting from 1. Ensure paragraphs contain text.
- Output ONLY the XML `<patch>` structure containing the single rewritten `<chapter>` element (ID: {chapter_id}) with its new, full `<content>`.

Full Book Context {"(Chapter " + chapter_id + " content removed)" if blackout else ""}:
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML for the rewritten Chapter {chapter_id} now.
"""

        suggested_patch_xml_str = self._get_llm_response(
            prompt, f"{mode} Chapter {chapter_id}", allow_stream=True
        )
        if suggested_patch_xml_str:
            apply_success = self._apply_patch(suggested_patch_xml_str)
            self._handle_patch_result(apply_success, f"{mode} (Ch {chapter_id})")

    def _edit_suggest_edits(self):
        """Handler for asking the LLM for edit suggestions."""
        console.print(Panel("Edit Option: Ask LLM for Suggestions", style="cyan"))
        if self.book_root is None:
            return

        current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

        # --- Prompt 1: Get Suggestions ---
        prompt_suggest = f"""
You are an expert editor reviewing the novel manuscript provided below.
Analyze the entire book context (plot, pacing, character arcs, consistency, style, clarity, dialogue, descriptions).
Identify potential areas for improvement.
Provide a numbered list of 5-10 concrete, actionable suggestions for specific edits. Keep suggestions concise (1-2 sentences each). Focus on high-impact changes.

Example Suggestion Format:
1. Strengthen the foreshadowing in Chapter 3 regarding the villain's true motives.
2. Improve the pacing of the chase scene in Chapter 12 by shortening descriptive paragraphs.
3. Make Character A's dialogue in Chapter 5 sound more hesitant to reflect their uncertainty.

Full Book Context:
```xml
{current_book_xml_str}
```

Generate the numbered list of edit suggestions now. Output ONLY the list.
"""
        suggestions_text = self._get_llm_response(
            prompt_suggest, "Generating edit suggestions list", allow_stream=False
        )

        if not suggestions_text or not suggestions_text.strip():
            console.print("[red]Could not get suggestions from the LLM.[/red]")
            return

        # --- Parse and Display Suggestions ---
        suggestions = []
        # Try to parse numbered list (simple regex)
        matches = re.findall(r"^\s*(\d+)\.?\s+(.*)", suggestions_text, re.MULTILINE)
        if matches:
            suggestions = [f"{num}. {text.strip()}" for num, text in matches]
        else:
            # Fallback: split by newline if no numbers found
            lines = [
                line.strip() for line in suggestions_text.splitlines() if line.strip()
            ]
            if lines:
                suggestions = [f"{i + 1}. {line}" for i, line in enumerate(lines)]

        if not suggestions:
            console.print(
                "[yellow]LLM response did not contain a parsable list of suggestions.[/yellow]"
            )
            console.print("[dim]LLM Raw Response:[/dim]")
            console.print(f"[dim]{suggestions_text[:1000]}[/dim]")
            return

        console.print("[bold cyan]Suggested Edits:[/bold cyan]")
        choices_map = {}
        display_choices = []
        for i, suggestion in enumerate(suggestions):
            choice_num_str = str(i + 1)
            console.print(suggestion)
            choices_map[choice_num_str] = (
                suggestion  # Map number string to full suggestion text
            )
            display_choices.append(choice_num_str)

        display_choices.append("0")  # Option to cancel
        choices_map["0"] = "Cancel / None"

        chosen_num = Prompt.ask(
            "\n[yellow]Enter the number of the suggestion to implement (or 0 to cancel)[/yellow]",
            choices=display_choices,
            default="0",
        )

        if chosen_num == "0":
            console.print("No suggestion selected.")
            return

        chosen_suggestion_text = choices_map.get(chosen_num)
        # Remove the leading number "N. " from the chosen suggestion for the next prompt
        cleaned_suggestion = re.sub(r"^\d+\.?\s*", "", chosen_suggestion_text).strip()

        console.print(
            f"\nImplementing suggestion {chosen_num}: '[italic]{cleaned_suggestion}[/italic]'"
        )

        # --- Prompt 2: Implement Chosen Suggestion ---
        # Reuse the general edit prompt structure
        prompt_implement = f"""
You are an expert editor implementing a specific suggestion on the novel provided below.

Editing Suggestion to Implement: {cleaned_suggestion}

Guidelines for Patch:
- Analyze the entire book context provided.
- Generate an XML `<patch>` structure that implements the specific suggestion above.
- Patches can modify chapters or paragraphs.
    - Full chapter replace: `<patch><chapter id="..."><content><paragraph id="1">...</paragraph>...</content></chapter></patch>` (Ensure new para IDs are sequential from 1, ensure text).
    - Paragraph replace: `<patch><chapter id="..."><content-patch><paragraph id="para_id">...</paragraph></content-patch></chapter></patch>` (Ensure replacement para has text).
    - Top-level changes (Title, Synopsis, Characters): Use appropriate tags within `<patch>`. Characters list is usually replaced wholesale.
- Be specific and target the changes accurately based on the suggestion.
- Use XML comments `<!-- ... -->` within the patch if brief justification is needed.
- If the suggestion cannot be directly translated to a patch (e.g., too vague), output an empty patch `<patch />` with a comment explaining why.

Full Book Context:
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML to implement the suggestion now. Output ONLY the patch XML.
"""
        suggested_patch_xml_str = self._get_llm_response(
            prompt_implement,
            f"Implementing Suggestion {chosen_num}",
            allow_stream=False,
        )

        if suggested_patch_xml_str:
            console.print(
                Panel(
                    f"[bold cyan]Suggested Patch for Suggestion {chosen_num}:[/bold cyan]",
                    border_style="magenta",
                )
            )
            syntax = Syntax(
                suggested_patch_xml_str, "xml", theme="default", line_numbers=True
            )
            console.print(syntax)

            is_empty_or_comment = False
            try:
                patch_elem = parse_xml_string(suggested_patch_xml_str, "patch")
                if patch_elem is None:
                    is_empty_or_comment = True  # Treat parse failure as empty/failed
                elif not list(patch_elem) or all(
                    child.tag is ET.Comment for child in patch_elem
                ):
                    is_empty_or_comment = True
            except Exception:
                is_empty_or_comment = True  # Treat errors as empty

            if is_empty_or_comment:
                console.print(
                    "[yellow]LLM indicated no changes needed or could not generate a patch for this suggestion.[/yellow]"
                )
            elif Confirm.ask(
                "\n[yellow]Apply this suggested patch? [/yellow]", default=True
            ):
                apply_success = self._apply_patch(suggested_patch_xml_str)
                self._handle_patch_result(
                    apply_success, f"Implement Suggestion {chosen_num}"
                )
            else:
                console.print("Suggested patch discarded.")

    def _edit_general_llm(self):
        """Handler for general LLM editing based on user instructions."""
        console.print(Panel("Edit Option: General Edit Request", style="cyan"))
        instructions = Prompt.ask(
            "[yellow]Enter editing instructions (e.g., 'Improve pacing in chapters 5-7', 'Strengthen character X', 'Check for plot holes') [/yellow]",
            default="Review the entire book for consistency, pacing, character arcs, and suggest improvements via patch.",
        )
        if not instructions.strip():
            console.print("[red]No instructions provided.[/red]")
            return

        current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

        prompt = f"""
You are an expert editor reviewing the novel provided below in XML format.
Your task is to analyze the text based on the user's instructions and generate an XML patch with specific changes.

User's Editing Instructions: {instructions}

Guidelines for Patch:
- Analyze the entire book context provided.
- Identify areas for improvement based on the instructions (e.g., plot holes, inconsistent characterization, weak descriptions, pacing issues, dialogue problems).
- Propose concrete changes formatted STRICTLY as an XML `<patch>` structure.
- Patches can modify chapters or paragraphs.
    - To replace the ENTIRE content of a chapter:
      `<patch><chapter id="chap_id"><content><paragraph id="1">New para 1</paragraph>...</content></chapter></patch>` (Ensure new paras have sequential IDs starting from 1 and contain text).
    - To replace specific paragraphs within a chapter:
      `<patch><chapter id="chap_id"><content-patch><paragraph id="para_id">New text for this para</paragraph>...</content-patch></chapter></patch>` (Ensure replacement paragraph has text).
    - Ensure paragraph IDs referenced in `<content-patch>` exist in the original chapter.
- If suggesting changes to Title, Synopsis, or Characters, use the appropriate top-level tags within the `<patch>` tag (e.g., `<patch><title>New Title</title>...</patch>`). Characters should generally be replaced as a whole list: `<patch><characters><character id="newId1">...</character>...</characters></patch>`.
- Be specific and justify changes briefly if possible using XML comments: `<!-- Suggestion: Rephrased for clarity -->`.
- If no significant changes are needed based on the instructions, output an empty patch like `<patch />` or a patch with only comments: `<patch><!-- No major changes suggested --></patch>`

Full Book Context:
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML containing your suggested edits now. Output ONLY the patch XML.
"""

        suggested_patch_xml_str = self._get_llm_response(
            prompt, "Generating general edit patch", allow_stream=False
        )

        if suggested_patch_xml_str:
            console.print(
                Panel(
                    "[bold cyan]Suggested Edits (Patch XML):[/bold cyan]",
                    border_style="magenta",
                )
            )
            syntax = Syntax(
                suggested_patch_xml_str, "xml", theme="default", line_numbers=True
            )
            console.print(syntax)

            is_empty_or_comment_patch = False
            try:
                patch_elem = parse_xml_string(
                    suggested_patch_xml_str, expected_root_tag="patch"
                )
                if patch_elem is None:
                    is_empty_or_comment_patch = True
                elif not list(patch_elem) or all(
                    child.tag is ET.Comment for child in patch_elem
                ):
                    is_empty_or_comment_patch = True
            except Exception:
                is_empty_or_comment_patch = True

            if is_empty_or_comment_patch:
                console.print(
                    "[cyan]LLM suggested no specific changes or only provided comments.[/cyan]"
                )
            elif Confirm.ask(
                "\n[yellow]Do you want to apply these suggested changes? [/yellow]",
                default=True,
            ):
                apply_success = self._apply_patch(suggested_patch_xml_str)
                self._handle_patch_result(apply_success, "General Edit")
            else:
                console.print("Suggested patch discarded.")

    # --- Export Menu Handler ---
    def _show_export_menu(self):
        """Displays the export options and handles the user choice."""
        console.print(Panel("Export Options", style="bold blue"))

        export_options = {
            "1": "Export Full Book (Single Markdown File)",
            "2": "Export Chapters (Markdown File per Chapter)",
            "3": "Return to Editing Menu",
            "4": "Exit Program",
        }

        while True:
            console.print("\n[bold cyan]Choose an Export Format or Action:[/bold cyan]")
            choices = []
            for key, desc in export_options.items():
                console.print(f"{key}. {desc}")
                choices.append(key)

            choice = Prompt.ask(
                "[yellow]Select option[/yellow]", choices=choices, default="3"
            )

            if choice == "1":
                filename = f"{self.book_title_slug}-full-export.md"
                output_path = self.book_dir / filename
                markdown_exporter.export_single_markdown(self.book_root, output_path)
            elif choice == "2":
                # Pass the main book directory; the exporter function creates the subfolder
                markdown_exporter.export_markdown_per_chapter(
                    self.book_root, self.book_dir, self.book_title_slug
                )
            elif choice == "3":
                return  # Go back to the main editing menu loop
            elif choice == "4":
                console.print(
                    "[bold yellow]Exiting program as requested.[/bold yellow]"
                )
                sys.exit(0)  # Clean exit

    # --- Step 3: Editing (Main Loop) ---
    def edit_book(self):
        """Guides the user through the enhanced editing process."""
        console.print(Panel("Step 3: Editing the Novel (Enhanced)", style="bold blue"))
        if self.book_root is None:
            console.print("[red]Cannot edit, book data missing.[/red]")
            return False  # Indicate failure

        # Generate initial HTML for editing starting point
        self._generate_html_output("latest.html")
        console.print("[green]Generated 'latest.html' with current book state.[/green]")

        edit_options = {
            "1": ("Make Chapter(s) Longer", self._edit_make_longer),
            "2": (
                "Rewrite Chapter (with instructions)",
                lambda: self._edit_rewrite_chapter(blackout=False),
            ),
            "3": (
                "Rewrite Chapter (Fresh Rewrite - keep summary, remove content)",
                lambda: self._edit_rewrite_chapter(blackout=True),
            ),
            "4": ("Ask LLM for Edit Suggestions", self._edit_suggest_edits),
            "5": ("General Edit Request (LLM Patch)", self._edit_general_llm),
            "6": (
                "Export Menu / Exit",
                self._show_export_menu,
            ),  # Changed to Export menu trigger
            "7": ("Finish Editing", None),  # Use None to signal exit
        }

        while True:
            self._display_summary()  # Show current state including word counts
            console.print("\n[bold cyan]Editing Options:[/bold cyan]")
            choices = []
            for key, (desc, _) in edit_options.items():
                console.print(f"{key}. {desc}")
                choices.append(key)

            choice = Prompt.ask(
                "[yellow]Choose an editing action[/yellow]",
                choices=choices,
                default="6",
            )

            desc, handler = edit_options.get(choice)

            if handler:
                try:
                    handler()  # Call the corresponding handler method
                    # HTML is generated within _handle_patch_result after successful saves
                except Exception:
                    console.print(
                        f"[bold red]An error occurred during the '{desc}' action:[/bold red]"
                    )
                    console.print_exception(show_locals=False, word_wrap=True)
                    console.print("[yellow]Returning to editing menu.[/yellow]")
            elif choice == "7":  # Finish Editing
                console.print("\nFinishing editing process.")
                return True
            else:
                console.print(
                    "[red]Invalid choice.[/red]"
                )  # Should not happen with Prompt

        # Should not be reached normally
        return True

    # --- HTML Generation ---
    def _generate_html_output(self, filename="latest.html"):
        """Generates a simple HTML file from the book_root, including word counts."""
        if self.book_root is None or not self.book_dir:
            console.print(
                f"[red]Cannot generate HTML ('{filename}'), book data or directory missing.[/red]"
            )
            return

        filepath = self.book_dir / filename
        console.print(f"Generating HTML output to: [cyan]{filepath.resolve()}[/cyan]")

        try:
            title = escape(self.book_root.findtext("title", "Untitled Book"))
            synopsis = escape(
                self.book_root.findtext("synopsis", "No synopsis available.")
            ).replace("\n", "<br>")
            # Include initial idea if present
            initial_idea_raw = self.book_root.findtext("initial_idea", "")
            initial_idea_html = ""
            if initial_idea_raw:
                initial_idea_html = f"""
    <div class="initial-idea">
        <h2>Initial Idea/Description</h2>
        <p>{escape(initial_idea_raw).replace(chr(10), "<br>")}</p>
    </div>
"""

            characters = self.book_root.findall(".//character")
            chapters_raw = self.book_root.findall(".//chapter")
            chapters = sorted(chapters_raw, key=lambda c: int(c.get("id", 0)))

            # Calculate word counts for HTML
            total_wc_html = 0
            chapter_wc_html = {}
            for chap in chapters:
                chap_id = chap.get("id")
                content = chap.find("content")
                chap_wc = 0
                if content is not None:
                    for para in content.findall(".//paragraph"):
                        chap_wc += _count_words(para.text)
                chapter_wc_html[chap_id] = chap_wc
                total_wc_html += chap_wc

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Georgia, serif; line-height: 1.7; padding: 20px 40px; max-width: 900px; margin: auto; background-color: #fdfdfd; color: #333; }}
        h1, h2, h3 {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1a1a1a; font-weight: 400;}}
        h1 {{ text-align: center; margin-bottom: 10px; font-size: 2.8em; border-bottom: 2px solid #eee; padding-bottom: 15px; font-weight: 300; }}
        .total-word-count {{ text-align: center; font-size: 0.9em; color: #555; margin-bottom: 30px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }}
        h2 {{ margin-top: 45px; border-bottom: 1px solid #eee; padding-bottom: 10px; font-size: 1.9em; color: #2a2a2a; }}
        h3 {{ margin-top: 35px; font-size: 1.5em; color: #444; }}
        .synopsis, .initial-idea, .characters, .chapter {{ margin-bottom: 40px; }}
        .synopsis p, .initial-idea p {{ font-style: italic; color: #555; }}
        .characters ul {{ list-style: none; padding: 0; }}
        .characters li {{ margin-bottom: 14px; border-left: 3px solid #ddd; padding-left: 12px; }}
        .characters b {{ color: #1a1a1a; font-weight: 600;}}
        .chapter-content p {{ margin-bottom: 1.3em; text-indent: 2.5em;}}
        .chapter-content p:first-of-type {{ text-indent: 0; }} /* No indent for first paragraph */
        .missing-content {{ font-style: italic; color: #999; text-align: center; padding: 20px; background-color: #f9f9f9; border-radius: 4px; }}
        hr {{ border: 0; height: 1px; background: #ddd; margin: 50px 0; }}
        .chapter-title {{ font-weight: 600; }} /* Slightly bolder chapter titles */
        .chapter-meta {{ font-size: 0.8em; color: #aaa; margin-left: 10px; font-family: monospace; }} /* Show chapter ID subtly */
        .chapter-word-count {{ font-weight: normal; color: #777; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="total-word-count">Total Word Count: {total_wc_html:,}</div>

    <div class="synopsis">
        <h2>Synopsis</h2>
        <p>{synopsis}</p>
    </div>
{initial_idea_html}
    <div class="characters">
        <h2>Characters</h2>
        {f"<ul>{''.join(f'<li><b>{escape(c.findtext("name", "N/A"))}</b> (ID: {escape(c.get("id", "N/A"))}): {escape(c.findtext("description", "N/A"))}</li>' for c in characters)}</ul>" if characters else '<p class="missing-content">No characters defined.</p>'}
    </div>

    <hr>

    <div class="chapters">
        <h2>Chapters</h2>
"""
            if chapters:
                for chap in chapters:
                    chap_id = escape(chap.get("id", "N/A"))
                    chap_num = escape(
                        chap.findtext("number", chap_id)
                    )  # Use ID as fallback for number
                    chap_title = escape(chap.findtext("title", "No Title"))
                    chap_wc_val = chapter_wc_html.get(chap.get("id"), 0)
                    html_content += f"""
    <div class="chapter" id="chapter-{chap_id}">
        <h3>
            <span class="chapter-title">Chapter {chap_num}: {chap_title}</span>
            <span class="chapter-meta">(ID: {chap_id} | Words: <span class="chapter-word-count">{chap_wc_val:,}</span>)</span>
        </h3>
"""
                    content = chap.find("content")
                    if content is not None:
                        paragraphs = content.findall(".//paragraph")
                        has_text_content = any(
                            p.text and p.text.strip() for p in paragraphs
                        )
                        if paragraphs and has_text_content:
                            html_content += '<div class="chapter-content">\n'
                            for para in paragraphs:
                                # Handle potential None text, strip, escape, replace newlines
                                para_text = escape((para.text or "").strip()).replace(
                                    "\n", "<br>"
                                )
                                if not para_text:
                                    continue  # Skip empty paragraphs in output
                                html_content += f"  <p>{para_text}</p>\n"
                            html_content += "</div>\n"
                        else:
                            html_content += '<p class="missing-content"><i>[Chapter content not generated or empty]</i></p>\n'
                    else:
                        html_content += '<p class="missing-content"><i>[Chapter content structure missing]</i></p>\n'
                    html_content += "</div>\n"  # Close chapter div
            else:
                html_content += '<p class="missing-content">No chapters defined.</p>'

            html_content += """
    </div>

</body>
</html>"""

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            console.print(f"[green]HTML file '{filename}' saved successfully.[/green]")

        except Exception as e:
            console.print(
                f"[bold red]Error generating HTML file {filepath}: {e}[/bold red]"
            )
            console.print_exception(show_locals=False)

    # --- Main Execution Flow ---
    def run(self):
        """Runs the main interactive novel writing process."""
        console.print(
            Panel(
                f"📚 Welcome to the Interactive Novel Writer! (Project: {self.book_dir.name if self.book_dir else 'New Project'}) 📚",
                style="bold green",
            )
        )

        # Ensure client is initialized
        if self.client is None:
            console.print(
                "[bold red]Gemini client failed to initialize. Cannot continue.[/bold red]"
            )
            return

        # Ensure book root is loaded/initialized
        if self.book_root is None:
            console.print(
                "[bold red]Book data failed to load or initialize. Cannot continue.[/bold red]"
            )
            return

        # --- Outline Step ---
        outline_step_run = False
        outline_success = False
        chapters_element = self.book_root.find(".//chapters")
        has_chapters = chapters_element is not None and bool(
            chapters_element.findall(".//chapter")
        )
        has_summaries = has_chapters and any(
            c.findtext("summary", "").strip()
            for c in chapters_element.findall(".//chapter")
        )
        has_characters = self.book_root.find(".//characters") is not None and bool(
            self.book_root.findall(".//character")
        )
        has_full_outline = has_chapters and has_summaries and has_characters

        if not has_full_outline:
            console.print(
                "[yellow]Outline seems incomplete (missing chapters, summaries, or characters). Running Outline Generation.[/yellow]"
            )
            outline_success = self.generate_outline()
            outline_step_run = True
            if not outline_success:
                console.print(
                    "[bold red]Outline generation failed. Cannot proceed reliably.[/bold red]"
                )
                return  # Exit if outline failed critically
        else:
            console.print("[green]Existing full outline loaded.[/green]")
            self._display_summary()
            if Confirm.ask(
                "[yellow]Do you want to run the Outline Generation step again? (Replaces existing characters/summaries)[/yellow]",
                default=False,
            ):
                outline_success = self.generate_outline()
                outline_step_run = True
                if not outline_success:
                    console.print("[bold red]Outline re-generation failed.[/bold red]")
                    # Ask if user wants to proceed with old outline?
                    if not Confirm.ask(
                        "[yellow]Proceed with the previously loaded outline anyway? [/yellow]",
                        default=True,
                    ):
                        return
                else:  # Outline re-gen succeeded
                    pass  # Proceed

        # Re-check if outline exists after potential generation
        chapters_element = (
            self.book_root.find(".//chapters") if self.book_root is not None else None
        )
        # Also recheck characters as outline generation might add/remove them
        has_characters = self.book_root.find(".//characters") is not None and bool(
            self.book_root.findall(".//character")
        )
        has_outline = (
            chapters_element is not None
            and bool(chapters_element.findall(".//chapter"))
            and has_characters
        )

        if has_outline:
            # Ask to proceed only if outline step was successful OR wasn't run because it existed
            if (
                outline_success or not outline_step_run
            ):  # Check if successful OR wasn't run because it existed
                if not Confirm.ask(
                    "\n[yellow]Outline complete. Proceed to Chapter Content Generation? [/yellow]",
                    default=True,
                ):
                    console.print("Exiting as requested before chapter generation.")
                    return
        elif (
            outline_step_run
        ):  # Outline step ran but failed to produce chapters/characters
            console.print(
                "[red]Outline generation was run but failed to produce a valid outline. Cannot proceed.[/red]"
            )
            return
        else:  # No outline and it wasn't run - should have been caught by initial check or load error
            console.print(
                "[red]No outline available and outline generation did not run or failed. Cannot proceed.[/red]"
            )
            return

        # --- Chapter Generation Step ---
        chapter_gen_run_or_skipped = (
            False  # Track if this step was attempted or explicitly skipped
        )
        if has_outline:
            # Check if *any* chapter needs content (using precise definition)
            needs_generation = False
            for c in chapters_element.findall(".//chapter"):
                content = c.find("content")
                if content is None:
                    needs_generation = True
                    break
                paragraphs = content.findall(".//paragraph")
                if not paragraphs or not any(
                    p.text and p.text.strip() for p in paragraphs
                ):
                    needs_generation = True
                    break

            run_generation = False
            if needs_generation:
                console.print(
                    "\n[yellow]Some chapters are missing content. Proceeding to Chapter Generation.[/yellow]"
                )
                run_generation = True
            elif (
                outline_step_run
                and outline_success  # Only ask if outline was just successfully (re)generated
            ):
                if Confirm.ask(
                    "\n[green]All chapters have empty content as expected after outline generation.[/green] [yellow]Run Chapter Generation now? [/yellow]",
                    default=True,
                ):
                    run_generation = True
            else:  # Outline exists and doesn't strictly need generation
                console.print(
                    "\n[green]All chapters appear to have content.[/green]"
                )  # Simplified message
                if Confirm.ask(
                    "[yellow]Do you want to run Chapter Generation anyway (may overwrite based on strategy)? [/yellow]",
                    default=False,
                ):
                    run_generation = True

            if run_generation:
                gen_result = (
                    self.generate_chapters()
                )  # Returns True if completed loop, False if aborted early
                chapter_gen_run_or_skipped = (
                    True  # Mark as run (even if some batches failed)
                )
                if not gen_result:
                    console.print(
                        "[yellow]Chapter generation phase was aborted.[/yellow]"
                    )
                    if not Confirm.ask(
                        "[yellow]Continue to Editing phase despite aborted chapter generation? [/yellow]",
                        default=False,
                    ):
                        console.print("Exiting.")
                        return
                # If gen_result is True, it means the loop finished (might have had errors but wasn't aborted by user)
                console.print("[cyan]Chapter generation phase finished.[/cyan]")
            else:
                console.print("[cyan]Skipping Chapter Generation step.[/cyan]")
                chapter_gen_run_or_skipped = True  # Skipped successfully

        else:
            # This path should ideally not be reached if outline checks above work
            console.print(
                "[yellow]Skipping Chapter Generation as no valid outline exists.[/yellow]"
            )

        # --- Proceed Confirmation 2 ---
        if has_outline:
            if (
                chapter_gen_run_or_skipped
            ):  # Check if the step was run or explicitly skipped
                if not Confirm.ask(
                    "\n[yellow]Chapter Generation phase complete (or skipped). Proceed to Editing? [/yellow]",
                    default=True,
                ):
                    console.print("Exiting as requested before editing.")
                    return
            # If chapter gen was aborted and user chose not to continue, we already exited

        # --- Editing Step ---
        editing_run_or_skipped = False
        if has_outline:  # Only edit if there's something to edit
            console.print("\n[yellow]Proceeding to the Editing phase.[/yellow]")
            editing_result = self.edit_book()  # Returns True if finished normally
            editing_run_or_skipped = True  # Mark as run
            if not editing_result:  # Assuming False means aborted or major issue
                console.print(
                    "[yellow]Editing phase was aborted or encountered issues.[/yellow]"
                )
                # Editing loop handles its own exit, so just proceed to final save prompt
                pass
            else:
                console.print("[green]Editing phase completed.[/green]")
        else:
            console.print("[yellow]Skipping Editing as no outline exists.[/yellow]")
            # Mark as skipped only if we didn't intend to run it
            if not has_outline:
                editing_run_or_skipped = True

        # --- Proceed Confirmation 3 ---
        if self.book_root is not None and self.book_dir is not None:
            # Always ask for final save unless editing was totally skipped because no outline existed
            if editing_run_or_skipped or chapter_gen_run_or_skipped or outline_step_run:
                if not Confirm.ask(
                    "\n[yellow]Editing complete (or skipped). Proceed to Final Save and HTML Generation? [/yellow]",
                    default=True,
                ):
                    console.print("Skipping final save and generation as requested.")
                    return
            # If editing failed and user opted out, we already exited during edit loop
            # If generation failed and user opted out, we exited there
            # If outline failed, we exited there

        # --- Final Save & HTML Output ---
        if self.book_root is not None and self.book_dir is not None:
            console.print("[cyan]Performing final save...[/cyan]")
            final_xml_filename = "final.xml"
            save_final_xml_ok = self._save_book_state(final_xml_filename)
            final_html_filename = (
                "final.html"  # Overwrite final.html with the actual final state
            )
            self._generate_html_output(
                final_html_filename
            )  # This uses the current self.book_root

            if save_final_xml_ok:
                console.print(
                    Panel(
                        f"[bold green]🎉 Novel writing process complete! Final version saved as {final_xml_filename} and {final_html_filename}. 🎉[/bold green]",
                        border_style="green",
                    )
                )
                console.print(
                    f"All files are located in: [cyan]{self.book_dir.resolve()}[/cyan]"
                )
            else:
                console.print(
                    f"[bold red]Could not save final XML state ({final_xml_filename}), but HTML ({final_html_filename}) was generated from current state.[/bold red]"
                )

        else:
            console.print(
                "[bold red]Could not perform final save as book data or directory is missing.[/bold red]"
            )


# --- Main Execution Guard ---
if __name__ == "__main__":
    console.print(Panel("📚 Novel Writer Setup 📚", style="blue"))

    # --- Command Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Interactive Novel Writer using LLMs.")
    parser.add_argument(
        "--resume",
        metavar="FOLDER_PATH",
        help="Path to an existing book project folder (e.g., '20231027-my-novel-abc123ef') to resume.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        metavar="FILE_PATH",
        help="Path to a text file containing the initial book idea/prompt (used only when starting a new book).",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # --- Handle Argument Logic ---
    resume_folder_path = args.resume
    initial_prompt_file = args.prompt

    # If --resume is provided, --prompt is ignored (prompt is only for new books)
    if resume_folder_path and initial_prompt_file:
        console.print(
            "[yellow]Warning: --prompt argument is ignored when using --resume.[/yellow]"
        )
        initial_prompt_file = None  # Ensure it's not passed to __init__

    # If neither --resume nor --prompt is given, the default behavior is new book + interactive prompt
    if not resume_folder_path and not initial_prompt_file:
        console.print("Starting a new book project (interactive prompt).")
        # No need to explicitly set anything, __init__ defaults handle this

    elif not resume_folder_path and initial_prompt_file:
        console.print(
            f"Starting a new book project using prompt from file: [cyan]{initial_prompt_file}[/cyan]"
        )
        # Pass the file path to __init__

    elif resume_folder_path:
        # Validate if the resume path actually exists before proceeding
        resume_path_obj = Path(resume_folder_path)
        if not resume_path_obj.is_dir():
            console.print(
                f"[bold red]Error: Resume directory not found: '{resume_folder_path}'[/bold red]"
            )
            exit(1)  # Exit if resume path is invalid
        console.print(
            f"Attempting to resume book project from folder: [cyan]{resume_folder_path}[/cyan]"
        )
        # Pass the folder path to __init__

    # --- Instantiate and Run ---
    try:
        # Pass the parsed arguments to the NovelWriter constructor
        writer = NovelWriter(
            resume_folder_name=resume_folder_path,
            initial_prompt_file=initial_prompt_file,
        )
        writer.run()
    except FileNotFoundError as fnf_error:
        # This might catch errors during loading within __init__ if path was valid initially but files are missing
        console.print(
            f"[bold red]File Not Found Error during setup/loading: {fnf_error}[/bold red]"
        )
    except ValueError as ve:
        # Catch API key missing error or other init value errors
        console.print(f"[bold red]Initialization Error: {ve}[/bold red]")
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Operation interrupted by user. Exiting.[/bold yellow]"
        )
    except Exception as main_exception:
        console.print(
            "[bold red]An unexpected critical error occurred during execution:[/bold red]"
        )
        console.print_exception(
            show_locals=False, word_wrap=True, exc_info=main_exception
        )  # Rich traceback
