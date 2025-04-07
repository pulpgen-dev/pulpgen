# -*- coding: utf-8 -*-
import google.generativeai as genai
from google.generativeai import types

# Import specific exceptions for better handling
from google.api_core import exceptions as google_api_exceptions

import xml.etree.ElementTree as ET
from xml.dom import minidom  # For pretty printing XML
import os

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

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv
from html import escape  # For HTML generation

# --- Configuration ---
WRITING_MODEL_NAME = "gemini-2.5-pro-exp-03-25"
WRITING_MODEL_CONFIG = types.GenerationConfig(
    temperature=1,
    max_output_tokens=65536,  # Gemini 1.5 Pro max output
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
    def __init__(self, resume_folder_name=None):
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

        self._init_client()  # Init client early

        if resume_folder_name:
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

            console.print(f"Resuming work on book: [cyan]{self.book_dir.name}[/cyan]")
            self._load_book_state()  # Load state using self.book_dir

        else:
            # --- New Book Creation ---
            console.print(Panel("Starting a New Novel Project", style="bold green"))

            # --- Feature: File Input for Idea ---
            idea_input = Prompt.ask(
                "[yellow]Enter your book idea/description, OR provide the path to a text file containing it[/yellow]"
            )
            idea_path = Path(idea_input)
            idea = ""
            if idea_path.is_file():
                try:
                    idea = idea_path.read_text(encoding="utf-8")
                    console.print(
                        f"[green]✓ Read book idea from file: {idea_path}[/green]"
                    )
                    console.print(f"[dim]Content preview: {idea[:200]}...[/dim]")
                except Exception as e:
                    console.print(
                        f"[bold red]Error reading file '{idea_path}': {e}. Please enter idea manually.[/bold red]"
                    )
                    idea = Prompt.ask(
                        "[yellow]Enter your book idea/description[/yellow]"
                    )
            else:
                if len(idea_input) > 200:  # Arbitrary length check
                    console.print(
                        "[yellow]Input doesn't seem to be a file path. Using the entered text directly.[/yellow]"
                    )
                idea = idea_input  # Use the input as text

            if not idea:
                console.print(
                    "[red]No valid idea provided. Using a generic placeholder.[/red]"
                )
                idea = "A default book idea for generating a title."

            # 2. Generate initial title (and maybe synopsis) quickly
            temp_uuid = str(uuid.uuid4())[:8]
            initial_outline_data = self._generate_minimal_outline(
                idea  # Use the idea from file or text
            )

            book_title = initial_outline_data.get(
                "title", f"Untitled Novel {temp_uuid}"
            )
            self.book_title_slug = slugify(book_title) if book_title else "untitled"

            # 3. Construct Folder Name
            today_date_str = datetime.now().strftime(DATE_FORMAT_FOR_FOLDER)
            self.book_id = temp_uuid  # Keep UUID as the core unique ID
            folder_name = f"{today_date_str}-{self.book_title_slug}-{self.book_id}"
            self.book_dir = Path(folder_name)

            # 4. Create Directory
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

            # 5. Initialize Book Root (using data from minimal outline if available)
            self.book_root = ET.Element("book")
            ET.SubElement(self.book_root, "title").text = book_title
            ET.SubElement(self.book_root, "synopsis").text = initial_outline_data.get(
                "synopsis", "Synopsis not yet generated."
            )
            # Add the full idea/description read from file/input into a dedicated tag for reference?
            ET.SubElement(self.book_root, "initial_idea").text = idea
            ET.SubElement(self.book_root, "characters")
            ET.SubElement(self.book_root, "chapters")

            # 6. Save the initial (often empty chapters) state as outline.xml
            self._save_book_state("outline.xml")  # Save this basic structure first
            self.patch_counter = 0  # Explicitly set for new book

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
        # Ensure patches are sorted numerically correctly
        patch_files = sorted(
            [
                p
                for p in self.book_dir.glob("patch-*.xml")
                if p.stem.split("-")[-1].isdigit()
            ],
            key=lambda p: int(p.stem.split("-")[-1]),
        )

        latest_file = outline_file
        if patch_files:
            latest_file = patch_files[-1]
            try:
                self.patch_counter = int(latest_file.stem.split("-")[-1])
            except (ValueError, IndexError):
                console.print(
                    f"[yellow]Warning: Could not parse patch number from {latest_file.name}. Estimating counter.[/yellow]"
                )
                self.patch_counter = len(patch_files)  # Estimate based on count
        else:
            self.patch_counter = 0  # No patches yet

        if not latest_file.exists():
            if outline_file.exists():
                latest_file = outline_file
                self.patch_counter = 0
                console.print(
                    f"[yellow]Warning: No patch files found, loading {outline_file.name}.[/yellow]"
                )
            else:
                console.print(
                    f"[yellow]Warning: No state file found ({latest_file.name}). Book may be empty or corrupted.[/yellow]"
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
            # Ensure patch counter reflects the loaded file
            if latest_file.name.startswith("patch-"):
                try:
                    self.patch_counter = int(latest_file.stem.split("-")[-1])
                except:
                    pass  # Keep previous estimation if parsing fails here too

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
            return
        if not self.book_dir:
            console.print(
                "[bold red]Error: Cannot save state, book directory not set.[/bold red]"
            )
            return

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
        except Exception as e:
            console.print(
                f"[bold red]Error saving book state to {filepath}: {e}[/bold red]"
            )

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
                                chunk.prompt_feedback
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
                                msg = f"Content generation blocked during streaming.\nReason: {reason}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                                console.print(
                                    f"\n[bold red]API Safety Error: {msg}[/bold red]"
                                )
                                # This is likely unrecoverable by simple retry, treat as fatal for this attempt
                                raise types.BlockedPromptException(
                                    msg
                                )  # Re-raise for outer catch

                            # Append text safely
                            try:
                                chunk_text = chunk.text
                                print(chunk_text, end="", flush=True)
                                full_response += chunk_text
                            except ValueError as ve:
                                # This might happen if the stream is abruptly terminated or blocked without explicit reason yet
                                console.print(
                                    f"\n[yellow]Warning: Received potentially invalid chunk data: {ve}[/yellow]"
                                )
                                # Check overall response feedback *after* the loop if needed, but blocking check above is better
                                continue  # Try to get next chunk

                        print()  # Newline after streaming finishes
                        response_completed_normally = True  # Mark as finished ok

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
                        response.prompt_feedback
                        and response.prompt_feedback.block_reason
                    ):
                        reason = response.prompt_feedback.block_reason
                        ratings = response.prompt_feedback.safety_ratings
                        ratings_str = "\n".join(
                            [
                                f"  - {r.category.name}: {r.probability.name}"
                                for r in ratings
                            ]
                        )
                        msg = f"Content generation blocked.\nReason: {reason}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                        console.print(f"[bold red]API Safety Error: {msg}[/bold red]")
                        raise types.BlockedPromptException(msg)  # Raise for outer catch

                    # Extract text safely
                    try:
                        full_response = response.text
                        console.print(
                            f"[cyan]>>> Gemini Response ({task_description}):[/cyan]"
                        )
                        console.print(
                            f"[dim]{full_response[:1000]}{'...' if len(full_response) > 1000 else ''}[/dim]"
                        )
                    except ValueError as ve:
                        # Handle case where response might be blocked but didn't have block_reason set explicitly?
                        console.print(
                            f"[bold red]Error retrieving text from non-streaming response: {ve}[/bold red]"
                        )
                        console.print(f"[dim]Response object: {response}[/dim]")
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
                types.StopCandidateException,
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
                    # Ensure new paragraphs have IDs
                    for i, para in enumerate(new_content.findall(".//paragraph")):
                        if para.get("id") is None:
                            para.set("id", str(i + 1))
                    target_chapter.append(
                        copy.deepcopy(new_content)
                    )  # Deepcopy sub-elements being added
                    console.print(
                        f"[green]Applied full content patch to Chapter {chapter_id}.[/green]"
                    )
                    chapters_patched.append(chapter_id)
                    applied_changes = True

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
                        if para_id is None:
                            console.print(
                                f"[yellow]Warning: Paragraph patch for Chapter {chapter_id} missing 'id'. Skipping.[/yellow]"
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
        """Displays a summary of the current book state."""
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

        console.print(
            Panel(
                f"Book Summary: [cyan]{title}[/cyan]",
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
        if characters:  # Pythonic truthiness check
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
        if chapters:  # Pythonic truthiness check
            chap_table = Table(title="Chapters")
            chap_table.add_column("ID", style="dim")
            chap_table.add_column("Number")
            chap_table.add_column("Title")
            chap_table.add_column("Summary Status")
            chap_table.add_column("Content Status")

            # Sort chapters by ID numerically before display
            sorted_chapters = sorted(chapters, key=lambda c: int(c.get("id", 0)))

            for chap in sorted_chapters:
                chap_id = chap.get("id", "N/A")
                num = chap.findtext("number", "N/A")
                chap_title = chap.findtext("title", "N/A")
                summary = chap.findtext("summary", "").strip()
                content = chap.find("content")
                paragraphs = (
                    content.findall(".//paragraph") if content is not None else []
                )
                has_content = bool(paragraphs)  # More direct check

                summary_status = "[green]✓[/green]" if summary else "[red]Missing[/red]"
                content_status = (
                    f"[green]✓ {len(paragraphs)} paras[/green]"
                    if has_content
                    else "[red]Empty[/red]"
                )
                if chap_id in self.chapters_generated_in_session:
                    content_status += " [cyan](new)[/cyan]"  # Mark newly generated

                chap_table.add_row(
                    chap_id, num, chap_title, summary_status, content_status
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

        # Check if chapters element exists and has children
        chapters_element = self.book_root.find(".//chapters")
        has_chapters = chapters_element is not None and len(chapters_element) > 0
        has_summaries = (
            chapters_element is not None
            and chapters_element.find(".//summary") is not None
        )
        if has_chapters and has_summaries:
            if not Confirm.ask(
                "[yellow]Outline (chapters with summaries, characters) seems to exist. Regenerate? (This will replace existing structure)[/yellow]",
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
                            ET.SubElement(chapter, "content")  # Add empty content tag
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
                    ET.SubElement(new_book_root, "characters")
                else:
                    used_char_ids = set()
                    for char in chars_elem.findall("character"):
                        char_id = char.get("id")
                        name_elem = char.find("name")
                        name = (
                            name_elem.text
                            if name_elem is not None
                            else f"Character_{uuid.uuid4().hex[:4]}"
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

                        if name_elem is None:
                            ET.SubElement(char, "name").text = "Unnamed Character"
                        if char.find("description") is None:
                            ET.SubElement(
                                char, "description"
                            ).text = "Description needed."

                # Ensure Title/Synopsis/Idea exist (copy from original if LLM removed them)
                for tag in ["title", "synopsis", "initial_idea"]:
                    if new_book_root.find(tag) is None:
                        original_elem = self.book_root.find(tag)
                        if original_elem is not None:
                            console.print(
                                f"[yellow]Warning: LLM removed <{tag}>. Restoring from original.[/yellow]"
                            )
                            # Insert at appropriate position (simplistic: near top)
                            new_book_root.insert(0, copy.deepcopy(original_elem))
                        else:
                            # Add placeholder if it was missing originally too
                            ET.SubElement(
                                new_book_root, tag
                            ).text = f"{tag.replace('_', ' ').title()} needed."

                # --- Commit Changes ---
                if validation_passed:
                    self.book_root = (
                        new_book_root  # Replace current root with the validated one
                    )
                    self._save_book_state(
                        "outline.xml"
                    )  # Save as the definitive outline
                    console.print(
                        "[green]Full outline generated/updated and saved as outline.xml[/green]"
                    )
                    # Update internal state if title changed
                    self.book_title_slug = slugify(
                        self.book_root.findtext("title", "untitled")
                    )
                    self._display_summary()
                    self.patch_counter = (
                        0  # Reset patch counter as this is the new base
                    )
                    self.chapters_generated_in_session.clear()
                    return True  # Indicate success
                else:
                    console.print(
                        "[bold red]Outline generation completed, but some validation issues occurred (check warnings). Outline saved, but review recommended.[/bold red]"
                    )
                    # Still save the potentially problematic outline, but warn user
                    self.book_root = new_book_root
                    self._save_book_state("outline.xml")
                    self.patch_counter = 0
                    self.chapters_generated_in_session.clear()
                    return True  # Saved, but with issues

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
            # Check if content tag exists and has any paragraph children
            if content is not None and content.findall(
                ".//paragraph"
            ):  # Checks for list emptiness
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
                    index = min(
                        total_chapters - 1,
                        max(0, int(round(i * total_chapters / steps))),
                    )
                    # Only add if it's different from start/end
                    if index != 0 and index != total_chapters - 1:
                        indices_to_pick.add(index)
            elif total_chapters > 2:  # Add middle if possible
                indices_to_pick.add(total_chapters // 2)

            # Ensure we don't exceed batch size, prioritize unique indices
            potential_indices = sorted(list(indices_to_pick))
            actual_indices = potential_indices[:batch_size]

            potential_chapters = [
                all_chapters[i] for i in actual_indices if 0 <= i < total_chapters
            ]
            selected_chapters = [
                ch for ch in potential_chapters if ch in chapters_needing_content
            ]  # Final check needed

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
            if current_gap:
                gaps.append(current_gap)

            # Simple strategy: flatten gaps and take first N
            chapters_from_gaps = [chap for gap in gaps for chap in gap]
            selected_chapters = chapters_from_gaps[:batch_size]

        return selected_chapters

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
            chapters_to_generate = self._select_chapters_to_generate(batch_size=5)

            if not chapters_to_generate:
                # Double check if any are *really* missing content
                truly_needs_generation = any(
                    c.find("content") is None or not c.findall(".//content/paragraph")
                    for c in self.book_root.findall(".//chapter")
                )
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

            prompt = f"""
You are a novelist continuing the story based on the full book context provided below.
Your task is to write the full prose content for the following {len(chapters_to_generate)} chapters:
{chapter_details_prompt}

Guidelines:
- Write approximately 3000-5000 words *per chapter*. Adjust length based on the chapter's summary and importance within the narrative arc.
- Maintain consistency with the established plot, characters (personalities, motivations, relationships), tone, and writing style evident in the rest of the book context (including summaries of unwritten chapters and content of written ones).
- Ensure the events of these chapters align with their summaries and logically connect preceding/succeeding chapters. Pay attention to how these chapters function together within the narrative arc.
- For EACH chapter you generate content for, structure it within `<content>` tags, divided into paragraphs using `<paragraph>` tags. Each paragraph tag MUST have a unique sequential `id` attribute within that chapter (e.g., `<paragraph id="1">...</paragraph>`, `<paragraph id="2">...</paragraph>`, etc.). Start paragraph IDs from "1" for each chapter.
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

Generate the `<patch>` XML containing the content for the requested chapters now. Ensure all specified chapters are included in the response patch with correct IDs and paragraph structure.
"""

            response_patch_xml_str = self._get_llm_response(
                prompt, f"Writing Chapters {chapter_ids_str}"
            )

            # Check if response was obtained (handles None return from _get_llm_response)
            if response_patch_xml_str is None:
                console.print(
                    f"[bold red]Failed to get response from LLM for batch starting with Chapter {chapters_to_generate[0].get('id')}.[/bold red]"
                )
                # Ask to continue (already handled within _get_llm_response, but double-check needed?)
                # No, _get_llm_response handles the confirmation to retry/abort. If it returns None, it failed definitively.
                if not Confirm.ask(
                    "[yellow]Failed to get content for this batch. Continue trying the NEXT batch? [/yellow]",
                    default=True,
                ):
                    console.print("Aborting chapter generation.")
                    return False  # Indicate failure to continue
                else:
                    continue  # Skip apply/save for this failed batch, try next selection

            # --- Apply the patch ---
            # Patch application logic moved inside the `if response_patch_xml_str:` block
            if self._apply_patch(response_patch_xml_str):
                # Use get_next_patch_number *after* successful apply to find the next available number
                self.patch_counter = get_next_patch_number(self.book_dir)
                patch_filename = f"patch-{self.patch_counter:02d}.xml"  # Use the calculated next number for saving
                self._save_book_state(patch_filename)
                console.print(
                    f"[green]Successfully generated and applied content patch for chapters {chapter_ids_str}. Saved state.[/green]"
                )
                self._display_summary()  # Display summary showing new chapters
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

    # --- Step 3: Editing ---
    def edit_book(self):
        """Guides the user through the editing process with LLM suggestions."""
        console.print(Panel("Step 3: Editing the Novel", style="bold blue"))
        if self.book_root is None:
            console.print("[red]Cannot edit, book data missing.[/red]")
            return False  # Indicate failure

        while True:
            self._display_summary()
            console.print("\n[bold cyan]Editing Options:[/bold cyan]")
            console.print(
                "1. Ask LLM to suggest edits (provide general or specific instructions)"
            )
            console.print("2. Finish editing")

            choice = Prompt.ask(
                "[yellow]Choose an option[/yellow]", choices=["1", "2"], default="1"
            )

            if choice == "1":
                instructions = Prompt.ask(
                    "[yellow]Enter editing instructions (e.g., 'Improve pacing in chapters 5-7', 'Strengthen character X's motivation', 'Check for plot inconsistencies', 'Make the dialogue more natural') [/yellow]",
                    default="Review the entire book for consistency, pacing, character arcs, and suggest improvements.",
                )

                current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

                prompt = f"""
You are an expert editor reviewing the novel provided below in XML format.
Your task is to analyze the text based on the user's instructions and suggest specific changes.

User's Editing Instructions: {instructions}

Guidelines for Suggestions:
- Analyze the entire book context provided.
- Identify areas for improvement based on the instructions (e.g., plot holes, inconsistent characterization, weak descriptions, pacing issues, dialogue problems).
- Propose concrete changes.
- Format your suggested changes STRICTLY as an XML `<patch>` structure.
- A patch can modify chapters or paragraphs.
    - To replace the ENTIRE content of a chapter:
      `<patch><chapter id="chap_id"><content><paragraph id="1">New para 1</paragraph>...</content></chapter></patch>` (Ensure new paragraphs have sequential IDs starting from 1).
    - To replace specific paragraphs within a chapter:
      `<patch><chapter id="chap_id"><content-patch><paragraph id="para_id">New text for this para</paragraph>...</content-patch></chapter></patch>`
    - You can include multiple `<chapter>` patches within a single `<patch>` tag if suggesting changes across chapters.
    - Ensure paragraph IDs referenced in `<content-patch>` exist in the original chapter.
- If suggesting changes to Title, Synopsis, or Characters, use the appropriate top-level tags within the `<patch>` tag (e.g., `<patch><title>New Title</title>...</patch>`). Characters should generally be replaced as a whole list for simplicity: `<patch><characters><character id="newId1">...</character>...</characters></patch>`.
- Be specific and justify your changes briefly if possible using XML comments within the patch: `<!-- Suggestion: Rephrased for clarity -->`.
- If no significant changes are needed based on the instructions, output an empty patch like `<patch />` or a patch with only comments: `<patch><!-- No major changes suggested --></patch>`

Full Book Context:
```xml
{current_book_xml_str}
```

Generate the `<patch>` XML containing your suggested edits now. Output ONLY the patch XML.
"""

                suggested_patch_xml_str = self._get_llm_response(
                    prompt, "Generating edit suggestions"
                )

                # Check if response was obtained
                if suggested_patch_xml_str is None:
                    console.print(
                        "[bold red]Failed to get edit suggestions from the LLM.[/bold red]"
                    )
                    # Ask to try again?
                    if not Confirm.ask(
                        "[yellow]Try asking for edits again? [/yellow]", default=True
                    ):
                        continue  # Let user try again from options
                    else:
                        break  # Go back to editing options menu

                # --- Process Suggestion ---
                console.print(
                    Panel(
                        "[bold cyan]Suggested Edits (Patch XML):[/bold cyan]",
                        border_style="magenta",
                    )
                )
                syntax = Syntax(
                    suggested_patch_xml_str,
                    "xml",
                    theme="default",
                    line_numbers=True,
                )
                console.print(syntax)

                is_empty_or_comment_patch = False
                try:
                    # Use parse_xml_string to handle cleaning before checking emptiness
                    patch_elem = parse_xml_string(
                        suggested_patch_xml_str, expected_root_tag="patch"
                    )
                    # Check if None, or has no child elements, or only comment children
                    if patch_elem is None:
                        # Parsing failed, _apply_patch will handle error message
                        pass
                    elif not list(patch_elem):  # No children
                        is_empty_or_comment_patch = True
                    else:
                        # Check if all children are comments
                        all_comments = all(
                            isinstance(child, ET.Comment) for child in patch_elem
                        )
                        if all_comments:
                            is_empty_or_comment_patch = True

                except Exception:
                    # Error during the check itself, let apply_patch handle it formally
                    pass

                if is_empty_or_comment_patch:
                    console.print(
                        "[cyan]LLM suggested no specific changes or only provided comments.[/cyan]"
                    )
                    # Ask if user wants to proceed or try different instructions
                    if not Confirm.ask(
                        "\n[yellow]Try providing different editing instructions? [/yellow]",
                        default=True,
                    ):
                        continue  # Go back to editing options menu
                elif Confirm.ask(
                    "\n[yellow]Do you want to apply these suggested changes? [/yellow]",
                    default=True,
                ):
                    # Get next patch number *before* applying
                    next_patch_num = get_next_patch_number(self.book_dir)
                    if self._apply_patch(suggested_patch_xml_str):
                        self.patch_counter = next_patch_num  # Use pre-calculated number
                        patch_filename = f"patch-{self.patch_counter:02d}.xml"
                        self._save_book_state(patch_filename)
                        console.print(
                            "[green]Patch applied successfully and state saved.[/green]"
                        )
                        # Loop continues, will display updated summary next
                    else:
                        console.print(
                            "[bold red]Failed to apply the patch. State not saved for this edit.[/bold red]"
                        )
                        # Ask to continue editing?
                        if not Confirm.ask(
                            "[yellow]Continue editing despite failed patch apply? [/yellow]",
                            default=True,
                        ):
                            break  # Exit editing loop

                else:  # User chose not to apply
                    console.print("Suggested patch discarded.")
                    # Ask to continue editing?
                    if not Confirm.ask(
                        "\n[yellow]Perform another editing action? [/yellow]",
                        default=True,
                    ):
                        break  # Exit editing loop

            elif choice == "2":
                console.print("\nFinishing editing process.")
                return True  # Indicate editing finished successfully

        # Should not be reached normally
        return True

    # --- HTML Generation ---
    def _generate_html_output(self, filename="final.html"):
        """Generates a simple HTML file from the book_root."""
        if self.book_root is None or not self.book_dir:
            console.print(
                "[red]Cannot generate HTML, book data or directory missing.[/red]"
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

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Georgia, serif; line-height: 1.7; padding: 20px 40px; max-width: 900px; margin: auto; background-color: #fdfdfd; color: #333; }}
        h1, h2, h3 {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1a1a1a; font-weight: 400;}}
        h1 {{ text-align: center; margin-bottom: 30px; font-size: 2.8em; border-bottom: 2px solid #eee; padding-bottom: 15px; font-weight: 300; }}
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
        .chapter-id {{ font-size: 0.8em; color: #aaa; margin-left: 10px; font-family: monospace; }} /* Show chapter ID subtly */
    </style>
</head>
<body>
    <h1>{title}</h1>

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
                    html_content += f'<h3><span class="chapter-title">Chapter {chap_num}: {chap_title}</span><span class="chapter-id">(ID: {chap_id})</span></h3>\n'
                    content = chap.find("content")
                    if content is not None:
                        paragraphs = content.findall(".//paragraph")
                        if paragraphs:
                            html_content += '<div class="chapter-content">\n'
                            for para in paragraphs:
                                # Handle potential None text, strip, escape, replace newlines
                                para_text = escape((para.text or "").strip()).replace(
                                    "\n", "<br>"
                                )
                                if not para_text:
                                    para_text = "&nbsp;"  # Use non-breaking space for visually empty paras
                                html_content += f"  <p>{para_text}</p>\n"
                            html_content += "</div>\n"
                        else:
                            html_content += '<p class="missing-content"><i>[Chapter content not generated]</i></p>\n'
                    else:
                        html_content += '<p class="missing-content"><i>[Chapter content structure missing]</i></p>\n'
            else:
                html_content += '<p class="missing-content">No chapters defined.</p>'

            html_content += """
    </div>

</body>
</html>"""

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            console.print("[green]HTML file saved successfully.[/green]")

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
        has_full_outline = (
            chapters_element is not None
            and bool(chapters_element.findall(".//chapter"))  # Check if chapters exist
            and chapters_element.find(".//summary")
            is not None  # Check if at least one summary exists
        )

        if not has_full_outline:
            console.print(
                "[yellow]Outline seems incomplete (missing chapters or chapter summaries). Running Outline Generation.[/yellow]"
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

        # Re-check if outline exists after potential generation
        chapters_element = (
            self.book_root.find(".//chapters") if self.book_root is not None else None
        )
        has_outline = chapters_element is not None and bool(
            chapters_element.findall(".//chapter")
        )

        if has_outline:
            if not Confirm.ask(
                "\n[yellow]Outline complete. Proceed to Chapter Content Generation? [/yellow]",
                default=True,
            ):
                console.print("Exiting as requested before chapter generation.")
                return
        elif not has_outline and outline_step_run:
            console.print(
                "[red]Outline generation was run but failed to produce chapters. Cannot proceed.[/red]"
            )
            return
        # If no outline and it wasn't run, implies starting with corrupted state? Exit likely needed.

        # --- Chapter Generation Step ---
        chapter_gen_run_or_skipped = (
            False  # Track if this step was attempted or explicitly skipped
        )
        if has_outline:
            needs_generation = any(
                c.find("content") is None or not c.findall(".//content/paragraph")
                for c in chapters_element.findall(".//chapter")
            )
            run_generation = False
            if needs_generation:
                console.print(
                    "\n[yellow]Some chapters are missing content. Proceeding to Chapter Generation.[/yellow]"
                )
                run_generation = True
            elif (
                outline_step_run
            ):  # If outline was just run, ask if user wants to gen content
                if Confirm.ask(
                    "\n[green]All chapters have empty content as expected after outline generation.[/green] [yellow]Run Chapter Generation now? [/yellow]",
                    default=True,
                ):
                    run_generation = True
            else:  # Outline exists and doesn't strictly need generation
                console.print(
                    "\n[green]All chapters appear to have content (or content structure).[/green]"
                )
                if Confirm.ask(
                    "[yellow]Do you want to run Chapter Generation again (may overwrite existing content based on strategy)? [/yellow]",
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
            console.print(
                "[yellow]Skipping Chapter Generation as no valid outline exists.[/yellow]"
            )
            # This path should ideally not be reached if outline checks above work

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
        if has_outline:
            console.print("\n[yellow]Proceeding to the Editing phase.[/yellow]")
            editing_result = self.edit_book()  # Returns True if finished normally, False if issues/aborted? (Check edit_book return logic)
            editing_run_or_skipped = True  # Mark as run
            if not editing_result:  # Assuming False means aborted or major issue
                console.print(
                    "[yellow]Editing phase was aborted or encountered issues.[/yellow]"
                )
                if not Confirm.ask(
                    "[yellow]Continue to Final Save despite editing issues? [/yellow]",
                    default=True,
                ):
                    console.print("Exiting.")
                    return
            else:
                console.print("[green]Editing phase completed.[/green]")
        else:
            console.print("[yellow]Skipping Editing as no outline exists.[/yellow]")
            editing_run_or_skipped = True  # Skipped successfully

        # --- Proceed Confirmation 3 ---
        if self.book_root is not None and self.book_dir is not None:
            if editing_run_or_skipped:  # Only proceed if editing finished/skipped OK
                if not Confirm.ask(
                    "\n[yellow]Editing complete (or skipped). Proceed to Final Save and HTML Generation? [/yellow]",
                    default=True,
                ):
                    console.print("Skipping final save and generation as requested.")
                    return
            # If editing failed and user opted out, we already exited

        # --- Final Save & HTML Output ---
        if self.book_root is not None and self.book_dir is not None:
            console.print("[cyan]Performing final save...[/cyan]")
            final_xml_filename = "final.xml"
            self._save_book_state(final_xml_filename)
            final_html_filename = "final.html"
            self._generate_html_output(final_html_filename)

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
                "[bold red]Could not perform final save as book data or directory is missing.[/bold red]"
            )


# --- Main Execution Guard ---
if __name__ == "__main__":
    console.print(Panel("📚 Novel Writer Setup 📚", style="blue"))

    resume_folder_path = None
    # --- Resume Logic ---
    try:
        # Find potential project folders: YYYYMMDD-slug-uuid pattern
        potential_folders = [
            d
            for d in Path(".").iterdir()
            if d.is_dir() and re.match(r"^\d{8}-[\w-]+-[\w-]+$", d.name)
        ]

        if potential_folders:
            console.print("Found existing book projects:")
            choices_map = {}
            display_choices = []
            # Sort by date descending (first part of name)
            potential_folders.sort(key=lambda p: p.name.split("-", 1)[0], reverse=True)

            # Limit display to maybe 10 most recent?
            MAX_RESUME_OPTIONS = 10
            displayed_count = 0
            for folder_path in potential_folders:
                if displayed_count >= MAX_RESUME_OPTIONS:
                    console.print(
                        f"[dim]... ({len(potential_folders) - displayed_count} more found)[/dim]"
                    )
                    break

                try:
                    parts = folder_path.name.split("-", 2)
                    date_str = parts[0]
                    title_slug = parts[1]
                    # Try reading title from outline.xml or final.xml for better display name
                    display_title = title_slug.replace(
                        "-", " "
                    ).title()  # Default from slug
                    for xml_name in ["final.xml", "outline.xml"]:
                        state_file = folder_path / xml_name
                        if state_file.exists():
                            try:
                                tree = ET.parse(state_file)
                                root = tree.getroot()
                                title_text = root.findtext("title")
                                if title_text:
                                    display_title = title_text
                                    break  # Found title, stop checking
                            except Exception:
                                pass  # Ignore errors reading XML for display

                    # Format date for display
                    try:
                        display_date = datetime.strptime(
                            date_str, DATE_FORMAT_FOR_FOLDER
                        ).strftime("%Y-%m-%d")
                    except ValueError:
                        display_date = date_str  # Fallback to raw date

                    display_name = f"{display_date}: {display_title}"
                    choices_map[display_name] = (
                        folder_path.name
                    )  # Map display name to actual folder name
                    display_choices.append(display_name)
                    console.print(f"- {display_name} [dim]({folder_path.name})[/dim]")
                    displayed_count += 1

                except Exception as e:
                    console.print(
                        f"[yellow]Could not parse folder {folder_path.name}: {e}[/yellow]"
                    )

            if display_choices:
                # Add option to start new
                NEW_BOOK_CHOICE = "--- Start a New Book ---"
                display_choices.append(NEW_BOOK_CHOICE)

                selected_display_name = Prompt.ask(
                    "\n[yellow]Select a project to resume, or start new[/yellow]",
                    choices=display_choices,
                    default=display_choices[0],  # Default to most recent
                )
                if selected_display_name != NEW_BOOK_CHOICE:
                    resume_folder_path = choices_map.get(selected_display_name)
                    if not resume_folder_path:
                        console.print(
                            "[red]Invalid selection. Starting new book.[/red]"
                        )
                        resume_folder_path = None  # Force start new
                else:
                    resume_folder_path = None  # User chose to start new

            else:
                console.print("[dim]No valid project folders found to resume.[/dim]")
                resume_folder_path = None  # Start new if none found

    except Exception as e:
        console.print(f"[red]Error scanning for existing projects: {e}[/red]")
        resume_folder_path = None  # Start new on error

    # --- Instantiate and Run ---
    try:
        # Pass the folder name (which includes date-slug-uuid) if resuming
        writer = NovelWriter(resume_folder_name=resume_folder_path)
        writer.run()
    except FileNotFoundError:
        console.print(
            "[bold red]Could not start Novel Writer: Project directory not found.[/bold red]"
        )
    except ValueError as ve:
        # Catch API key missing error from init
        console.print(f"[bold red]Initialization Error: {ve}[/bold red]")
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Operation interrupted by user. Exiting.[/bold yellow]"
        )
    except Exception:
        console.print(
            "[bold red]An unexpected critical error occurred during execution:[/bold red]"
        )
        console.print_exception(show_locals=False, word_wrap=True)  # Rich traceback
