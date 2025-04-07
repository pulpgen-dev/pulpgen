# -*- coding: utf-8 -*-
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom  # For pretty printing XML
import uuid
from pathlib import Path
import time
import copy
import re
import json
import getpass
import unicodedata  # For slugify
import math  # For word count display
from datetime import datetime
from html import escape  # For HTML generation

# API Client Libraries (Import conditionally later if preferred, but fine here)
import google.generativeai as genai
from google.generativeai import types as gemini_types
from google.api_core import exceptions as google_api_exceptions
import openai # Requires `pip install openai`

# Rich for UI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

# Environment Variables
from dotenv import load_dotenv

# --- Configuration ---
# Default model names (can be overridden by args)
DEFAULT_GEMINI_MODEL = "gemini-1.5-pro-latest" # Use latest 1.5 Pro as a good default
DEFAULT_OPENAI_MODEL = "gpt-4o" # Use GPT-4o as a good default

# Default Generation Config Parameters (used by both APIs where applicable)
DEFAULT_TEMPERATURE = 1.0
# Increased max output tokens as requested
DEFAULT_MAX_OUTPUT_TOKENS = 65536 # Note: Actual output depends on model limits & context window.
DEFAULT_TOP_P = 0.95
# Gemini specific
DEFAULT_TOP_K = 40
# OpenAI specific (can be added if needed, None means default)
# DEFAULT_FREQUENCY_PENALTY = None
# DEFAULT_PRESENCE_PENALTY = None

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT_FOR_FOLDER = "%Y%m%d"
# --- API Retry Configuration ---
MAX_API_RETRIES = 3
API_RETRY_BACKOFF_FACTOR = 2  # Base seconds for backoff (e.g., 2s, 4s, 8s)

# --- Rich Console Setup ---
console = Console()

# --- Helper Functions ---
# ... (slugify, pretty_xml, _count_words - unchanged) ...

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

def _count_words(text):
    """Simple word counter."""
    if not text or not text.strip():
        return 0
    return len(text.split())

def clean_llm_xml_output(xml_string):
    """Attempts to clean potential markdown/text surrounding LLM XML output."""
    if not isinstance(xml_string, str):
        return ""  # Handle None or other types

    # 1. Remove markdown code fences first (making language optional)
    # Apply strip() before regex to handle leading/trailing spaces around the whole block
    cleaned = re.sub(r"^```(.*)?\s*", "", xml_string.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 2. Strip leading/trailing whitespace again after fence removal
    cleaned = cleaned.strip()

    # 3. We will rely on ET.fromstring to handle potential remaining non-XML
    #    content before the root element (like <?xml...?> directives or comments).
    #    Aggressively searching for '<' and '>' might truncate valid XML.

    return cleaned


def parse_xml_string(xml_string, expected_root_tag="patch", attempt_clean=True):
    """
    Safely parses an XML string, optionally cleaning it first.
    expected_root_tag: The root tag name expected ('patch', 'book', etc.).
                       If 'patch', may attempt to wrap fragments.
    """
    if not xml_string:
        console.print("[bold red]Error: Received empty XML string from LLM.[/bold red]")
        return None

    cleaned_xml_string = xml_string # Keep original for error display
    if attempt_clean:
        cleaned_xml_string = clean_llm_xml_output(xml_string)

    if not cleaned_xml_string.strip():
        console.print("[bold red]Error: XML string is empty after cleaning.[/bold red]")
        return None

    try:
        # Attempt to wrap with expected_root_tag ONLY if expecting a patch and it's missing
        # More robust check: only wrap if it doesn't look like a complete XML doc already
        if expected_root_tag == "patch" and \
           not cleaned_xml_string.strip().startswith(f"<{expected_root_tag}>") and \
           not cleaned_xml_string.strip().startswith("<?xml"):
            # Check if it looks like chapter content fragments
            # This heuristic might need refinement
            if "<chapter" in cleaned_xml_string and "</chapter>" in cleaned_xml_string:
                console.print(
                    f"[yellow]Warning: LLM output seems to be missing root <{expected_root_tag}> tag for chapters, attempting to wrap.[/yellow]"
                )
                cleaned_xml_string = f"<{expected_root_tag}>{cleaned_xml_string}</{expected_root_tag}>"
            # Add more heuristics if needed for other patch types

        return ET.fromstring(cleaned_xml_string)
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

        # Display the *cleaned* string when showing context
        console.print("[yellow]Attempted to parse:[/yellow]")
        if line_num > 0:
            lines = cleaned_xml_string.splitlines()
            context_start = max(0, line_num - 3)
            context_end = min(len(lines), line_num + 2)
            for i in range(context_start, context_end):
                prefix = ">> " if (i + 1) == line_num else "   "
                console.print(f"[dim]{prefix}{lines[i]}[/dim]")
            if col_num > 0:
                console.print(f"[dim]   {' ' * (col_num - 1)}^ Error near here[/dim]")
        else:
            console.print(
                f"[dim]{cleaned_xml_string[:1000]}...[/dim]"
            )  # Fallback if line number unknown

        # Optionally show original uncleaned string if cleaning happened
        if attempt_clean and xml_string != cleaned_xml_string:
            console.print("[yellow]Original uncleaned string:[/yellow]")
            console.print(f"[dim]{xml_string[:1000]}...[/dim]")

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
    # Modified __init__ to handle API/model selection, folder naming, file input for idea
    def __init__(
        self, api_type, model_name, openai_base_url=None, # Added openai_base_url
        resume_folder_name=None, initial_prompt_file=None
    ):
        self.api_type = api_type.lower() # Store 'gemini' or 'openai'
        self.model_name = model_name
        self.openai_base_url = openai_base_url # Store OpenAI base URL

        api_panel_title = "LLM Configuration"
        api_panel_content = f"Using API: [bold cyan]{self.api_type}[/bold cyan] | Model: [bold cyan]{self.model_name}[/bold cyan]"
        if self.api_type == 'openai' and self.openai_base_url:
            api_panel_content += f"\nOpenAI Base URL: [bold cyan]{self.openai_base_url}[/bold cyan]"
        console.print(Panel(api_panel_content, title=api_panel_title))


        self.api_key = self._get_api_key()
        if not self.api_key:
            console.print(
                f"[bold red]API Key for {self.api_type.upper()} is required. Exiting.[/bold red]"
            )
            raise ValueError(f"Missing {self.api_type.upper()} API Key")

        self.client = None
        self.book_root = None
        self.book_dir = None
        self.book_id = None  # This will be the UUID part
        self.book_title_slug = "untitled"  # Default
        self.patch_counter = 0
        self.chapters_generated_in_session = set()
        self.total_word_count = 0  # Initialize word count

        if not self._init_client(): # Init client early
             raise RuntimeError(f"Failed to initialize {self.api_type.upper()} client.")

        # --- Resume or Start New ---
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
            # Use the selected LLM to generate the initial outline data
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

    def _get_api_key(self):
        """Gets the appropriate API key (Gemini or OpenAI) from environment variables or prompts the user."""
        load_dotenv()
        key_name = f"{self.api_type.upper()}_API_KEY" # e.g., GEMINI_API_KEY or OPENAI_API_KEY
        api_key = os.getenv(key_name)
        if not api_key:
            console.print(
                f"[yellow]{key_name} not found in environment variables or .env file.[/yellow]"
            )
            try:
                api_key = getpass.getpass(f"Enter your {self.api_type.upper()} API Key: ")
            except (
                EOFError
            ):  # Handle environments where getpass isn't available (e.g., some CI/CD)
                console.print(f"[red]Could not read {self.api_type.upper()} API key from input.[/red]")
                return None
        return api_key

    def _init_client(self):
        """Initializes the appropriate API client (Gemini or OpenAI)."""
        if not self.api_key:
            console.print(
                f"[bold red]Cannot initialize {self.api_type.upper()} client without API Key.[/bold red]"
            )
            return False

        try:
            if self.api_type == 'gemini':
                genai.configure(api_key=self.api_key)
                # Defer setting generation config until the actual call for flexibility
                self.client = genai.GenerativeModel(model_name=self.model_name)
                # Simple test call (optional, consumes quota)
                # self.client.count_tokens("test")
                console.print(
                    f"[green]Successfully initialized Google Gemini client with model '{self.model_name}'[/green]"
                )
                return True
            elif self.api_type == 'openai':
                 # Setup parameters for OpenAI client, including base_url if provided
                openai_params = {
                    "api_key": self.api_key
                }
                if self.openai_base_url:
                    openai_params["base_url"] = self.openai_base_url
                    console.print(f"[dim]Using custom OpenAI base URL: {self.openai_base_url}[/dim]")

                self.client = openai.OpenAI(**openai_params)
                # Simple test call (optional, consumes quota/checks connection)
                # try:
                #      self.client.models.list()
                #      console.print("[green]Successfully tested connection to OpenAI endpoint.[/green]")
                # except Exception as test_e:
                #      console.print(f"[yellow]Warning: Could not verify connection to OpenAI endpoint: {test_e}[/yellow]")
                #      # Continue anyway, might work for completions

                console.print(
                    f"[green]Successfully initialized OpenAI client with model '{self.model_name}'[/green]"
                )
                return True
            else:
                console.print(f"[bold red]Unsupported API type: {self.api_type}[/bold red]")
                return False
        except Exception as e:
            console.print(
                f"[bold red]Fatal Error: Could not initialize {self.api_type.upper()} client.[/bold red]"
            )
            console.print(f"Error details: {e}")
            self.client = None
            return False

    # New helper to get just title/synopsis without full structure generation yet
    def _generate_minimal_outline(self, idea):
        """Attempts to generate just the title and synopsis using the selected LLM."""
        console.print(f"[cyan]Generating initial title and synopsis using {self.api_type}...[/cyan]")
        prompt = f"""
Based on the following book idea/description, please generate ONLY a compelling title and a brief (1-2 sentence) synopsis.

Idea/Description:
---
{escape(idea)}
---

Output format (Strictly JSON):
{{
  "title": "Your Generated Title",
  "synopsis": "Your generated brief synopsis."
}}

Do not include any other text, explanations, or markdown. Just the JSON object.
"""
        # Use the main LLM response function
        response_json_str = self._get_llm_response(
            prompt, "Generating Title/Synopsis", allow_stream=False
        ) # Don't need streaming for JSON

        if response_json_str:
            try:
                # --- Refined Cleaning Logic ---
                # 1. Strip leading/trailing whitespace from the raw response
                stripped_response = response_json_str.strip()

                # 2. Remove potential markdown fences (json optional)
                cleaned_json_str = re.sub(
                    # Match ``` potentially followed by 'json' and optional whitespace/newline
                    r"^```(json)?\s*",
                    "",
                    stripped_response,
                    flags=re.IGNORECASE | re.MULTILINE
                )
                # Match optional whitespace/newline followed by ``` at the end
                cleaned_json_str = re.sub(
                    r"\s*```$",
                    "",
                    cleaned_json_str
                )

                # 3. Strip again after fence removal, just in case
                cleaned_json_str = cleaned_json_str.strip()
                # --- End Refined Cleaning Logic ---


                # 4. Parse the cleaned string
                data = json.loads(cleaned_json_str)
                if isinstance(data, dict) and "title" in data and "synopsis" in data:
                    console.print("[green]✓ Title and Synopsis generated.[/green]")
                    return {
                        "title": data.get("title"),
                        "synopsis": data.get("synopsis"),
                    }
                else:
                    console.print(
                        f"[yellow]Warning: {self.api_type.upper()} response was not valid JSON with title/synopsis keys after cleaning.[/yellow]"
                    )
                    console.print(f"[dim]Cleaned String Attempted: {cleaned_json_str[:500]}[/dim]")
                    console.print(f"[dim]Original Response: {response_json_str[:500]}[/dim]")


            except json.JSONDecodeError as e:
                console.print(
                    f"[yellow]Warning: Failed to decode {self.api_type.upper()} response as JSON: {e}[/yellow]"
                )
                # Show both the original and what was attempted to parse
                console.print(f"[dim]Cleaned String Attempted: {cleaned_json_str[:500]}[/dim]")
                console.print(f"[dim]Original Response: {response_json_str[:500]}[/dim]")
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

    # --- API Abstraction Layer ---

    def _call_gemini_api(self, prompt_content, task_description, allow_stream):
        """Handles the specific logic for calling the Google Gemini API."""
        full_response = ""
        api_name = "Gemini"

        # Construct Gemini-specific generation config
        generation_config = gemini_types.GenerationConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, # Use the updated default
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
        )

        if allow_stream:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(description=f"[cyan]{api_name} is thinking...", total=None)
                response_stream = self.client.generate_content(
                    contents=prompt_content,
                    stream=True,
                    generation_config=generation_config
                )
                console.print(f"[cyan]>>> {api_name} Response ({task_description}):[/cyan]")
                first_chunk = True
                try:
                    for chunk in response_stream:
                        if first_chunk:
                            progress.update(task, description="[cyan]Receiving response...")
                            first_chunk = False

                        # Check for blocking reasons (more robust check)
                        block_reason = getattr(getattr(chunk, 'prompt_feedback', None), 'block_reason', None)
                        if block_reason:
                            ratings = getattr(chunk.prompt_feedback, 'safety_ratings', [])
                            ratings_str = "\n".join([f"  - {r.category.name}: {r.probability.name}" for r in ratings])
                            msg = f"Content generation blocked during streaming.\nReason: {block_reason.name}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                            console.print(f"\n[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                            raise gemini_types.BlockedPromptException(msg)

                        # Append text safely
                        try:
                            chunk_text = chunk.text
                            print(chunk_text, end="", flush=True)
                            full_response += chunk_text
                        except ValueError: # Sometimes mid-stream errors occur
                             # Check finish_reason if available on the chunk (less common)
                            finish_reason = getattr(getattr(chunk, 'candidates', [None])[0], 'finish_reason', None)
                            if finish_reason and finish_reason.name == 'SAFETY':
                                msg = "Content generation stopped mid-stream due to safety."
                                console.print(f"\n[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                                raise gemini_types.BlockedPromptException(msg)
                            else:
                                console.print(f"\n[yellow]Warning: Received potentially invalid chunk data from {api_name}.[/yellow]")
                        except AttributeError: # Handle cases where chunk might not have .text
                             # Check parts if available
                            try:
                                for part in chunk.parts:
                                    if hasattr(part, "text"):
                                        chunk_text = part.text
                                        print(chunk_text, end="", flush=True)
                                        full_response += chunk_text
                            except (AttributeError, ValueError):
                                console.print(f"\n[yellow]Warning: Could not extract text from chunk part in {api_name} stream.[/yellow]")


                finally:
                    print() # Newline after streaming finishes or errors

                # Final check after stream (important)
                final_block_reason = getattr(getattr(response_stream, 'prompt_feedback', None), 'block_reason', None)
                if final_block_reason:
                    ratings = getattr(response_stream.prompt_feedback, 'safety_ratings', [])
                    ratings_str = "\n".join([f"  - {r.category.name}: {r.probability.name}" for r in ratings])
                    msg = f"Content generation blocked (final check).\nReason: {final_block_reason.name}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                    console.print(f"\n[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                    raise gemini_types.BlockedPromptException(msg)
                # Check candidate finish reason if no block reason
                elif not final_block_reason:
                     final_candidate = getattr(response_stream, 'candidates', [None])[0]
                     if final_candidate:
                         final_finish_reason = getattr(final_candidate, 'finish_reason', None)
                         if final_finish_reason:
                            if final_finish_reason.name == 'SAFETY':
                                 msg = "Content generation likely blocked due to safety (Final Finish Reason: SAFETY)."
                                 console.print(f"\n[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                                 raise gemini_types.BlockedPromptException(msg)
                            elif final_finish_reason.name == 'MAX_TOKENS':
                                 console.print(f"\n[yellow]Warning ({api_name}): Generation stopped because maximum token limit ({DEFAULT_MAX_OUTPUT_TOKENS}) was reached.[/yellow]")


        else: # Non-streaming Gemini
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"[cyan]{api_name} is thinking...", total=None)
                response = self.client.generate_content(
                    contents=prompt_content,
                    generation_config=generation_config
                )

            # Check for blocking
            block_reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', None)
            if block_reason:
                ratings = getattr(response.prompt_feedback, 'safety_ratings', [])
                ratings_str = "\n".join([f"  - {r.category.name}: {r.probability.name}" for r in ratings])
                msg = f"Content generation blocked.\nReason: {block_reason.name}\nSafety Ratings:\n{ratings_str or 'N/A'}"
                console.print(f"[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                raise gemini_types.BlockedPromptException(msg)

            # Extract text safely
            try:
                full_response = response.text
            except ValueError as ve: # Often indicates blocked content without explicit reason
                 # Check candidate finish reason
                 final_candidate = getattr(response, 'candidates', [None])[0]
                 if final_candidate:
                    finish_reason = getattr(final_candidate, 'finish_reason', None)
                    if finish_reason:
                        if finish_reason.name == 'SAFETY':
                             msg = f"Content generation likely blocked due to safety (Finish Reason: SAFETY). Error: {ve}"
                             console.print(f"[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                             raise gemini_types.BlockedPromptException(msg) from ve
                        elif finish_reason.name == 'MAX_TOKENS':
                             console.print(f"\n[yellow]Warning ({api_name}): Generation stopped because maximum token limit ({DEFAULT_MAX_OUTPUT_TOKENS}) was reached.[/yellow]")
                             # Try to extract partial text anyway if possible
                             try:
                                  full_response = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                                  if not full_response: raise AttributeError("No text found even after MAX_TOKENS")
                             except Exception:
                                   console.print(f"[bold red]Error extracting text from {api_name} response after MAX_TOKENS finish: {ve}[/bold red]")
                                   raise ValueError(f"Could not extract text from {api_name} response after MAX_TOKENS finish: {ve}") from ve

                        else:
                            # Re-raise original error if not safety/max_tokens related
                            console.print(f"[bold red]Error extracting text from {api_name} response: {ve}[/bold red]")
                            console.print(f"[dim]Response object (summary): {response}[/dim]")
                            raise ValueError(f"Could not extract text from {api_name} response: {ve}") from ve
                    else: # No finish reason, raise original error
                        console.print(f"[bold red]Error extracting text from {api_name} response: {ve}[/bold red]")
                        raise ValueError(f"Could not extract text from {api_name} response: {ve}") from ve
                 else: # No candidate, raise original error
                     console.print(f"[bold red]Error extracting text from {api_name} response: {ve}[/bold red]")
                     raise ValueError(f"Could not extract text from {api_name} response: {ve}") from ve

            except AttributeError: # Handle cases where .text might be missing, check parts
                 try:
                     full_response = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                     if not full_response: # If parts exist but are empty
                         final_candidate = getattr(response, 'candidates', [None])[0]
                         finish_reason_name = 'Unknown'
                         if final_candidate and hasattr(final_candidate, 'finish_reason'):
                              finish_reason_name = final_candidate.finish_reason.name
                         raise ValueError(f"Response object has no 'text' and empty 'parts'. Finish Reason: {finish_reason_name}")
                 except (AttributeError, ValueError) as e:
                     console.print(f"[bold red]Error extracting text from {api_name} response parts: {e}[/bold red]")
                     console.print(f"[dim]Response object (summary): {response}[/dim]")
                     raise ValueError(f"Could not extract text from {api_name} response: {e}") from e


            console.print(f"[cyan]>>> {api_name} Response ({task_description}):[/cyan]")
            console.print(f"[dim]{full_response[:1000]}{'...' if len(full_response) > 1000 else ''}[/dim]")

        return full_response

    def _call_openai_api(self, prompt_content, task_description, allow_stream):
        """Handles the specific logic for calling the OpenAI API."""
        full_response = ""
        api_name = "OpenAI"

        messages = [{"role": "user", "content": prompt_content}]

        # Prepare common parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS, # Use the updated default
            "top_p": DEFAULT_TOP_P,
            "stream": allow_stream,
        }

        if allow_stream:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(description=f"[cyan]{api_name} is thinking...", total=None)
                response_stream = self.client.chat.completions.create(**params)
                console.print(f"[cyan]>>> {api_name} Response ({task_description}):[/cyan]")
                first_chunk = True
                finish_reason = None
                try:
                    for chunk in response_stream:
                        if first_chunk:
                            progress.update(task, description="[cyan]Receiving response...")
                            first_chunk = False

                        # Check for content and finish reason in each chunk
                        choice = chunk.choices[0] if chunk.choices else None
                        if choice:
                            content = choice.delta.content
                            if content:
                                print(content, end="", flush=True)
                                full_response += content

                            # Store finish reason if present
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                                break # Stop processing chunks once finished

                finally:
                     print() # Newline

                # Check finish reason after stream
                if finish_reason == 'content_filter':
                    msg = "Content generation stopped by OpenAI's content filter."
                    console.print(f"\n[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                    raise openai.BadRequestError(msg, body={"message": msg, "type": "content_filter"}) # Simulate error structure
                elif finish_reason == 'length':
                     console.print(f"\n[yellow]Warning ({api_name}): Generation stopped because maximum token limit ({DEFAULT_MAX_OUTPUT_TOKENS}) was reached.[/yellow]")
                # Handle other finish reasons if necessary

        else: # Non-streaming OpenAI
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task(description=f"[cyan]{api_name} is thinking...", total=None)
                response = self.client.chat.completions.create(**params)

             # Check finish reason
             finish_reason = response.choices[0].finish_reason if response.choices else None
             if finish_reason == 'content_filter':
                 msg = "Content generation blocked by OpenAI's content filter."
                 console.print(f"[bold red]API Safety Error ({api_name}): {msg}[/bold red]")
                 # Include response details if helpful
                 error_body = response.model_dump() if hasattr(response, 'model_dump') else {'message': msg}
                 raise openai.BadRequestError(msg, body=error_body)
             elif finish_reason == 'length':
                  console.print(f"\n[yellow]Warning ({api_name}): Generation stopped because maximum token limit ({DEFAULT_MAX_OUTPUT_TOKENS}) was reached.[/yellow]")
             # Check other finish reasons if needed

             # Extract text
             if response.choices and response.choices[0].message:
                 full_response = response.choices[0].message.content or ""
             else:
                 # This case should be rare if no error was raised
                 console.print(f"[bold red]Error ({api_name}): No response content found.[/bold red]")
                 console.print(f"[dim]Response object (summary): {response}[/dim]")
                 raise ValueError(f"Could not extract text from {api_name} response.")

             console.print(f"[cyan]>>> {api_name} Response ({task_description}):[/cyan]")
             console.print(f"[dim]{full_response[:1000]}{'...' if len(full_response) > 1000 else ''}[/dim]")

        return full_response

    # --- Enhanced Error Handling & Retry ---
    def _get_llm_response(
        self, prompt_content, task_description="Generating content", allow_stream=True
    ):
        """
        Sends prompt to the selected LLM (Gemini or OpenAI), handles streaming/non-streaming,
        and manages API errors with retries and backoff.
        """
        if self.client is None:
            console.print(
                f"[bold red]Error: {self.api_type.upper()} client not initialized. Cannot make API call.[/bold red]"
            )
            return None

        retries = 0
        last_error = None # Store last error for final message

        while retries < MAX_API_RETRIES:
            full_response = ""
            api_name_upper = self.api_type.upper()
            try:
                console.print(
                    Panel(
                        f"[yellow]Sending request to {api_name_upper} ({task_description})... (Attempt {retries + 1}/{MAX_API_RETRIES})[/yellow]",
                        border_style="dim",
                    )
                )

                # Call the appropriate API-specific function
                if self.api_type == 'gemini':
                    full_response = self._call_gemini_api(prompt_content, task_description, allow_stream)
                elif self.api_type == 'openai':
                    full_response = self._call_openai_api(prompt_content, task_description, allow_stream)
                else:
                    # Should have been caught during init, but safety check
                    console.print(f"[bold red]Internal Error: Unsupported API type '{self.api_type}' in _get_llm_response.[/bold red]")
                    return None

                # --- Success Case ---
                # Check if the response is effectively empty (might happen even without errors)
                if not full_response or not full_response.strip():
                     console.print(f"[yellow]Warning: Received an empty response from {api_name_upper}.[/yellow]")
                     # Consider if this should trigger a retry or just return None/empty
                     # For now, treat it as success but log warning. User can retry via editing if needed.


                console.print(
                    Panel(
                        f"[green]✓ {api_name_upper} response received successfully.[/green]",
                        border_style="dim",
                    )
                )
                return full_response.strip()  # Successful, exit the retry loop

            # --- Exception Handling & Retry Logic ---

            # Gemini Specific Errors
            except gemini_types.BlockedPromptException as safety_error:
                error_msg = f"Safety Error ({api_name_upper}): {safety_error}"
                last_error = error_msg
                # Message already printed in _call_gemini_api
                console.print("[yellow]Safety errors often require prompt changes. Retrying may not help.[/yellow]")
                if not Confirm.ask(f"Try sending the request again anyway? (Attempt {retries + 1}/{MAX_API_RETRIES})", default=False):
                    return None

            except google_api_exceptions.ResourceExhausted as rate_limit_error:
                error_msg = f"API Rate Limit Error ({api_name_upper}): {rate_limit_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry makes sense

            except google_api_exceptions.DeadlineExceeded as timeout_error:
                error_msg = f"API Timeout Error ({api_name_upper}): {timeout_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help

            except google_api_exceptions.GoogleAPICallError as api_error:
                error_msg = f"Gemini API Call Error: {type(api_error).__name__} - {api_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help for transient issues

            # OpenAI Specific Errors
            except openai.RateLimitError as rate_limit_error:
                error_msg = f"API Rate Limit Error ({api_name_upper}): {rate_limit_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry makes sense

            except openai.APITimeoutError as timeout_error:
                 error_msg = f"API Timeout Error ({api_name_upper}): {timeout_error}"
                 last_error = error_msg
                 console.print(f"[bold red]{error_msg}[/bold red]")
                 # Retry might help

            except openai.APIConnectionError as conn_error:
                 error_msg = f"API Connection Error ({api_name_upper}): {conn_error}"
                 last_error = error_msg
                 console.print(f"[bold red]{error_msg}[/bold red]")
                 # Retry might help (especially with local endpoints)

            except openai.AuthenticationError as auth_error:
                 error_msg = f"API Authentication Error ({api_name_upper}): {auth_error}"
                 last_error = error_msg
                 console.print(f"[bold red]{error_msg}[/bold red]")
                 # Retrying won't help - likely bad key
                 console.print("[red]Please check your API key. Aborting call.[/red]")
                 return None # Don't retry auth errors

            except openai.BadRequestError as bad_request_error:
                # Often includes content filter errors, check message
                error_msg = f"API Bad Request Error ({api_name_upper}): {bad_request_error}"
                last_error = error_msg
                # Message already printed in _call_openai_api if content_filter
                if "content_filter" in str(bad_request_error).lower():
                     console.print("[yellow]This was likely due to the content safety filter. Retrying may not help.[/yellow]")
                     if not Confirm.ask(f"Try sending the request again anyway? (Attempt {retries + 1}/{MAX_API_RETRIES})", default=False):
                         return None
                else:
                     # Other bad requests (e.g., invalid model, malformed input) likely won't be fixed by retry
                     console.print("[red]This error suggests an issue with the request itself (e.g. invalid model, context too long). Aborting call.[/red]")
                     # Optionally print more details if available
                     if hasattr(bad_request_error, 'body') and bad_request_error.body:
                          console.print(f"[dim]Error Body: {bad_request_error.body}[/dim]")
                     return None

            except openai.APIError as api_error: # General OpenAI API errors
                error_msg = f"OpenAI API Error: {type(api_error).__name__} - {api_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help for transient server issues (e.g., 5xx)

            # General Errors (potentially affecting both)
            except ET.ParseError as xml_error:
                # This happens *after* getting a response, indicates bad response format
                error_msg = f"XML Parsing Error after receiving response: {xml_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print(f"[yellow]This often means the {api_name_upper} response was incomplete or malformed.[/yellow]")
                console.print(f"[dim]Partial response received before error:\n{full_response[:500]}...[/dim]")
                # Retry might get a complete response

            except ValueError as val_error:  # Catch extraction errors raised above
                error_msg = f"Data Extraction/Validation Error: {val_error}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                # Retry might help if it was transient (e.g., empty response)

            except Exception as e:
                error_msg = f"Unexpected Error during API call: {type(e).__name__} - {e}"
                last_error = error_msg
                console.print(f"[bold red]{error_msg}[/bold red]")
                console.print_exception(show_locals=False, word_wrap=True)
                # Retry might help for some transient errors

            # --- Retry Action ---
            retries += 1
            if retries < MAX_API_RETRIES:
                wait_time = API_RETRY_BACKOFF_FACTOR * (2 ** (retries - 1)) # Exponential backoff
                console.print(f"[yellow]Waiting {wait_time:.1f} seconds before retrying...[/yellow]")
                time.sleep(wait_time)
                # Ask user if they want to proceed with the retry
                if not Confirm.ask(f"Error encountered. Proceed with retry attempt {retries + 1}/{MAX_API_RETRIES}?", default=True):
                    console.print(f"[red]Aborting API call after error due to user choice.[/red]")
                    return None  # User cancelled retry
                # Continue loop
            else:
                console.print(f"[bold red]Maximum retries ({MAX_API_RETRIES}) reached. Failed to get a valid response from {api_name_upper}.[/bold red]")
                if last_error:
                     console.print(f"[bold red]Last error: {last_error}[/bold red]")
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
            sorted_chapters_for_wc = []
            invalid_id_chapters = []
            for c in chapters:
                try:
                    # Try converting ID to int for sorting
                    int(c.get("id", "NaN"))
                    sorted_chapters_for_wc.append(c)
                except ValueError:
                    invalid_id_chapters.append(c) # Collect chapters with non-integer IDs

            # Sort the ones with valid integer IDs
            sorted_chapters_for_wc.sort(key=lambda c: int(c.get("id")))
            # Append the invalid ID chapters at the end (or handle differently if needed)
            all_sorted_chapters = sorted_chapters_for_wc + invalid_id_chapters

            for chap in all_sorted_chapters:
                chap_id = chap.get("id")
                content = chap.find("content")
                chap_wc = 0
                if content is not None:
                    paragraphs = content.findall(".//paragraph")
                    for para in paragraphs:
                        chap_wc += _count_words(para.text)
                if chap_id: # Only count if ID exists
                    chapter_word_counts[chap_id] = chap_wc
                    total_wc += chap_wc
                else: # Chapter missing ID
                     # Count towards total but don't store by ID
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
            # Sort characters by name for display
            sorted_characters = sorted(characters, key=lambda char: char.findtext("name", "zzz").lower())
            for char in sorted_characters:
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

            # Sort chapters by ID numerically before display (reuse sorted list from wc calc)
            sorted_chapters_display = all_sorted_chapters # Use the list already sorted

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
                chap_wc = chapter_word_counts.get(chap_id, 0) if chap_id != "N/A" else _count_words(ET.tostring(content, method='text', encoding='unicode')) # Recalculate if no ID


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
        # Create a minimal representation for the prompt to avoid excessive length
        prompt_root = ET.Element("book")
        ET.SubElement(prompt_root, "title").text = self.book_root.findtext("title", "")
        ET.SubElement(prompt_root, "synopsis").text = self.book_root.findtext("synopsis", "")
        initial_idea_elem = self.book_root.find("initial_idea")
        if initial_idea_elem is not None and initial_idea_elem.text:
             ET.SubElement(prompt_root, "initial_idea").text = initial_idea_elem.text
        # Do NOT include existing chapters/characters in this specific prompt
        current_book_xml_for_prompt = ET.tostring(prompt_root, encoding="unicode")


        prompt = f"""
You are a creative assistant expanding an initial book concept into a full outline.
The current minimal state of the book is:
```xml
{escape(current_book_xml_for_prompt)}
```

Based on the title, synopsis, and initial idea (if present), please generate the missing or incomplete parts of the outline, specifically:
1.  A detailed `<characters>` section with multiple `<character>` elements (each with a unique alphanumeric `id`, `<name>`, `<description>`). Define the main characters and their arcs briefly. Use CamelCase or simple alphanumeric IDs (e.g., 'mainHero', 'villain'). Ensure IDs are unique.
2.  A detailed `<chapters>` section containing approximately {num_chapters} `<chapter>` elements. For each chapter:
    *   Ensure it has a unique sequential string `id` (e.g., '1', '2', ...).
    *   Include `<number>` (matching the id), `<title>`, and a detailed `<summary>` (150-200 words) outlining key events and progression.
    *   Keep the `<content>` tag EMPTY for now (like `<content/>` or `<content></content>`).
    *   Ensure the summaries form a coherent narrative arc (setup, rising action, climax, falling action, resolution).

Output ONLY the complete `<book>` XML structure, merging the generated details with the existing title/synopsis/initial_idea provided in the input XML above. Do not include any text outside the `<book>` tags. Ensure chapter IDs are correct and sequential starting from 1. Ensure character IDs are unique. Strictly adhere to XML format.
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
                    for char in list(chars_elem.findall("character")): # Iterate over a list copy for safe removal
                        valid_chars_found = True # Assume valid until proven otherwise
                        char_id = char.get("id")
                        name_elem = char.find("name")
                        name = (
                            name_elem.text.strip()
                            if name_elem is not None
                            and name_elem.text
                            and name_elem.text.strip()
                            else None  # Mark as None if missing or empty
                        )

                        # Generate name if missing or empty
                        if not name:
                            name = f"Character_{uuid.uuid4().hex[:4]}"
                            if name_elem is not None:
                                name_elem.text = name
                            else:
                                name_elem = ET.SubElement(char, "name")
                                name_elem.text = name
                            console.print(
                                f"[yellow]Warning: Character missing name. Assigned '{name}'.[/yellow]"
                            )

                        # Generate unique ID if missing, empty, or duplicate
                        if (
                            not char_id
                            or not char_id.strip()
                            or char_id in used_char_ids
                        ):
                            original_id_for_msg = char_id if char_id else "missing"
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
                                f"[yellow]Warning: Character '{name}' has missing/duplicate ID ('{original_id_for_msg}'). Setting unique ID to '{new_id}'.[/yellow]"
                            )
                            char.set("id", new_id)
                            char_id = new_id  # Use the new one

                        used_char_ids.add(char_id)

                        # Check description - add placeholder if missing or empty
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
                        # Don't fail validation for this, maybe the story has no characters? Less likely but possible.


                # Ensure Title/Synopsis/Idea exist (copy from original if LLM removed them)
                for tag_info in [("title", 0), ("synopsis", 1), ("initial_idea", 2)]:
                    tag, default_pos = tag_info
                    if new_book_root.find(tag) is None:
                        # Check the original self.book_root (before potential overwrite)
                        original_elem = self.book_root.find(tag)
                        insert_elem = None
                        if original_elem is not None and original_elem.text:
                            console.print(
                                f"[yellow]Warning: LLM removed <{tag}>. Restoring from original state.[/yellow]"
                            )
                            insert_elem = copy.deepcopy(original_elem)
                        else:
                            # Add placeholder if it was missing originally too
                            console.print(
                                f"[yellow]Warning: LLM removed <{tag}> or it was empty/missing. Adding placeholder.[/yellow]"
                            )
                            insert_elem = ET.Element(tag)
                            placeholder_text = f"{tag.replace('_', ' ').title()} needed."
                            if tag == 'title': placeholder_text = "Placeholder Title"
                            insert_elem.text = placeholder_text

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
                        if tag in ["title"]: # Missing title is critical
                            validation_passed = False

                # --- Commit Changes ---
                if validation_passed or Confirm.ask("[yellow]Outline validation found issues (see warnings). Save this outline anyway?[/yellow]", default=True):
                    self.book_root = new_book_root # Replace current root
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

                else: # Validation failed and user chose not to save
                     console.print("[red]Outline generation failed validation and was discarded by user.[/red]")
                     return False # Indicate failure


            else:
                console.print(
                    "[bold red]Failed to parse the generated outline XML or root tag was not <book>. Outline not saved.[/bold red]"
                )
                return False
        else:
            console.print(
                f"[bold red]Failed to get a valid response from the {self.api_type.upper()} for the outline.[/bold red]"
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
                batch_size=3 # Slightly smaller batch size might improve consistency/reduce token usage per call
            )

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
                chapter_details_prompt += f'- Chapter {chap_num} (ID: {chap_id}): "{escape(chap_title)}"\n  Summary: {escape(chap_summary)}\n'

            # Create a context representation - potentially prune older chapters if context gets too large
            # For now, send the whole book state. Be mindful of token limits.
            current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

            # Adjusted word count goal - more flexible instruction
            prompt = f"""
You are a novelist continuing the story based on the full book context provided below.
Your task is to write the full prose content for the following {len(chapters_to_generate)} chapters:
{chapter_details_prompt}

Guidelines:
- Write detailed and engaging prose for each chapter, aiming for a substantial length appropriate for a novel chapter (e.g., ~1500-3500 words *per chapter*). Use the summary as a guide for the required detail and length. Focus on quality over hitting an exact word count.
- Maintain consistency with the established plot, characters (personalities, motivations, relationships), tone, and writing style evident in the rest of the book context (including summaries of unwritten chapters and content of written ones).
- Ensure the events of these chapters align with their summaries and logically connect preceding/succeeding chapters. Pay attention to how these chapters function together within the narrative arc.
- For EACH chapter you generate content for, structure it within `<content>` tags, divided into paragraphs using `<paragraph>` tags. Each paragraph tag MUST have a unique sequential `id` attribute within that chapter (e.g., `<paragraph id="1">...</paragraph>`, `<paragraph id="2">...</paragraph>`, etc.). Start paragraph IDs from "1" for each chapter. Ensure paragraphs contain meaningful text and are not overly short unless stylistically required (e.g., dialogue).
- Output ONLY an XML `<patch>` structure containing the `<chapter>` elements for the requested chapters. Each `<chapter>` element must have the correct `id` attribute and contain the fully generated `<content>` tag with its `<paragraph>` children. DO NOT include chapters that were not requested in this batch. DO NOT include XML comments unless absolutely necessary for structure clarification. Strictly adhere to XML format.

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
{escape(current_book_xml_str)}
```

Generate the `<patch>` XML containing the content for the requested chapters now ({chapter_ids_str}). Ensure all specified chapters are included in the response patch with correct IDs and valid paragraph structure (sequential IDs starting from 1, non-empty text). Output ONLY the XML structure.
"""

            response_patch_xml_str = self._get_llm_response(
                prompt,
                f"Writing Chapters {chapter_ids_str}",
                allow_stream=True,  # Streaming preferred for long content
            )

            # Check if response was obtained (handles None return from _get_llm_response)
            if response_patch_xml_str is None:
                console.print(
                    f"[bold red]Failed to get response from {self.api_type.upper()} for batch starting with Chapter {chapters_to_generate[0].get('id')}.[/bold red]"
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

            # Optional small delay between batches to avoid hitting rate limits
            time.sleep(2) # Increase delay slightly

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
        all_chapters_elements = self.book_root.findall(".//chapter")
        all_chapter_ids = {chap.get("id") for chap in all_chapters_elements if chap.get("id")}
        if not all_chapter_ids:
            console.print("[yellow]No chapters found in the book.[/yellow]")
            return []

        # Sort IDs numerically for display
        try:
             sorted_ids_str = ", ".join(sorted(all_chapter_ids, key=int))
        except ValueError: # Handle non-integer IDs if they somehow exist
             sorted_ids_str = ", ".join(sorted(list(all_chapter_ids)))

        while True:
            raw_input = Prompt.ask(
                f"{prompt_text} (Available: {sorted_ids_str})"
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

        # Calculate current average word count for suggestion
        num_selected = len(selected_chapters)
        current_selected_wc = 0
        chapter_ids_for_wc = [c.get("id") for c in selected_chapters]
        for chap_id in chapter_ids_for_wc:
            chap = find_chapter(self.book_root, chap_id)
            if chap:
                content = chap.find("content")
                if content:
                    current_selected_wc += sum(_count_words(p.text) for p in content.findall(".//paragraph"))

        current_avg_wc = (current_selected_wc / num_selected) if num_selected > 0 else 2000 # Default if no content yet
        suggested_target_wc = max(3000, math.ceil(current_avg_wc * 1.5))

        target_word_count = IntPrompt.ask(
            f"[yellow]Enter target word count per chapter[/yellow] (current avg: ~{current_avg_wc:,.0f})",
            default=suggested_target_wc,
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
            # Include existing content for context
            existing_content_xml = ""
            content_elem = chapter_element.find("content")
            if content_elem is not None:
                # Limit context size slightly if needed, maybe only first/last N paras? For now, full content.
                existing_content_xml = ET.tostring(content_elem, encoding="unicode")

            chapter_details_prompt += f"""
<chapter id="{chap_id}">
  <number>{chap_num}</number>
  <title>{escape(chap_title)}</title>
  <summary>{escape(chap_summary)}</summary>
  <!-- Existing Content Start -->
  {escape(existing_content_xml)}
  <!-- Existing Content End -->
</chapter>
"""

        current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")
        prompt = f"""
You are a novelist tasked with expanding specific chapters of a manuscript.
Your goal is to rewrite the *entire content* for the chapter(s) listed below, significantly increasing their length to approximately {target_word_count:,} words *each*, while enriching the detail, description, dialogue, and internal monologue.

Chapters to Expand (including their original content for context):
--- XML START ---
{chapter_details_prompt}
--- XML END ---

Guidelines:
- Rewrite the full `<content>` for EACH specified chapter ({", ".join(chapter_ids)}). Replace the existing content entirely.
- Target word count: Approximately {target_word_count:,} words per chapter.
- Elaborate on existing scenes, add descriptive details, deepen character thoughts/feelings, expand dialogues, and potentially add small connecting scenes *within* the chapter's scope if necessary to reach the target length naturally.
- Maintain absolute consistency with the overall plot, established characters, tone, and style provided in the full book context below. The expansion should feel seamless.
- Ensure the rewritten chapter still fulfills the purpose outlined in its original summary.
- Structure the rewritten content within `<content>` tags, using sequentially numbered `<paragraph id="...">` tags starting from 1 for each chapter. Ensure paragraphs contain text.
- Output ONLY the XML `<patch>` structure containing the rewritten `<chapter>` elements (with their new, full `<content>`). Do not include chapters not specified. Do not output any text before `<patch>` or after `</patch>`. Strictly adhere to XML format.

Full Book Context:
```xml
{escape(current_book_xml_str)}
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
            f"[yellow]Enter specific instructions for the {mode.lower()}[/yellow]"
        )
        if not instructions.strip():
            console.print("[red]No instructions provided. Aborting rewrite.[/red]")
            return

        # --- Prepare Context ---
        current_book_xml_str = ""
        temp_book_root_for_prompt = copy.deepcopy(self.book_root) # Always make a copy for prompt context modification
        target_chapter_in_prompt_copy = find_chapter(temp_book_root_for_prompt, chapter_id)

        if target_chapter_in_prompt_copy is None:
             console.print(f"[red]Error: Could not find chapter {chapter_id} in temporary copy for prompt. Aborting.[/red]")
             return

        if blackout:
            console.print(
                "[yellow]Preparing context with target chapter content removed (fresh rewrite)...[/yellow]"
            )
            content_elem_copy = target_chapter_in_prompt_copy.find("content")
            if content_elem_copy is not None:
                target_chapter_in_prompt_copy.remove(content_elem_copy) # Remove the content element entirely for blackout
                # Add back an empty one for clarity in the prompt context if desired, or leave it out. Let's add an empty one.
                ET.SubElement(target_chapter_in_prompt_copy, "content")
                console.print(
                    f"[dim]Content of chapter {chapter_id} removed for prompt context.[/dim]"
                )
            else:
                console.print(
                    f"[yellow]Warning: Chapter {chapter_id} has no <content> tag in temporary copy to remove.[/yellow]"
                )
                # Add an empty one anyway to signify it should be generated
                ET.SubElement(target_chapter_in_prompt_copy, "content")

        current_book_xml_str = ET.tostring(temp_book_root_for_prompt, encoding="unicode")


        # --- Prepare Prompt ---
        chap_num = target_chapter.findtext("number", "N/A")
        chap_title = target_chapter.findtext("title", "N/A")
        chap_summary = target_chapter.findtext("summary", "N/A")

        # Get original content XML only if *not* blackout, for user reference in prompt
        original_content_xml_for_prompt = ""
        if not blackout:
            original_content_elem = target_chapter.find("content")
            if original_content_elem is not None:
                original_content_xml_for_prompt = ET.tostring(original_content_elem, encoding="unicode")


        rewrite_context_info = f"""
Chapter to Rewrite:
- ID: {chapter_id}
- Number: {chap_num}
- Title: {escape(chap_title)}
- Summary: {escape(chap_summary)}
"""
        if not blackout and original_content_xml_for_prompt:
            rewrite_context_info += f"""
<!-- Original Content for Reference -->
{escape(original_content_xml_for_prompt)}
"""

        prompt = f"""
You are a novelist rewriting a specific chapter based on user instructions.
Your task is to rewrite the *entire content* for Chapter {chapter_id}.

{rewrite_context_info}

User's Rewrite Instructions:
---
{escape(instructions)}
---
{"*Note: The original content of this chapter was intentionally removed from the full book context below (though shown above for your reference if available) to encourage a fresh perspective based on the summary and instructions.*" if blackout else ""}

Guidelines:
- Rewrite the full `<content>` for Chapter {chapter_id} according to the instructions, replacing the original content entirely.
- Ensure the rewritten chapter aligns with its summary and maintains consistency with the overall plot, characters, tone, and style provided in the full book context below.
- Structure the rewritten content within `<content>` tags, using sequentially numbered `<paragraph id="...">` tags starting from 1. Ensure paragraphs contain meaningful text.
- Output ONLY the XML `<patch>` structure containing the single rewritten `<chapter>` element (ID: {chapter_id}) with its new, full `<content>`. Do not output any text before `<patch>` or after `</patch>`. Strictly adhere to XML format.

Full Book Context {"(Chapter " + chapter_id + " content removed/empty)" if blackout else ""}:
```xml
{escape(current_book_xml_str)}
```

Generate the `<patch>` XML for the rewritten Chapter {chapter_id} now.
"""

        suggested_patch_xml_str = self._get_llm_response(
            prompt, f"{mode} Chapter {chapter_id}", allow_stream=True
        )
        if suggested_patch_xml_str:
            apply_success = self._apply_patch(suggested_patch_xml_str)
            self._handle_patch_result(apply_success, f"{mode} (Ch {chapter_id})")

    def _is_patch_effectively_empty(self, patch_xml_str):
        """Checks if a patch string is empty, fails to parse, or only contains comments."""
        if not patch_xml_str or not patch_xml_str.strip():
            return True # Definitely empty

        # Attempt to clean and parse first
        # Use attempt_clean=False because cleaning should happen before this check
        patch_elem = parse_xml_string(patch_xml_str, expected_root_tag="patch", attempt_clean=False)

        if patch_elem is None:
            console.print("[dim]Patch check: Parsing failed, considered empty/invalid.[/dim]")
            return True # Parse failure means invalid patch

        # Check if the <patch> element has no children
        if not list(patch_elem):
            console.print("[dim]Patch check: Root <patch> element has no children, considered empty.[/dim]")
            return True # No children means empty patch

        # Check if *all* direct children of the <patch> element are comments
        # ET.Comment is for <!-- -->, ET.ProcessingInstruction for <? ... ?>
        # We only care about comments indicating no change.
        # Note: ET._Comment_Tag might be needed for older Python/ET versions if ET.Comment fails
        # Safest check: iterate and see if any non-comment element exists
        has_non_comment_child = False
        for child in patch_elem:
             # Check if it's an Element and not a Comment
             if isinstance(child.tag, str) and child.tag != ET.Comment:
                  has_non_comment_child = True
                  break # Found a real element, patch is not just comments

        if not has_non_comment_child:
             console.print("[dim]Patch check: Root <patch> element only contains comments (or is empty), considered empty.[/dim]")
             return True # Only comments found

        console.print("[dim]Patch check: Found non-comment elements, considered valid patch.[/dim]")
        return False # Found actual elements, not an empty/comment-only patch

    def _edit_suggest_edits(self):
        """Handler for asking the LLM for edit suggestions."""
        console.print(Panel("Edit Option: Ask LLM for Suggestions", style="cyan"))
        if self.book_root is None:
            return

        current_book_xml_str = ET.tostring(self.book_root, encoding="unicode")

        # --- Prompt 1: Get Suggestions ---
        prompt_suggest = f"""
You are an expert editor reviewing the novel manuscript provided below in XML format.
Analyze the entire book context (plot, pacing, character arcs, consistency, style, clarity, dialogue, descriptions, potential plot holes).
Identify potential areas for improvement.
Provide a numbered list of 5-10 concrete, actionable suggestions for specific edits. Keep suggestions concise (1-2 sentences each). Focus on high-impact changes. Reference specific chapter IDs or character IDs where possible.

Example Suggestion Format:
1. Strengthen the foreshadowing in Chapter 3 regarding the villain's true motives.
2. Improve the pacing of the chase scene in Chapter 12 by shortening descriptive paragraphs.
3. Make Character 'heroProtagonist' dialogue in Chapter 5 sound more hesitant to reflect their uncertainty.
4. Consider adding a brief scene in Chapter 7 showing Character 'sidekickFriend' reaction to the news.

Full Book Context:
```xml
{escape(current_book_xml_str)}
```

Generate the numbered list of edit suggestions now. Output ONLY the list. Do not include greetings or explanations.
"""
        suggestions_text = self._get_llm_response(
            prompt_suggest, "Generating edit suggestions list", allow_stream=False
        )

        if not suggestions_text or not suggestions_text.strip():
            console.print(f"[red]Could not get suggestions from the {self.api_type.upper()}.[/red]")
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
                f"[yellow]{self.api_type.upper()} response did not contain a parsable list of suggestions.[/yellow]"
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
        prompt_implement = f"""
You are an expert editor implementing a specific suggestion on the novel provided below in XML format.

Editing Suggestion to Implement: {escape(cleaned_suggestion)}

Guidelines for Patch:
- Analyze the entire book context provided.
- Generate an XML `<patch>` structure that implements the specific suggestion above.
- Patches can modify chapters or paragraphs.
    - Full chapter replace: `<patch><chapter id="..."><content><paragraph id="1">...</paragraph>...</content></chapter></patch>` (Ensure new para IDs are sequential from 1, ensure text).
    - Paragraph replace: `<patch><chapter id="..."><content-patch><paragraph id="para_id">...</paragraph></content-patch></chapter></patch>` (Ensure replacement para has text).
    - Top-level changes (Title, Synopsis, Characters): Use appropriate tags within `<patch>`. Characters list is usually replaced wholesale if the suggestion warrants it.
- Be specific and target the changes accurately based on the suggestion. Reference chapter/paragraph/character IDs accurately in the patch.
- Use XML comments `<!-- ... -->` within the patch ONLY if brief justification is absolutely needed for clarity. Avoid comments otherwise.
- If the suggestion cannot be directly translated to a patch (e.g., too vague, requires complex structural changes beyond simple replacement), output an empty patch `<patch/>` with a comment explaining why.
- Output ONLY the patch XML. Do not output any text before `<patch>` or after `</patch>`. Strictly adhere to XML format.

Full Book Context:
```xml
{escape(current_book_xml_str)}
```

Generate the `<patch>` XML to implement the suggestion now. Output ONLY the patch XML structure.
"""
        suggested_patch_xml_str_raw = self._get_llm_response(
            prompt_implement,
            f"Implementing Suggestion {chosen_num}",
            allow_stream=False, # Patch should be small enough for non-stream
        )

        if suggested_patch_xml_str_raw:
             # Clean the raw response *before* checking if it's empty or displaying
            cleaned_patch_str = clean_llm_xml_output(suggested_patch_xml_str_raw)

            console.print(
                Panel(
                    f"[bold cyan]Suggested Patch for Suggestion {chosen_num}:[/bold cyan]",
                    border_style="magenta",
                )
            )
            syntax = Syntax(
                cleaned_patch_str, "xml", theme="default", line_numbers=True
            )
            console.print(syntax)

            # Use the helper function to check if the *cleaned* patch is effectively empty
            if self._is_patch_effectively_empty(cleaned_patch_str):
                console.print(
                    f"[yellow]{self.api_type.upper()} indicated no changes needed or could not generate a valid patch for this suggestion.[/yellow]"
                )
            elif Confirm.ask(
                "\n[yellow]Apply this suggested patch? [/yellow]", default=True
            ):
                # Apply the cleaned string
                apply_success = self._apply_patch(cleaned_patch_str)
                self._handle_patch_result(
                    apply_success, f"Implement Suggestion {chosen_num}"
                )
            else:
                console.print("Suggested patch discarded.")
        else:
             # Handle case where LLM response itself was None/empty
             console.print(f"[red]Failed to get patch suggestion response from {self.api_type.upper()}.[/red]")

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

User's Editing Instructions: {escape(instructions)}

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
- Be specific and justify changes briefly ONLY if necessary using XML comments: `<!-- Suggestion: Rephrased for clarity -->`. Avoid comments otherwise.
- If no significant changes are needed based on the instructions, output an empty patch like `<patch/>` or a patch with only comments: `<patch><!-- No major changes suggested --></patch>`
- Output ONLY the patch XML. Do not output any text before `<patch>` or after `</patch>`. Strictly adhere to XML format.

Full Book Context:
```xml
{escape(current_book_xml_str)}
```

Generate the `<patch>` XML containing your suggested edits now. Output ONLY the patch XML structure.
"""

        suggested_patch_xml_str_raw = self._get_llm_response(
            prompt, "Generating general edit patch", allow_stream=False # Patch should be small
        )

        if suggested_patch_xml_str_raw:
            # Clean the raw response *before* checking if it's empty or displaying
            cleaned_patch_str = clean_llm_xml_output(suggested_patch_xml_str_raw)

            console.print(
                Panel(
                    "[bold cyan]Suggested Edits (Patch XML):[/bold cyan]",
                    border_style="magenta",
                )
            )
            syntax = Syntax(
                cleaned_patch_str, "xml", theme="default", line_numbers=True
            )
            console.print(syntax)

            # Use the helper function to check if the *cleaned* patch is effectively empty
            if self._is_patch_effectively_empty(cleaned_patch_str):
                console.print(
                    f"[cyan]{self.api_type.upper()} suggested no specific changes or only provided comments.[/cyan]"
                )
            elif Confirm.ask(
                "\n[yellow]Do you want to apply these suggested changes? [/yellow]",
                default=True,
            ):
                # Apply the cleaned string
                apply_success = self._apply_patch(cleaned_patch_str)
                self._handle_patch_result(apply_success, "General Edit")
            else:
                console.print("Suggested patch discarded.")
        else:
             # Handle case where LLM response itself was None/empty
             console.print(f"[red]Failed to get general edit patch response from {self.api_type.upper()}.[/red]")


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
            "6": ("Finish Editing", None),  # Use None to signal exit
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
            elif choice == "6":  # Finish Editing
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
             # Sort characters by name for display
            sorted_characters = sorted(characters, key=lambda char: char.findtext("name", "zzz").lower())

            chapters_raw = self.book_root.findall(".//chapter")
            # Sort chapters numerically for display/calculation
            sorted_chapters_display = []
            invalid_id_chapters = []
            for c in chapters_raw:
                try:
                    int(c.get("id", "NaN"))
                    sorted_chapters_display.append(c)
                except ValueError:
                    invalid_id_chapters.append(c)
            sorted_chapters_display.sort(key=lambda c: int(c.get("id")))
            chapters = sorted_chapters_display + invalid_id_chapters


            # Calculate word counts for HTML
            total_wc_html = 0
            chapter_wc_html = {}
            for chap in chapters: # Use the sorted list
                chap_id = chap.get("id")
                content = chap.find("content")
                chap_wc = 0
                if content is not None:
                    for para in content.findall(".//paragraph"):
                        chap_wc += _count_words(para.text)
                if chap_id:
                    chapter_wc_html[chap_id] = chap_wc
                    total_wc_html += chap_wc
                else:
                     total_wc_html += chap_wc # Add to total even if no ID

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
        .api-info {{ text-align: center; font-size: 0.8em; color: #888; margin-bottom: 20px; font-family: monospace; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="total-word-count">Total Word Count: {total_wc_html:,}</div>
    <div class="api-info">Generated/Edited using: API=[{escape(self.api_type)}], Model=[{escape(self.model_name)}]</div>

    <div class="synopsis">
        <h2>Synopsis</h2>
        <p>{synopsis}</p>
    </div>
{initial_idea_html}
    <div class="characters">
        <h2>Characters</h2>
        {f"<ul>{''.join(f'<li><b>{escape(c.findtext("name", "N/A"))}</b> (ID: {escape(c.get("id", "N/A"))}): {escape(c.findtext("description", "N/A"))}</li>' for c in sorted_characters)}</ul>" if sorted_characters else '<p class="missing-content">No characters defined.</p>'}
    </div>

    <hr>

    <div class="chapters">
        <h2>Chapters</h2>
"""
            if chapters:
                for chap in chapters: # Iterate through the sorted list
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

        # Client initialization is checked in __init__

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
            for c in chapters_element.findall("chapter")
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
    parser = argparse.ArgumentParser(
        description="Interactive Novel Writer using LLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
        )
    parser.add_argument(
        "--api",
        type=str.lower, # Convert to lowercase
        choices=['gemini', 'openai'],
        default='gemini', # Default to Gemini
        help="Which API provider to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None, # Default is decided based on API choice later
        help=f"Specific model name (e.g., '{DEFAULT_GEMINI_MODEL}', '{DEFAULT_OPENAI_MODEL}').",
    )
    parser.add_argument(
        "--openai_base_url",
        metavar="URL",
        type=str,
        default=None,
        help="Optional custom base URL for the OpenAI API (e.g., for local models like Ollama). Only used if --api is 'openai'.",
    )
    parser.add_argument(
        "--resume",
        metavar="FOLDER_PATH",
        help="Path to an existing book project folder to resume.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        metavar="FILE_PATH",
        help="Path to a text file containing the initial book idea/prompt (used only for new books).",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # --- Set Model Defaults ---
    if args.model is None:
        if args.api == 'gemini':
            args.model = DEFAULT_GEMINI_MODEL
        elif args.api == 'openai':
            args.model = DEFAULT_OPENAI_MODEL
        # Add more API defaults here if needed
        console.print(f"[dim]No model specified, defaulting to '{args.model}' for the '{args.api}' API.[/dim]")

    # Check if base URL is provided when not using OpenAI API
    if args.openai_base_url and args.api != 'openai':
        console.print(f"[yellow]Warning: --openai_base_url ('{args.openai_base_url}') is provided but --api is '{args.api}'. The base URL will be ignored.[/yellow]")
        args.openai_base_url = None # Ignore it


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
            api_type=args.api,
            model_name=args.model,
            openai_base_url=args.openai_base_url, # Pass base URL
            resume_folder_name=resume_folder_path,
            initial_prompt_file=initial_prompt_file,
        )
        writer.run()
    except FileNotFoundError as fnf_error:
        # This might catch errors during loading within __init__ if path was valid initially but files are missing
        console.print(
            f"[bold red]File Not Found Error during setup/loading: {fnf_error}[/bold red]"
        )
    except (ValueError, RuntimeError) as init_error:
        # Catch API key missing error or other init value/runtime errors (like client init failure)
        console.print(f"[bold red]Initialization Error: {init_error}[/bold red]")
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]Operation interrupted by user. Exiting.[/bold yellow]"
        )
    except ImportError as import_err:
         if 'openai' in str(import_err).lower():
              console.print(f"[bold red]Import Error: {import_err}. Have you installed the OpenAI library? Try: pip install openai[/bold red]")
         elif 'google.generativeai' in str(import_err).lower():
               console.print(f"[bold red]Import Error: {import_err}. Have you installed the Google Generative AI library? Try: pip install google-generativeai[/bold red]")
         else:
              console.print(f"[bold red]Import Error: {import_err}. Please ensure all required libraries are installed.[/bold red]")
    except Exception as main_exception:
        console.print(
            "[bold red]An unexpected critical error occurred during execution:[/bold red]"
        )
        console.print_exception(
            show_locals=False, word_wrap=True, exc_info=main_exception
        )  # Rich traceback

