
# 📚 pulpgen: Your AI Ghostwriter - Draft Novels Instantly! ✍️

---

**Tired of the blank page? Wish you had a tireless co-writer?** `pulpgen` is your personal AI agent that transforms your story idea into a complete first-draft novel! Choose your AI powerhouse (**Google Gemini** or any **OpenAI-compatible API**, including local models!), guide the outlining process, let the AI handle the heavy lifting of chapter writing, and then jump in with powerful, AI-assisted interactive editing tools.

Go from concept to ~60,000+ word draft faster than ever before! 🚀

![screenshot](screenshot.png)

---

## ✨ Key Features

*   **Choose Your AI Brain:** Works with **Google Gemini** models *or* any **OpenAI-compatible API** (including local LLMs via Ollama/LMStudio or services like OpenRouter).
*   **Idea to Outline:** Provide a simple prompt, and the AI generates a title, synopsis, character sketches, and detailed chapter summaries.
*   **Automatic First Draft:** The AI systematically writes the prose for each chapter based on the outline.
*   **Interactive AI Editing Suite:** Step in and collaborate!
    *   Expand chapters ("Make Longer").
    *   Rewrite sections with specific instructions.
    *   Get fresh perspectives with "blackout" rewrites.
    *   Ask the AI for high-level editing suggestions.
    *   Request general fixes via AI-generated patches.
*   **Flexible Workflow:** Start new projects or easily resume existing ones.
*   **Organized Output:** Creates a project folder with XML state files (outline, patches, final) and readable HTML versions (`latest.html`, `version-NN.html`, `final.html`).

---

## 🚀 Quick Start

Get your AI co-writer running in minutes!

1.  **Prerequisites:**
    *   Python 3.10 or newer.
    *   An API Key for either Google Gemini or an OpenAI compatible service.

2.  **Installation:**
    ```bash
    # Clone or download the repository
    git clone https://your-repo-url/pulpgen.git # Replace with actual URL if applicable
    cd pulpgen

    # Install required libraries using pip
    pip install google-generativeai openai python-dotenv rich

    # (Alternative using uv, if you have it)
    # uv sync
    ```

3.  **API Keys:**
    *   Create a file named `.env` in the `pulpgen` directory.
    *   Add the keys for the APIs you plan to use (you only need the one you'll select via `--api`):
        ```dotenv
        GEMINI_API_KEY=YOUR_GEMINI_KEY_HERE
        OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
        ```
    *   Get keys:
        *   Gemini: <https://aistudio.google.com/app/apikey>
        *   OpenAI: <https://platform.openai.com/api-keys>
    *   *(If keys aren't in `.env`, the script will prompt you securely)*

4.  **Run Your First Project!**

    *   **Simplest (Default: Gemini, Interactive Prompt):**
        ```bash
        python pulpgen.py
        ```

    *   **Using OpenAI (Default: GPT-4o, Interactive Prompt):**
        ```bash
        python pulpgen.py --api openai
        ```

    *   **Using a Prompt File (with Gemini):**
        ```bash
        python pulpgen.py --prompt my_cool_idea.txt
        ```

    *   **Using a Specific Model (e.g., Gemini Flash):**
        ```bash
        python pulpgen.py --api gemini --model gemini-1.5-flash-latest --prompt my_idea.txt
        ```

    *   **Using a Local Model via Ollama (OpenAI API):**
        *(Make sure Ollama is running and has the model pulled)*
        ```bash
        python pulpgen.py --api openai --model 'llama3' --openai_base_url 'http://localhost:11434/v1' --prompt my_idea.txt
        ```

    *   **Using OpenRouter (OpenAI API):**
        *(Set OPENAI_API_KEY to your OpenRouter key in .env)*
        ```bash
        python pulpgen.py --api openai --model 'mistralai/mixtral-8x7b-instruct' --openai_base_url 'https://openrouter.ai/api/v1' --prompt my_idea.txt
        ```

5.  **Resume a Project:**
    ```bash
    # Specify the API you used previously if not Gemini
    python pulpgen.py --resume path/to/your/project_folder
    python pulpgen.py --api openai --resume path/to/your/openai_project_folder
    ```

---

## 💡 Why Use Pulpgen?

*   **Beat Writer's Block:** Generate a complete draft to get momentum.
*   **Rapid Prototyping:** Quickly flesh out different story ideas.
*   **Focus on the Big Picture:** Let the AI handle the initial prose generation while you focus on plot, character, and revision.
*   **AI Collaboration:** Use the interactive editing tools as a powerful brainstorming and revision partner.
*   **Flexibility:** Choose the AI model and API that best suits your needs and budget (including free local models!).

---

## ⚙️ How It Works (The Process)

`pulpgen` acts like an AI project manager, guiding the language model through distinct phases:

1.  **Phase 1: Architecting the Story (Outline Generation)**
    *   **Your Input:** Book idea/prompt, approximate chapter count.
    *   **AI Action:** Creates title, synopsis, character profiles (`<characters>`), and detailed chapter summaries (`<chapters>`).
    *   **Result:** Saves the blueprint to `outline.xml`.

2.  **Phase 2: Laying the Foundation (Content Drafting)**
    *   **AI Action:** Works through the outline, selecting chapters needing content. Writes the full prose for these chapters, using the summary and existing story context.
    *   **Result:** Saves chapter content in sequential `patch-NN.xml` files. Updates `version-NN.html` and `latest.html` previews.

3.  **Phase 3: Renovation & Polish (Interactive Editing)**
    *   **Your Role:** You take the reins! Use the menu to direct AI revisions:
        *   *Make Longer:* Expand chapter word counts.
        *   *Rewrite (Instructions):* Rework chapters based on your specific notes.
        *   *Rewrite (Fresh):* Get a new take on a chapter, ignoring existing content.
        *   *Suggest Edits:* Ask the AI for overall improvement ideas.
        *   *General Request:* Give broad instructions for the AI to implement.
    *   **AI Action:** Generates suggested changes as XML patches.
    *   **Your Decision:** Review the AI's proposed patch and confirm whether to apply it.
    *   **Result:** Applied changes are saved in new `patch-NN.xml` files, and HTML previews are updated.

4.  **Phase 4: The Final Print (Finalization)**
    *   **AI Action:** Consolidates the original outline and all applied patches.
    *   **Result:** Produces the complete manuscript as `final.xml` and the reader-friendly `final.html`.

---

## 🗂️ Project Structure (Inside `YYYYMMDD-slug-uuid/`)

Your project folder keeps everything organized:

*   **`outline.xml`:** The initial plan (metadata, characters, chapter summaries).
*   **`patch-NN.xml`:** Step-by-step records of generated content or edits. Essential for resuming work.
*   **`final.xml`:** The complete final draft in structured XML.
*   **`final.html`:** Clean, readable HTML version of the final book.
*   **`version-NN.html`:** HTML snapshots after each patch save (content/editing steps).
*   **`latest.html`:** Always the most up-to-date HTML preview.

---

## 🛠️ Command-Line Options

```bash
usage: pulpgen.py [-h] [--api {gemini,openai}] [--model MODEL_NAME] [--openai_base_url URL] [--resume FOLDER_PATH] [--prompt FILE_PATH]

Interactive Novel Writer using LLMs.

options:
  -h, --help            show this help message and exit
  --api {gemini,openai}
                        Which API provider to use. (default: gemini)
  --model MODEL_NAME    Specific model name (e.g., 'gemini-1.5-pro-latest', 'gpt-4o'). Defaults vary by API. (default: None)
  --openai_base_url URL
                        Optional custom base URL for OpenAI compatible APIs (e.g., for local models, proxies). Only used if --api is 'openai'. (default: None)
  --resume FOLDER_PATH  Path to an existing book project folder to resume. (default: None)
  --prompt FILE_PATH    Path to a .txt file containing the initial book idea/prompt (used only for new books). (default: None)
```

---

## 🏆 Tips for Best Results

1.  **Strong Concept:** A clear, compelling initial idea gives the AI the best direction. Spend time on your prompt!
2.  **Experiment with AI:** Try both Gemini and different OpenAI models/endpoints. See which one best fits your style. Use the `--model` flag!
3.  **Guide the Edit:** Phase 3 is crucial. Don't just accept every AI suggestion. Use the tools to refine the draft towards *your* vision. Treat the AI like an assistant, not the sole author.
4.  **It's a *First* Draft:** `pulpgen` excels at speed and getting content down. Expect to do final human polishing on the output.
5.  **Watch for AIisms:** Look out for repetition, inconsistencies, or odd phrasing during editing.
6.  **Review `latest.html`:** Frequently check the latest HTML preview during the process to catch issues early.
7.  **Iterate:** Don't be afraid to run `pulpgen` multiple times with tweaked prompts or different AI settings if the first attempt isn't perfect.

---

## 🧑‍💻 Advanced Use / Configuration

*   **API Keys:** Securely managed via `.env` or environment variables.
*   **Models/API:** Easily select via `--api`, `--model`, and `--openai_base_url` flags.
*   **Generation Tweaks:** Constants inside `pulpgen.py` like `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`, etc., control AI behavior. `DEFAULT_MAX_OUTPUT_TOKENS` is set to `65536`, but actual output depends on the specific model and API limits. Modify these carefully.

## License

MIT License. See the [LICENSE](LICENSE) file for details.

---

**Ready to write? Fire up `pulpgen` and bring your story to life!**

![pulpgen Cover](cover.webp)
*A PULP FICTION ADVENTURE*
```

