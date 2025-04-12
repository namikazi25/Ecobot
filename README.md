# EcoBot

**EcoBot is an AI-powered ecological assistant that helps with species identification, ecological analysis, and scientific research.** Built upon a modular agentic workflow and advanced models like BioTrove-CLIP, it integrates powerful multi-modal tools for meaningful, context-aware ecological insights.

![EcoBot Demo](screenshot-ecobot.jpeg)

---

## Features

- **Agentic Workflow**: Modular decision pipeline (Planner → Evaluator → Executor) for reliable and transparent reasoning
- **Biological Image Classification**: Uses BioTrove-CLIP for zero-shot species identification
- **Multi-Modal Input**: Supports image, PDF, and text queries
- **Taxonomic Verification**: Integrates Wikipedia API for scientific validation
- **Research Paper Analysis**: Extracts and summarizes key findings from research PDFs
- **Conversation History**: Maintains context across user sessions

---

## Architecture

The EcoBot workflow consists of three main agents:

- **Planning Agent (`planner.py`)**: Analyzes user intent and selects suitable tools
- **Evaluating Agent (`evaluator.py`)**: Validates tool selection and plan logic
- **Executing Agent (`executor.py`)**: Runs tools such as:
  - `wiki_tool.py` (Wikipedia lookup)
  - `image_tools.py` (BioTrove-CLIP, GPT-4o Vision)
  - `pdf_tools.py` (PDF extraction and Q&A)

### Pipeline

```
User Input → Planner → Evaluator → Executor → Response
    |                                             ^
    +---------------------------------------------+
```

---

## Setup and Usage

### Prerequisites
- Python 3.9 or higher
- OpenAI API key

### Setup
1. **Clone the repository**
    ```bash
    git clone https://github.com/namikazi25/ecobot.git
    cd ecobot
    ```
2. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure your environment**
    ```bash
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    ```

### Launch
- **Start backend**
    ```bash
    cd backend
    uvicorn main:app --reload
    ```
- **Start frontend (Streamlit)**
    ```bash
    streamlit run app.py
    ```

---

## Directory Overview

```
assets/          # Sample images and PDFs
backend/
├── agents/      # Core agent logic (planner, evaluator, executor)
├── tools/       # Domain tools (image, pdf, wiki)
├── utils/       # Shared helpers
app.py           # Streamlit UI
requirements.txt
README.md
```

---

## Contributing

We welcome all contributions—developers, ecologists, and enthusiasts alike!

- Open issues for bugs or improvements
- Discuss new features or workflow enhancements
- Submit PRs (first-timers welcome!)
- See `backend/tools/` to extend capabilities

---

## License
EcoBot is released under the MIT License.

---

_Empowering ecological exploration through intelligent agentic workflows._
