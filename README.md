# StudyMate AI

A RAG-based study assistant. Students ask questions about lecture materials; the system retrieves relevant passages and answers **only** from them, shows its sources, and says so when the lectures do not contain the answer.

Course: Computers and Society (Dr. Sherif M. Tawfik)

## Tech stack
- Python, LangChain
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector DB: Chroma (cosine distance)
- LLM: Google Gemini (model set in `src/config.py`)
- UI: Streamlit

## Architecture
Document flow: lecture files -> loader -> chunker -> embeddings -> Chroma
Question flow: question -> embedding -> top-k search -> prompt -> Gemini -> answer + sources

![Architecture](docs/architecture.png)

## Project structure
```
src/        config, loader, chunker, embeddings, vectorstore, retriever, prompts, generator, pipeline
app.py      Streamlit interface
ingest.py   builds the vector index (run once)
evaluate.py runs the test questions
eval/       questions.csv and results
data/       lecture files, one folder per subject (not in git)
```

## Setup
1. Clone and enter the repo:
   ```
   git clone https://github.com/salma-alaa182/studymate-ai.git
   cd studymate-ai
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root (get a key from https://aistudio.google.com):
   ```
   GOOGLE_API_KEY=your_key_here
   ```
5. Put lecture files (`.pdf`, `.pptx`, `.txt`) in `data/<subject>/`, for example:
   ```
   data/computer_science/
   data/ai/
   data/databases/
   data/software_engineering/
   ```

## Run
Build the index (once, and again whenever the data or chunk settings change):
```
python ingest.py
```
Start the app:
```
streamlit run app.py
```
Command-line test:
```
python -m src.pipeline
```

## Evaluation
Questions live in `eval/questions.csv` (columns: question, expected_answer, source_file, answerable).
```
python evaluate.py                        # baseline
python evaluate.py --top-k 6 --tag k6     # experiment
```
Results are saved to `eval/results_<tag>.csv`. Relevance, faithfulness and correctness are scored manually.

## How hallucination is limited
- The prompt tells the model to use only the numbered context passages.
- If the closest chunk is farther than `NO_EVIDENCE_DISTANCE` (cosine distance), the LLM is not called and the system answers that it has not enough information.
- Sources (file, page, passage) are shown with every answer.

## Example
Question: What is digital transformation?
Answer and sources: see `docs/screenshots/`.

## Known limitations
- Short slide titles can rank above the slide that holds the definition.
- The Gemini model used ignores the `temperature` setting, so wording can vary between runs.
- Requires an internet connection for the LLM call.

## Team
Add member names and roles here.