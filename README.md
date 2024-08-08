# GenQA: Document QA Extraction Tool

This tool extracts question-answer pairs from documents using a local LLM. It chunks documents, generates QA pairs, and collects the results.

## Installation

Install directly from GitHub:

```
pip install git+https://github.com/sbetko/genqa.git
```

## Usage

GenQA works in two steps: extraction and collection.

### Step 1: Extract QA Pairs

```
python -m genqa.extract <input_files> --output_dir <output_directory> [options]
```

This processes documents and saves QA pairs as JSON files, preserving context and structure.

Options:
- `--temperature`: LLM temperature (default: 0.0)
- `--n_ctx`: Max context length (default: 16384)
- `--chunk_size`: Max text chunk size (default: 4096)
- `--max_questions`: Max questions per chunk (default: 3)
- `--overwrite`: Overwrite existing files

Example:
```
python -m genqa.extract docs/*.pdf docs/*.docx docs/*.html --output_dir qa_results --temperature 0.1 --max_questions 3
```

### Step 2: Collect Results

```
python -m genqa.make_csv <input_directory> <output_csv_file>
```

This compiles all QA pairs from JSON files into a single CSV file for easier analysis.

Example:
```
python -m genqa.make_csv qa_results output_qa_pairs.csv
```

This two-step process lets you process documents in batches and review outputs before final collection.

### Document Conversion

There's also a standalone converter for DOCX, HTML, and PDF to Markdown:

```
python -m genqa.convert <input_file> > output.md
```

## Performance and Memory

### Context Length

The model supports up to 128,000 tokens, but defaults to 16,384 to balance context and memory use. Adjust with `--n_ctx` if needed.

### Chunk Size

The default chunk size fits instructions, content, and output within the max context. If generation halts due to context overflow, the script will retry with higher temperature or skip the chunk.

### GPU Acceleration

GPU support significantly speeds up processing. Installation instructions for GPU acceleration coming soon.

## Supported Formats

- DOCX
- HTML
- PDF

## Output Formats

### JSON (genqa.extract)

Each document gets a JSON file with:
- `source_filepath`: Original document path
- `markdown_text`: Full document text in Markdown
- `chunks`: Array of text chunks, each with:
  - `chunk_text`: Chunk content
  - `qa_pairs`: Array of QA pairs (question, answer, supporting_quotes)

### CSV (genqa.make_csv)

The CSV contains:
- `file_path`: Original document path
- `chunk_number`: Chunk number in document
- `qa_number`: QA pair number in chunk
- `question`: Generated question
- `answer`: Generated answer
- `supporting_quotes`: Relevant quotes from chunk, delimited by pipes ("`|`")

## Model

Uses Phi-3.1-mini-128k-instruct. The GGUF file auto-downloads to `~/.cache/huggingface/hub` on first run.