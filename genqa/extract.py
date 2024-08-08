import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import semchunk
from genqa.convert import DocToMarkdown
from llama_cpp import Llama
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


qa_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "supporting_quotes": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["question", "supporting_quotes", "answer"],
    },
}


def token_count(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))


def generate_qa_pairs(
    llm: Llama,
    text: str,
    max_questions: int,
    initial_temperature: float = 0.0,
    max_retries: int = 3,
    temperature_increment: float = 0.1,
) -> List[Dict[str, Any]]:
    prompt = f"""
    Generate a list of 1-{max_questions} question-answer pairs based on the following text. Adhere to these guidelines:
    1. Focus on quality over quantity.
    2. Phrase questions so they could be used as search queries to find the source document.
    3. Include a verbatim list of supporting quotations from the text for each answer.
    4. Ensure answers are relevant to the questions and use only information from the given text.

    Text: {text}

    Respond in JSON format like this:
    [
        {{
            "question": "Question text here",
            "answer": "Answer text here"
            "supporting_quotes": ["Quote 1", "Quote 2", ...],
        }},
        ...
    ]
    """

    temperature = initial_temperature
    for attempt in range(max_retries):
        try:
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates question-answer pairs from given text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object", "schema": qa_schema},
            )

            output = json.loads(response["choices"][0]["message"]["content"])
            return output

        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Failed to generate QA pairs: {str(e)}")
            if attempt == max_retries - 1:
                raise
            temperature += temperature_increment
            logging.info(f"Retrying with temperature {temperature}")

    return None


def process_chunk(
    llm: Llama,
    chunk: str,
    max_questions: int,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    try:
        qa_pairs = generate_qa_pairs(llm, chunk, max_questions, temperature)
        return {"chunk_text": chunk, "qa_pairs": qa_pairs if qa_pairs else []}
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return {"chunk_text": chunk, "qa_pairs": [], "error": str(e)}


def process_file(
    file_path: str,
    output_dir: Path,
    llm: Llama,
    chunk_size: int,
    max_questions: int,
    temperature: float = 0.0,
    overwrite: bool = False,
) -> None:
    try:
        converter = DocToMarkdown()
        text = converter.convert(file_path)
        file_path = Path(file_path)

        if text is None:
            logging.error(f"Failed to extract text from file: {file_path}")
            return

        chunker = semchunk.chunkerify(lambda t: token_count(llm, t), chunk_size)
        chunks = list(chunker(text))

        output_file = output_dir / f"{file_path.stem}_qa.json"

        if output_file.exists() and not overwrite:
            with open(output_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            if len(result["chunks"]) == len(chunks):
                logging.info(f"File {file_path} already processed. Skipping.")
                return
            else:
                logging.info(f"Resuming processing for {file_path}")
        else:
            result = {
                "source_filepath": str(file_path),
                "markdown_text": text,
                "chunks": [],
            }

        with tqdm(
            total=len(chunks), desc=f"Processing {file_path.name}", position=0, leave=True
        ) as pbar:
            pbar.update(len(result["chunks"]))
            for i, chunk in enumerate(chunks[len(result["chunks"]) :], start=len(result["chunks"])):
                chunk_result = process_chunk(llm, chunk, max_questions, temperature)
                result["chunks"].append(chunk_result)

                # Write intermediate results after each chunk
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                pbar.update(1)

        logging.info(f"Completed processing file: {file_path}")
        logging.info(f"Results written to {output_file}")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Process files and generate QA pairs.")
    parser.add_argument(
        "input",
        nargs="+",
        help="Files to process",
    )
    parser.add_argument("--output_dir", default="qa_result", help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
    parser.add_argument(
        "--n_ctx", type=int, default=16384, help="Maximum context length for the model"
    )
    parser.add_argument("--chunk_size", type=int, default=4096, help="Maximum size of text chunks")
    parser.add_argument(
        "--max_questions",
        type=int,
        default=3,
        help="Maximum number of questions per chunk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    llm = Llama.from_pretrained(
        repo_id="bartowski/Phi-3.1-mini-128k-instruct-GGUF",
        filename="*Q4_K_M.gguf",
        n_ctx=args.n_ctx,
        verbose=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with logging_redirect_tqdm():
        for file_path in tqdm(args.input, desc="Processing files", position=0, leave=True):
            process_file(
                file_path,
                output_dir,
                llm,
                args.chunk_size,
                args.max_questions,
                temperature=args.temperature,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
