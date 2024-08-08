import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, Iterator
from tqdm import tqdm

def process_qa_file(file_path: Path) -> Iterator[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for chunk_number, chunk in enumerate(data['chunks'], start=1):
        for qa_number, qa_pair in enumerate(chunk['qa_pairs'], start=1):
            yield {
                'file_path': data['source_filepath'],
                'chunk_number': chunk_number,
                'qa_number': qa_number,
                'question': qa_pair['question'],
                'answer': qa_pair['answer'],
                'supporting_quotes': ' | '.join(qa_pair['supporting_quotes'])
            }

def process_directory(input_dir: Path, csv_writer: csv.DictWriter):
    json_files = list(input_dir.glob('*_qa.json'))
    for file in tqdm(json_files, desc="Processing files"):
        for row in process_qa_file(file):
            csv_writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Convert QA JSON outputs to CSV")
    parser.add_argument("input_dir", help="Directory containing QA JSON files")
    parser.add_argument("output_file", help="Path to the output CSV file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    fieldnames = ['file_path', 'chunk_number', 'qa_number', 'question', 'answer', 'supporting_quotes']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        process_directory(input_dir, writer)

    print(f"CSV file has been created: {output_file}")

if __name__ == "__main__":
    main()