import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


def normalize_domain(value: object) -> str:
	if value is None:
		return ""

	domain = str(value).strip().lower()
	if not domain:
		return ""

	domain = re.sub(r"^https?://", "", domain)
	domain = re.sub(r"^www\.", "", domain)
	domain = domain.split("/")[0].split("?")[0].split("#")[0]
	domain = domain.split(":")[0]
	return domain.strip()


def get_first_dataset_split(dataset: DatasetDict | Dataset) -> Dataset:
	if isinstance(dataset, Dataset):
		return dataset
	first_split = next(iter(dataset.keys()))
	return dataset[first_split]


def get_first_iterable_split(
	dataset: IterableDatasetDict | IterableDataset,
) -> IterableDataset:
	if isinstance(dataset, IterableDataset):
		return dataset
	first_split = next(iter(dataset.keys()))
	return dataset[first_split]


def load_reliability_frame() -> pd.DataFrame:
	reliability_dataset = load_dataset("sergioburdisso/news_media_reliability")
	reliability_split = get_first_dataset_split(reliability_dataset)
	reliability_df = reliability_split.to_pandas()

	if "domain" not in reliability_df.columns:
		raise ValueError("Expected 'domain' column in news_media_reliability dataset.")

	reliability_df["_join_domain"] = reliability_df["domain"].map(normalize_domain)
	reliability_df = reliability_df[reliability_df["_join_domain"] != ""].copy()
	reliability_df = reliability_df.drop_duplicates(subset=["_join_domain"], keep="first")
	return reliability_df


def clear_previous_outputs(output_dir: Path, output_prefix: str) -> None:
	for file_path in output_dir.glob(f"{output_prefix}_part_*.csv"):
		file_path.unlink()


def write_joined_shards(
	joined_df: pd.DataFrame,
	output_dir: Path,
	output_prefix: str,
	rows_per_file: int,
	file_index: int,
	rows_in_current_file: int,
) -> tuple[int, int, int]:
	rows_written = 0
	start = 0

	while start < len(joined_df):
		if rows_in_current_file >= rows_per_file:
			file_index += 1
			rows_in_current_file = 0

		chunk_capacity = rows_per_file - rows_in_current_file
		chunk = joined_df.iloc[start : start + chunk_capacity]
		output_file = output_dir / f"{output_prefix}_part_{file_index:04d}.csv"
		write_header = rows_in_current_file == 0

		chunk.to_csv(output_file, mode="a", header=write_header, index=False)

		written_now = len(chunk)
		rows_in_current_file += written_now
		rows_written += written_now
		start += written_now

	return file_index, rows_in_current_file, rows_written


def process_and_write_batch(
	batch_rows: List[Dict[str, object]],
	reliability_df: pd.DataFrame,
	output_dir: Path,
	output_prefix: str,
	rows_per_file: int,
	file_index: int,
	rows_in_current_file: int,
) -> tuple[int, int, int]:
	ccnews_batch_df = pd.DataFrame(batch_rows)
	if "publisher" not in ccnews_batch_df.columns:
		raise ValueError("Expected 'publisher' column in ccnews dataset.")

	ccnews_batch_df["_join_domain"] = ccnews_batch_df["publisher"].map(normalize_domain)
	joined_df = ccnews_batch_df.merge(
		reliability_df,
		on="_join_domain",
		how="left",
		suffixes=("", "_reliability"),
	)

	return write_joined_shards(
		joined_df=joined_df,
		output_dir=output_dir,
		output_prefix=output_prefix,
		rows_per_file=rows_per_file,
		file_index=file_index,
		rows_in_current_file=rows_in_current_file,
	)


def run(
	sample_size: int,
	batch_size: int,
	rows_per_file: int,
	output_dir: Path,
	output_prefix: str,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	clear_previous_outputs(output_dir, output_prefix)

	reliability_df = load_reliability_frame()
	print(f"Loaded reliability rows: {len(reliability_df)}")

	ccnews_dataset = load_dataset(
		"stanford-oval/ccnews",
		"default",
		split="train",
		streaming=True,
	)
	ccnews_iterable = get_first_iterable_split(ccnews_dataset)

	batch_rows: List[Dict[str, object]] = []
	total_processed = 0
	total_written = 0
	file_index = 1
	rows_in_current_file = 0

	for row in ccnews_iterable:
		if total_processed >= sample_size:
			break

		batch_rows.append(row)
		total_processed += 1

		if len(batch_rows) >= batch_size:
			file_index, rows_in_current_file, written = process_and_write_batch(
				batch_rows=batch_rows,
				reliability_df=reliability_df,
				output_dir=output_dir,
				output_prefix=output_prefix,
				rows_per_file=rows_per_file,
				file_index=file_index,
				rows_in_current_file=rows_in_current_file,
			)
			total_written += written
			print(f"Processed rows: {total_processed} | Written rows: {total_written}")
			batch_rows = []

	if batch_rows:
		file_index, rows_in_current_file, written = process_and_write_batch(
			batch_rows=batch_rows,
			reliability_df=reliability_df,
			output_dir=output_dir,
			output_prefix=output_prefix,
			rows_per_file=rows_per_file,
			file_index=file_index,
			rows_in_current_file=rows_in_current_file,
		)
		total_written += written

	print("Completed.")
	print(f"CCNews rows processed: {total_processed}")
	print(f"Joined rows written: {total_written}")
	print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Join ccnews with news_media_reliability on publisher/domain and write CSV output."
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=100_000,
		help="Number of ccnews rows to process.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=5_000,
		help="Number of ccnews rows per in-memory processing batch.",
	)
	parser.add_argument(
		"--rows-per-file",
		type=int,
		default=100_000,
		help="Maximum number of rows per output CSV shard.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs"),
		help="Directory where CSV files will be written.",
	)
	parser.add_argument(
		"--output-prefix",
		type=str,
		default="joined_ccnews_reliability",
		help="Prefix for output CSV shard names.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		sample_size=args.sample_size,
		batch_size=args.batch_size,
		rows_per_file=args.rows_per_file,
		output_dir=args.output_dir,
		output_prefix=args.output_prefix,
	)

