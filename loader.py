import argparse
import re
from typing import cast
from typing import Iterable
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


DATE_FIELD_CANDIDATES = [
	"published_date",
	"publication_date",
	"publish_date",
	"published_at",
	"date",
]
TITLE_FIELD_CANDIDATES = ["title", "headline"]
TEXT_FIELD_CANDIDATES = ["plain_text", "text", "content", "article_text", "body"]
PUBLISHER_FIELD_CANDIDATES = ["publisher", "source", "domain", "source_domain"]
# Keywords that are specific enough to match anywhere in title or body.
NICE_TRUCK_ATTACK_KEYWORDS_ANYWHERE = [
	"nice truck",
	"nice attack",
	"attack in nice",
	"massacre in nice",
	"shooting in nice",
	"nice massacre",
	"nice terror",
	"truck attack in nice",
	"lorry attack in nice",
	"nice lorry",
	"lorry in nice",
	"promenade des anglais",
	"bastille day attack",
	"bastille day truck",
	"bastille day lorry",
	"bastille day massacre",
	"14 juillet attack",
	"14 juillet attentat",
	"attentat de nice",
	"terror attack in nice",
	"terrorist attack in nice",
]

# Keywords that are too generic for body-text matching; only matched against the title.
NICE_TRUCK_ATTACK_KEYWORDS_TITLE_ONLY = [
	"truck ploughed",
	"truck plowed",
	"lorry ploughed",
	"lorry plowed",
	"truck rammed",
	"lorry rammed",
	"nice, france",
]


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


def pick_first_field(available_fields: Iterable[str], candidates: list[str]) -> str | None:
	available_set = set(available_fields)
	for candidate in candidates:
		if candidate in available_set:
			return candidate
	return None


def parse_publish_datetime(value: object) -> pd.Timestamp | None:
	if value is None:
		return None

	timestamp = pd.to_datetime(str(value), errors="coerce", utc=True)
	if pd.isna(timestamp):
		return None
	return cast(pd.Timestamp, timestamp)


def as_lower_text(value: object) -> str:
	if value is None:
		return ""
	return str(value).strip().lower()


def is_nice_truck_attack_related(
	row: dict[str, object],
	title_field: str | None,
	text_field: str | None,
) -> bool:
	title_text = as_lower_text(row.get(title_field)) if title_field else ""
	body_text = as_lower_text(row.get(text_field)) if text_field else ""
	full_text = f"{title_text}\n{body_text}"
	if not full_text.strip():
		return False

	if any(kw in full_text for kw in NICE_TRUCK_ATTACK_KEYWORDS_ANYWHERE):
		return True

	if any(kw in title_text for kw in NICE_TRUCK_ATTACK_KEYWORDS_TITLE_ONLY):
		return True

	return False


def resolve_publisher_value(row: dict[str, object], publisher_field: str | None) -> str:
	if publisher_field:
		publisher_value = row.get(publisher_field)
		if publisher_value:
			return str(publisher_value)

	for url_field in ("requested_url", "responded_url", "url"):
		url_value = row.get(url_field)
		if url_value:
			parsed = urlparse(str(url_value))
			return parsed.netloc

	return ""


def load_reliability_frame() -> pd.DataFrame:
	reliability_dataset = load_dataset("sergioburdisso/news_media_reliability")
	reliability_split = get_first_dataset_split(reliability_dataset)
	reliability_df = cast(pd.DataFrame, reliability_split.to_pandas())

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


def run(
	sample_size: int,
	publish_cutoff: str,
	rows_per_file: int,
	output_dir: Path,
	output_prefix: str,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	clear_previous_outputs(output_dir, output_prefix)
	cutoff_ts = pd.Timestamp(publish_cutoff, tz="UTC")

	reliability_df = load_reliability_frame()
	print(f"Loaded reliability rows: {len(reliability_df)}")

	ccnews_dataset = load_dataset(
		"stanford-oval/ccnews",
		"2016",
		split="train",
		streaming=True,
	)
	ccnews_iterable = get_first_iterable_split(ccnews_dataset)

	date_field: str | None = None
	title_field: str | None = None
	text_field: str | None = None
	publisher_field: str | None = None

	total_scanned = 0
	total_after_date_filter = 0
	total_after_topic_filter = 0
	filtered_rows: list[dict[str, object]] = []

	print("Streaming CCNews 2016 and filtering for post-June Nice truck attack coverage...")
	for row in ccnews_iterable:
		if sample_size > 0 and total_scanned >= sample_size:
			break

		total_scanned += 1

		if date_field is None:
			available_fields = row.keys()
			date_field = pick_first_field(available_fields, DATE_FIELD_CANDIDATES)
			title_field = pick_first_field(available_fields, TITLE_FIELD_CANDIDATES)
			text_field = pick_first_field(available_fields, TEXT_FIELD_CANDIDATES)
			publisher_field = pick_first_field(available_fields, PUBLISHER_FIELD_CANDIDATES)

			if date_field is None:
				raise ValueError(
					"Could not resolve publication date field from ccnews rows. "
					f"Tried: {DATE_FIELD_CANDIDATES}"
				)
			if title_field is None and text_field is None:
				raise ValueError(
					"Could not resolve article text/title fields from ccnews rows. "
					f"Tried title fields: {TITLE_FIELD_CANDIDATES}; "
					f"text fields: {TEXT_FIELD_CANDIDATES}"
				)

			print(
				"Resolved fields -> "
				f"date: {date_field}, title: {title_field}, text: {text_field}, "
				f"publisher: {publisher_field or 'url-derived fallback'}"
			)

		publish_ts = parse_publish_datetime(row.get(date_field))
		if publish_ts is None or publish_ts < cutoff_ts:
			continue
		total_after_date_filter += 1

		if not is_nice_truck_attack_related(row, title_field=title_field, text_field=text_field):
			continue
		total_after_topic_filter += 1

		row_copy = dict(row)
		row_copy["_publish_dt"] = publish_ts
		row_copy["_publisher_resolved"] = resolve_publisher_value(
			row,
			publisher_field=publisher_field,
		)
		filtered_rows.append(row_copy)

		if total_scanned % 100_000 == 0:
			print(
				f"Scanned: {total_scanned} | After date filter: {total_after_date_filter} "
				f"| After topic filter: {total_after_topic_filter}"
			)

	if not filtered_rows:
		print("Completed. No CCNews rows matched date/topic filters.")
		print(f"CCNews rows scanned: {total_scanned}")
		print(f"Output directory: {output_dir}")
		return

	filtered_df = pd.DataFrame(filtered_rows)
	filtered_df["_join_domain"] = filtered_df["_publisher_resolved"].map(normalize_domain)

	joined_df = filtered_df.merge(
		reliability_df,
		on="_join_domain",
		how="left",
		suffixes=("", "_reliability"),
	)

	if "newsguard_score" not in joined_df.columns:
		raise ValueError(
			"Expected 'newsguard_score' column in joined data from "
			"news_media_reliability dataset."
		)

	joined_df = joined_df.dropna(subset=["newsguard_score"]).copy()
	joined_df = joined_df.sort_values(by="_publish_dt", ascending=True, kind="stable")
	joined_df["publish_datetime_utc"] = pd.to_datetime(
		joined_df["_publish_dt"],
		utc=True,
	).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

	file_index, rows_in_current_file, total_written = write_joined_shards(
		joined_df=joined_df,
		output_dir=output_dir,
		output_prefix=output_prefix,
		rows_per_file=rows_per_file,
		file_index=1,
		rows_in_current_file=0,
	)
	_ = file_index
	_ = rows_in_current_file

	print("Completed.")
	print(f"CCNews rows scanned: {total_scanned}")
	print(f"Rows after date filter: {total_after_date_filter}")
	print(f"Rows after Nice-attack topic filter: {total_after_topic_filter}")
	print(f"Rows with non-null reliability score: {len(joined_df)}")
	print(f"Joined rows written: {total_written}")
	print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Stream ccnews 2016, keep post-June Nice truck attack related rows, "
			"left-join with news_media_reliability, drop rows without score, and write CSV output."
		)
	)
	parser.add_argument(
		"--sample-size",
		type=int,
		default=0,
		help="Maximum number of ccnews rows to scan. Use 0 to scan all rows.",
	)
	parser.add_argument(
		"--publish-cutoff",
		type=str,
		default="2016-07-14",
		help="Keep only rows with publication date on/after this value (UTC).",
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
		publish_cutoff=args.publish_cutoff,
		rows_per_file=args.rows_per_file,
		output_dir=args.output_dir,
		output_prefix=args.output_prefix,
	)

