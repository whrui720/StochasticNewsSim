import argparse
import json
import re
import tomllib
from dataclasses import dataclass, field
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


@dataclass
class EventConfig:
	"""All event-specific parameters loaded from a TOML config file."""

	name: str
	ccnews_year: str
	publish_cutoff: str
	output_prefix: str
	keywords_anywhere: list[str] = field(default_factory=list)
	keywords_title_only: list[str] = field(default_factory=list)


def load_event_config(path: Path) -> EventConfig:
	with open(path, "rb") as f:
		data = tomllib.load(f)

	event = data.get("event", {})
	keywords = data.get("keywords", {})

	required = {"name", "ccnews_year", "publish_cutoff", "output_prefix"}
	missing = required - set(event.keys())
	if missing:
		raise ValueError(
			f"Event config {path} is missing required [event] keys: {sorted(missing)}"
		)

	return EventConfig(
		name=event["name"],
		ccnews_year=str(event["ccnews_year"]),
		publish_cutoff=event["publish_cutoff"],
		output_prefix=event["output_prefix"],
		keywords_anywhere=keywords.get("anywhere", []),
		keywords_title_only=keywords.get("title_only", []),
	)


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


def is_event_related(
	row: dict[str, object],
	title_field: str | None,
	text_field: str | None,
	keywords_anywhere: list[str],
	keywords_title_only: list[str],
) -> bool:
	title_text = as_lower_text(row.get(title_field)) if title_field else ""
	body_text = as_lower_text(row.get(text_field)) if text_field else ""
	full_text = f"{title_text}\n{body_text}"
	if not full_text.strip():
		return False

	if any(kw in full_text for kw in keywords_anywhere):
		return True

	if any(kw in title_text for kw in keywords_title_only):
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


def get_checkpoint_path(output_dir: Path, output_prefix: str) -> Path:
	return output_dir / f"{output_prefix}.checkpoint.json"


def load_checkpoint(path: Path) -> dict[str, object] | None:
	if not path.exists():
		return None

	with open(path, "r", encoding="utf-8") as f:
		checkpoint = json.load(f)

	if not isinstance(checkpoint, dict):
		raise ValueError(f"Checkpoint file {path} is not a JSON object.")

	return checkpoint


def save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
	tmp_path = path.with_suffix(path.suffix + ".tmp")
	with open(tmp_path, "w", encoding="utf-8") as f:
		json.dump(checkpoint, f, indent=2, sort_keys=True)
		f.write("\n")
	tmp_path.replace(path)


def checkpoint_int(checkpoint: dict[str, object], key: str) -> int:
	value = checkpoint.get(key)
	if not isinstance(value, int):
		raise ValueError(f"Checkpoint key '{key}' is missing or not an integer.")
	return value


def build_checkpoint_data(
	event_config: EventConfig,
	publish_cutoff: str,
	output_prefix: str,
	total_scanned: int,
	total_after_date_filter: int,
	total_after_topic_filter: int,
	total_written: int,
	file_index: int,
	rows_in_current_file: int,
) -> dict[str, object]:
	return {
		"version": 1,
		"event_name": event_config.name,
		"ccnews_year": event_config.ccnews_year,
		"publish_cutoff": publish_cutoff,
		"output_prefix": output_prefix,
		"total_scanned": total_scanned,
		"total_after_date_filter": total_after_date_filter,
		"total_after_topic_filter": total_after_topic_filter,
		"total_written": total_written,
		"file_index": file_index,
		"rows_in_current_file": rows_in_current_file,
	}


def flush_filtered_rows(
	filtered_rows: list[dict[str, object]],
	reliability_df: pd.DataFrame,
	output_dir: Path,
	output_prefix: str,
	rows_per_file: int,
	file_index: int,
	rows_in_current_file: int,
) -> tuple[int, int, int]:
	if not filtered_rows:
		return file_index, rows_in_current_file, 0

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
	if joined_df.empty:
		return file_index, rows_in_current_file, 0

	# Sort each flushed chunk for readability; full-dataset sort would require buffering all rows.
	joined_df = joined_df.sort_values(by="_publish_dt", ascending=True, kind="stable")
	joined_df["publish_datetime_utc"] = pd.to_datetime(
		joined_df["_publish_dt"],
		utc=True,
	).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

	return write_joined_shards(
		joined_df=joined_df,
		output_dir=output_dir,
		output_prefix=output_prefix,
		rows_per_file=rows_per_file,
		file_index=file_index,
		rows_in_current_file=rows_in_current_file,
	)


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
	event_config: EventConfig,
	sample_size: int,
	publish_cutoff: str,
	rows_per_file: int,
	flush_every: int,
	resume: bool,
	output_dir: Path,
	output_prefix: str,
) -> None:
	if flush_every <= 0:
		raise ValueError("--flush-every must be greater than 0.")

	output_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = get_checkpoint_path(output_dir, output_prefix)

	total_scanned = 0
	total_after_date_filter = 0
	total_after_topic_filter = 0
	total_written = 0
	file_index = 1
	rows_in_current_file = 0
	resume_skip_remaining = 0

	if resume:
		checkpoint = load_checkpoint(checkpoint_path)
		if checkpoint is None:
			raise ValueError(
				f"Cannot resume: checkpoint file not found at {checkpoint_path}."
			)

		if checkpoint.get("ccnews_year") != event_config.ccnews_year:
			raise ValueError(
				"Checkpoint ccnews_year does not match current event config. "
				"Use a matching config or run without --resume."
			)
		if checkpoint.get("publish_cutoff") != publish_cutoff:
			raise ValueError(
				"Checkpoint publish_cutoff does not match current run settings. "
				"Use the same cutoff or run without --resume."
			)
		if checkpoint.get("output_prefix") != output_prefix:
			raise ValueError(
				"Checkpoint output_prefix does not match current run settings. "
				"Use the same prefix or run without --resume."
			)

		total_scanned = checkpoint_int(checkpoint, "total_scanned")
		total_after_date_filter = checkpoint_int(checkpoint, "total_after_date_filter")
		total_after_topic_filter = checkpoint_int(checkpoint, "total_after_topic_filter")
		total_written = checkpoint_int(checkpoint, "total_written")
		file_index = checkpoint_int(checkpoint, "file_index")
		rows_in_current_file = checkpoint_int(checkpoint, "rows_in_current_file")
		resume_skip_remaining = total_scanned

		print(f"Resuming from checkpoint: {checkpoint_path}")
		print(f"Rows previously scanned: {total_scanned}")
		print(f"Rows previously written: {total_written}")
	else:
		clear_previous_outputs(output_dir, output_prefix)
		if checkpoint_path.exists():
			checkpoint_path.unlink()

	cutoff_ts = pd.Timestamp(publish_cutoff, tz="UTC")

	reliability_df = load_reliability_frame()
	print(f"Loaded reliability rows: {len(reliability_df)}")

	print(f"Event: {event_config.name}")
	print(f"CCNews year: {event_config.ccnews_year}")
	print(f"Publish cutoff: {publish_cutoff}")
	print(
		f"Keywords: {len(event_config.keywords_anywhere)} anywhere, "
		f"{len(event_config.keywords_title_only)} title-only"
	)

	ccnews_dataset = load_dataset(
		"stanford-oval/ccnews",
		event_config.ccnews_year,
		split="train",
		streaming=True,
	)
	ccnews_iterable = get_first_iterable_split(ccnews_dataset)

	date_field: str | None = None
	title_field: str | None = None
	text_field: str | None = None
	publisher_field: str | None = None
	filtered_rows_buffer: list[dict[str, object]] = []

	print(f"Streaming CCNews {event_config.ccnews_year} and filtering for event coverage...")
	for row in ccnews_iterable:
		if sample_size > 0 and total_scanned >= sample_size:
			break

		if resume_skip_remaining > 0:
			resume_skip_remaining -= 1
			continue

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

		if not is_event_related(
			row,
			title_field=title_field,
			text_field=text_field,
			keywords_anywhere=event_config.keywords_anywhere,
			keywords_title_only=event_config.keywords_title_only,
		):
			continue
		total_after_topic_filter += 1

		row_copy = dict(row)
		row_copy["_publish_dt"] = publish_ts
		row_copy["_publisher_resolved"] = resolve_publisher_value(
			row,
			publisher_field=publisher_field,
		)
		filtered_rows_buffer.append(row_copy)

		if len(filtered_rows_buffer) >= flush_every:
			file_index, rows_in_current_file, rows_written_now = flush_filtered_rows(
				filtered_rows=filtered_rows_buffer,
				reliability_df=reliability_df,
				output_dir=output_dir,
				output_prefix=output_prefix,
				rows_per_file=rows_per_file,
				file_index=file_index,
				rows_in_current_file=rows_in_current_file,
			)
			total_written += rows_written_now
			filtered_rows_buffer.clear()
			save_checkpoint(
				checkpoint_path,
				build_checkpoint_data(
					event_config=event_config,
					publish_cutoff=publish_cutoff,
					output_prefix=output_prefix,
					total_scanned=total_scanned,
					total_after_date_filter=total_after_date_filter,
					total_after_topic_filter=total_after_topic_filter,
					total_written=total_written,
					file_index=file_index,
					rows_in_current_file=rows_in_current_file,
				),
			)

		if total_scanned % 100_000 == 0:
			print(
				f"Scanned: {total_scanned} | After date filter: {total_after_date_filter} "
				f"| After topic filter: {total_after_topic_filter}"
			)
			save_checkpoint(
				checkpoint_path,
				build_checkpoint_data(
					event_config=event_config,
					publish_cutoff=publish_cutoff,
					output_prefix=output_prefix,
					total_scanned=total_scanned,
					total_after_date_filter=total_after_date_filter,
					total_after_topic_filter=total_after_topic_filter,
					total_written=total_written,
					file_index=file_index,
					rows_in_current_file=rows_in_current_file,
				),
			)

	file_index, rows_in_current_file, rows_written_now = flush_filtered_rows(
		filtered_rows=filtered_rows_buffer,
		reliability_df=reliability_df,
		output_dir=output_dir,
		output_prefix=output_prefix,
		rows_per_file=rows_per_file,
		file_index=file_index,
		rows_in_current_file=rows_in_current_file,
	)
	total_written += rows_written_now

	if checkpoint_path.exists():
		checkpoint_path.unlink()

	print("Completed.")
	print(f"CCNews rows scanned: {total_scanned}")
	print(f"Rows after date filter: {total_after_date_filter}")
	print(f"Rows after topic filter: {total_after_topic_filter}")
	print(f"Joined rows written: {total_written}")
	print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Stream a CCNews year, filter by event keywords from a TOML config, "
			"left-join with news_media_reliability, drop rows without score, and write CSV output."
		)
	)
	parser.add_argument(
		"--event-config",
		type=Path,
		required=True,
		help="Path to a TOML event config file (see events/ directory for examples).",
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
		default=None,
		help="Override the publish cutoff date from the event config (UTC, e.g. 2016-07-14).",
	)
	parser.add_argument(
		"--rows-per-file",
		type=int,
		default=100_000,
		help="Maximum number of rows per output CSV shard.",
	)
	parser.add_argument(
		"--flush-every",
		type=int,
		default=5_000,
		help=(
			"Flush every N event-matching rows to output CSV and checkpoint file "
			"for crash-safe progress."
		),
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help=(
			"Resume from a previous interrupted run by loading the checkpoint and "
			"continuing output writes."
		),
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
		default=None,
		help="Override the output prefix from the event config.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	event_config = load_event_config(args.event_config)

	# CLI flags override config file values when explicitly provided.
	publish_cutoff = args.publish_cutoff or event_config.publish_cutoff
	output_prefix = args.output_prefix or event_config.output_prefix

	run(
		event_config=event_config,
		sample_size=args.sample_size,
		publish_cutoff=publish_cutoff,
		rows_per_file=args.rows_per_file,
		flush_every=args.flush_every,
		resume=args.resume,
		output_dir=args.output_dir,
		output_prefix=output_prefix,
	)
