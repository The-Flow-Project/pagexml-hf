"""
Command-line interface for transkribus-hf.
"""

import argparse
import os
import sys
from pathlib import Path
from datasets import get_dataset_config_names
from datasets.utils.logging import disable_progress_bar, enable_progress_bar

from .converter import XmlConverter
from .parser import XmlParser

class SourcePathAction(argparse.Action):
    """Custom action to handle source path validation."""
    disable_progress_bar()

    def __call__(self, parser, namespace, value, option_string=None):
        token = getattr(namespace, 'token', None) or os.getenv('HF_TOKEN')
        path = Path(value)
        if path.exists():
            if path.is_dir():
                setattr(namespace, self.dest, (str(path), 'local'))
            else:
                setattr(namespace, self.dest, (str(path), 'zip'))
            return

        if '/' in value and len(value.split('/')) == 2:
            if not token:
                parser.error('HuggingFace token is required for remote dataset input.')
            try:
                configs = get_dataset_config_names(value)
                enable_progress_bar()
                if not configs:
                    raise ValueError()
            except Exception as e:
                parser.error(f"Hugging Face dataset '{value}' not found or inaccessible: {e}")

            setattr(namespace, self.dest, (value, 'huggingface'))
            return

        parser.error(f"Invalid source path: {value}. Must be a local path or a HuggingFace dataset ID.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Transkribus ZIP files or XML files from Folder to HuggingFace datasets"
    )

    parser.add_argument(
        "source_path",
        action=SourcePathAction,
        help="Path to the Transkribus ZIP file or folder containing XML files and images",
    )

    parser.add_argument(
        "--mode",
        choices=["raw_xml", "text", "region", "line", "window"],
        default="text",
        help="Export mode (default: text)",
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Namespace of the Page XML files (default: empty string)",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=2,
        help="Number of lines per window (only for window mode, default: 2)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of lines to overlap between windows (only for window mode, default: 0)",
    )

    parser.add_argument(
        "--repo-id", help="HuggingFace repository ID (e.g., username/dataset-name)"
    )

    parser.add_argument(
        "--token", help="HuggingFace token (or set HF_TOKEN environment variable)"
    )

    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )

    parser.add_argument(
        "--split-train",
        type=float,
        default=None,
        help="Split ratio for train split."
             "Between 0 and 1, default: None (means no split),"
             "e.g. 0.8 for 80%% train, 20%% test",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )

    parser.add_argument(
        "--split-shuffle",
        action="store_true",
        help="Shuffle the dataset before splitting (default: False)",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, don't convert or upload",
    )

    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Convert but don't upload to HuggingFace Hub",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the dataset locally (when using --local-only)",
    )

    parser.add_argument(
        "--mask-crop",
        action="store_true",
        help="Crop the mask from the image (default: False)",
    )

    parser.add_argument(
        "--min-width",
        type=int,
        default=None,
        help="Minimum width of the cropped region/line (default: None, no minimum)",
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=None,
        help="Minimum height of the cropped region/line (default: None, no minimum)",
    )

    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty regions or lines (default: False)",
    )

    args = parser.parse_args()
    source_path, source_type = args.source_path

    if not args.stats_only and not args.local_only and not args.repo_id:
        print("Error: --repo-id is required unless using --stats-only or --local-only")
        sys.exit(1)

    if args.min_width is not None and args.min_width <= 0:
        print("Error: --min-width has to be a positive integer")
        sys.exit(1)

    if args.min_height is not None and args.min_height <= 0:
        print("Error: --min-height has to be a positive integer")
        sys.exit(1)

    # Validate window parameters
    if args.mode == "window":
        if args.window_size < 1:
            print("Error: --window-size must be at least 1")
            sys.exit(1)
        if args.overlap < 0:
            print("Error: --overlap cannot be negative")
            sys.exit(1)
        if args.overlap >= args.window_size:
            print("Error: --overlap must be less than --window-size")
            sys.exit(1)

    # Initialize parser
    parser = XmlParser(namespace=args.namespace)

    if source_type == 'local':
        pages = parser.parse_folder(source_path)
    elif source_type == 'zip':
        pages = parser.parse_zip(source_path)
    elif source_type == 'huggingface':
        pages = parser.parse_dataset(source_path)
    else:
        print(f"Error: Unsupported source")
        sys.exit(1)

    # Initialize converter
    converter = XmlConverter(
        pages=pages,
        source_path=source_path,
        source_type=source_type,
    )

    # Show statistics if requested
    if args.stats_only:
        stats = converter.get_stats()
        print("Dataset Statistics:")
        print(f"  Total pages: {stats['total_pages']}")
        print(f"  Total regions: {stats['total_regions']}")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Projects: {', '.join(stats['projects'])}")
        print(f"  Avg regions per page: {stats['avg_regions_per_page']:.1f}")
        print(f"  Avg lines per page: {stats['avg_lines_per_page']:.1f}")

        # For window mode, show expected window count
        if args.mode == "window":
            step = args.window_size - args.overlap
            estimated_windows = 0
            for _ in ["total_regions"]:  # This is a rough estimate
                regions_with_lines = stats["total_lines"] // 5  # rough estimate
                estimated_windows += (
                        max(0, regions_with_lines - args.window_size + 1) // step
                )
            print(
                f"  Estimated windows (window_size={args.window_size}, overlap={args.overlap}): ~{estimated_windows}"
            )

        return

    try:
        # Convert the dataset
        dataset = converter.convert(
            export_mode=args.mode,
            window_size=args.window_size,
            overlap=args.overlap,
            split_train=args.split_train,
            split_seed=args.split_seed,
            split_shuffle=args.split_shuffle,
            mask_crop=args.mask_crop,
            min_width=args.min_width,
            min_height=args.min_height,
            allow_empty=args.allow_empty,
        )

        if args.local_only:
            # Save locally
            mode_suffix = f"_{args.mode}"
            if args.mode == "window":
                mode_suffix += f"_w{args.window_size}_o{args.overlap}"
            output_dir = args.output_dir or f"./pagexml_dataset{mode_suffix}"
            dataset.save_to_disk(output_dir)
            print(f"Dataset saved to: {output_dir}")
        else:
            # Upload to HuggingFace Hub
            repo_url = converter.upload_to_hub(
                dataset=dataset,
                repo_id=args.repo_id,
                token=args.token,
                private=args.private,
            )
            print(f"Success! Dataset available at: {repo_url}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
