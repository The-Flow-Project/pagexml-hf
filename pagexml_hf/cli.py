"""
Command-line interface for transkribus-hf.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
import traceback
from datasets import get_dataset_config_names

from .converter import XmlConverter
from .parser import XmlParser
from .logger import init_debug_logger, init_info_logger


class SourcePathAction(argparse.Action):
    """Custom action to handle source path validation."""

    def __call__(self, parser, namespace, value, option_string=None):
        token = getattr(namespace, 'token', None) or os.getenv('HF_TOKEN')

        if value.startswith('http://') or value.startswith('https://'):
            # Case SourcePath is Zip-URL
            setattr(namespace, self.dest, (value, 'zip_url'))
            return
        elif '/' in value and len(value.split('/')) == 2:
            # Case Source Path is Huggingface Dataset
            try:
                configs = get_dataset_config_names(value, token=token)
                if not configs:
                    raise ValueError()
            except Exception as e:
                if not token:
                    parser.error(
                        f"Hugging Face dataset '{value}' not found or inaccessible: {e} (did you set a token?)"
                    )
                else:
                    parser.error(
                        f"Hugging Face dataset '{value}' not found or inaccessible: {e}"
                    )

            setattr(namespace, self.dest, (value, 'huggingface'))
            return
        else:
            # Case Source Path is Folder
            path = Path(value)
            if path.exists():
                if path.is_dir():
                    setattr(namespace, self.dest, (str(path), 'local'))
                else:
                    setattr(namespace, self.dest, (str(path), 'zip'))
                return

        parser.error(f"Invalid source path: {value}. Must be a local path, a ZIP-URL, or a HuggingFace dataset ID.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert Transkribus ZIP files or XML files from Folder to HuggingFace datasets"
    )

    parser.add_argument(
        "source_path",
        action=SourcePathAction,
        help="Path to the Transkribus ZIP file (local or URL), Huggingface Dataset, \
            or folder containing XML files and images",
    )

    parser.add_argument(
        "--mode",
        choices=["raw_xml", "text", "region", "line", "window"],
        default="text",
        help="Export mode (default: text)",
    )

    # Changed to recognizing the namespace automatically
    '''
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Namespace of the Page XML files (default: empty string)",
    )
    '''

    # ################ - window mode arguments - ####################
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
    # ################ - end window mode arguments - ####################

    # ################ - huggingface arguments - ####################
    parser.add_argument(
        "--repo-id", help="HuggingFace repository ID to upload the dataset (e.g., username/dataset-name)"
    )

    parser.add_argument(
        "--token", help="HuggingFace token (or set HF_TOKEN environment variable)"
    )

    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    # ################ - end huggingface arguments - ####################

    # ################ - split arguments - ####################
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
    # ################ - end split arguments - ####################

    # ################ - operational arguments - ####################
    parser.add_argument(
        "--get-stats",
        action="store_true",
        help="Show statistics after converting/upload (default: False)",
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
        "--batchsize",
        type=int,
        default=32,
        help="Number of files per batch (default: 32, make it higher for line/window mode).",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing HuggingFace Hub dataset (only for uploads, not --local-only). "
             "If True, checks feature compatibility before processing. (default: False)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode (default: False)",
    )
    # ################ - end operational arguments - ####################

    # ################ - region/line mode arguments - ####################
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
    # ################ - end region/line mode arguments - ####################

    args = parser.parse_args()
    source_path, source_type = args.source_path

    if args.debug:
        logger = init_debug_logger()
    else:
        logger = init_info_logger()

    logger.info("Process started via CLI")
    logger.info("Source path: {}".format(source_path))
    logger.info("Source type: {}".format(source_type))
    logger.info("#" * 60)

    # ################ - validate parameters - ####################
    if not args.local_only and not args.repo_id:
        logger.error("Error: --repo-id is required unless using --local-only")
        sys.exit(1)

    if args.min_width is not None and args.min_width <= 0:
        logger.error("Error: --min-width has to be a positive integer")
        sys.exit(1)

    if args.min_height is not None and args.min_height <= 0:
        logger.error("Error: --min-height has to be a positive integer")
        sys.exit(1)

    # Validate window parameters
    if args.mode == "window":
        if args.window_size < 1:
            logger.error("Error: --window-size must be at least 1")
            sys.exit(1)
        if args.overlap < 0:
            logger.error("Error: --overlap cannot be negative")
            sys.exit(1)
        if args.overlap >= args.window_size:
            logger.error("Error: --overlap must be less than --window-size")
            sys.exit(1)
    # ################ - end validate parameters - ####################

    # Initialize parser
    xmlparser = XmlParser()

    parse_xml: bool = False
    if args.mode in ["line", "region", "text", "window"]:
        parse_xml = True

    # Get back generator functions and kwargs depending on source type
    if source_type == 'local':
        gen_func = xmlparser.parse_folder
        gen_kwargs = {"folder_path": source_path, "parse_xml": parse_xml}
    elif source_type in ['zip', 'zip_url']:
        gen_func = xmlparser.parse_zip
        gen_kwargs = {'zip_path': source_path, 'parse_xml': parse_xml}
    elif source_type == 'huggingface':
        gen_func = xmlparser.parse_dataset
        gen_kwargs = {'dataset': source_path, 'token': args.token, 'parse_xml': parse_xml}
    else:
        logger.error("Error: Unsupported source")
        sys.exit(1)

    logger.debug("Got generator function and kwargs")
    if (gen_func and gen_kwargs) is None:
        logger.error("Error: No generator function and/or kwargs available")
        sys.exit(1)

    # Initialize converter
    logger.info(f"Creating converter")
    converter = XmlConverter(
        gen_func=gen_func,
        gen_kwargs=gen_kwargs,
        source_path=source_path,
        source_type=source_type,
    )

    try:
        if args.local_only:
            # Convert the dataset
            dataset = converter.convert(
                export_mode=args.mode,
                window_size=args.window_size,
                overlap=args.overlap,
                batch_size=args.batchsize,
                split_train=args.split_train,
                split_seed=args.split_seed,
                split_shuffle=args.split_shuffle,
                mask_crop=args.mask_crop,
                min_width=args.min_width,
                min_height=args.min_height,
                allow_empty=args.allow_empty,
            )
            logger.debug(f"Converted dataset: {dataset.info}")

            # Save locally
            mode_suffix = f"_{args.mode}"
            if args.mode == "window":
                mode_suffix += f"_w{args.window_size}_o{args.overlap}"

            output_dir = args.output_dir or f"./pagexml_dataset{mode_suffix}"
            logger.info(f"Preparing to save dataset locally to: {output_dir}")

            if os.path.exists(output_dir):
                logger.warning(f"Output directory {output_dir} exists. Overwriting...")
                shutil.rmtree(output_dir)

            # Dataset saved as arrow
            logger.debug(f"Saving dataset to {output_dir} with save_to_disk()")
            dataset.save_to_disk(output_dir)

            logger.info(f"Dataset saved sucessfully to: {output_dir}")
        else:
            # Upload to HuggingFace Hub
            repo_url = converter.convert_and_upload(
                repo_id=args.repo_id,
                export_mode=args.mode,
                token=args.token,
                private=args.private,
                allow_empty=args.allow_empty,
                append=args.append,
                batch_size=args.batchsize,
                split_train=args.split_train,
                split_seed=args.split_seed,
                split_shuffle=args.split_shuffle,
                mask_crop=args.mask_crop,
                min_width=args.min_width,
                min_height=args.min_height,
                window_size=args.window_size,
                overlap=args.overlap,
            )
            logger.info(f"Success! Dataset available at: {repo_url}")

        # Show statistics if requested
        if args.get_stats:
            stats = converter.stats_cache
            logger.debug(f"Stats: {stats}")
            print("Dataset Statistics:")
            print(f"  Total pages_generator: {stats['total_pages']}")
            print(f"  Projects: {', '.join(stats['projects'])}")
            if args.mode in ["line", "region", "text", "window"]:
                print(f"  Total regions: {stats['total_regions']}")
                print(f"  Total lines: {stats['total_lines']}")
                print(f"  Avg regions per page: {stats['avg_regions_per_page']:.1f}")
                print(f"  Avg lines per page: {stats['avg_lines_per_page']:.1f}")

            # For window mode, show expected window count
            if args.mode == "window":
                step = args.window_size - args.overlap
                estimated_windows = 0
                regions_with_lines = stats["total_lines"] // 5
                estimated_windows += (
                        max(0, regions_with_lines - args.window_size + 1) // step
                )
                logger.info(
                    f"  Estimated windows (window_size={args.window_size}, "
                    f"overlap={args.overlap}): ~{estimated_windows}"
                )

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
