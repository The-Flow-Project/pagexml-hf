"""
Main converter class for Transkribus to HuggingFace datasets.
"""

from pathlib import Path
from typing import Dict, Any
from loguru import logger

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Image as DatasetImage,
    Value,
    List,
    load_dataset,
    disable_caching,
)
from huggingface_hub import repo_exists

from .exporters import (
    BaseExporter,
    RawXMLExporter,
    TextExporter,
    RegionExporter,
    LineExporter,
    WindowExporter,
)

from .hub_utils import HubUploader

# Disable dataset caching to force rebuilding each time
disable_caching()


class XmlConverter:
    """
    Main converter class for converting Transkribus
    ZIP/XML-folder files to HuggingFace datasets.
    """
    POLYGON_FEATURE = BaseExporter.POLYGON_FEATURE

    PRE_FEATURES = Features(
        {
            "image": DatasetImage(decode=False),
            "xml_content": Value("string"),
            "filename": Value("string"),
            "project_name": Value("string"),
            "image_width": Value("int64"),
            "image_height": Value("int64"),

            "regions": List(feature={
                "id": Value("string"),
                "type": Value("string"),
                "coords": POLYGON_FEATURE,

                "text_lines": List(feature={
                    "id": Value("string"),
                    "text": Value("string"),
                    "coords": POLYGON_FEATURE,
                    "baseline": POLYGON_FEATURE,
                    "reading_order": Value("int64"),
                    "region_id": Value("string"),
                }),

                "reading_order": Value("int64"),
                "full_text": Value("string"),
            }),
        }
    )

    EXPORT_MODES = {
        "raw_xml": RawXMLExporter,
        "text": TextExporter,
        "region": RegionExporter,
        "line": LineExporter,
        "window": WindowExporter,
    }

    def __init__(
            self,
            gen_func,
            gen_kwargs: Dict[str, Any],
            source_path: str | None = None,
            source_type: str | None = None,
    ):
        """
        Initialize the converter.

        Args:
            gen_func: Function to convert raw XML files to HuggingFace datasets.
            gen_kwargs: Kwargs to pass to gen_func
            source_path (str, optional): Source path for the data to convert.
            source_type (str, optional): Source type for the data to convert.
        """

        self.gen_func = gen_func
        self.gen_kwargs = gen_kwargs
        self.exporter = None

        # Metadata
        if source_type in ['huggingface', 'zip_url'] and source_path is not None:
            self.source_name = source_path
        elif source_type in ['zip', 'local'] and source_path is not None:
            self.source_name = Path(source_path).name
        else:
            self.source_name = 'unknown_source'

        self.stats_cache = None

    @staticmethod
    def generation_wrapper(gen_func, gen_kwargs):
        """
        Wrapper function to call the generator with the provided kwargs.
        """
        yield from gen_func(**gen_kwargs)

    def _create_base_dataset(self) -> Dataset:
        """
        Creates initial raw dataset from the generator.
        Writes the data to disk (Arrow format) to prevent OOM.
        """
        logger.debug(f"gen_func: {self.gen_func}")

        ds = Dataset.from_generator(
            generator=self.generation_wrapper,
            gen_kwargs={"gen_func": self.gen_func, "gen_kwargs": self.gen_kwargs},
            features=self.PRE_FEATURES,
            keep_in_memory=True,
            num_proc=1,
        )
        logger.debug(f"Created dataset ({ds.info}).")
        return ds

    def _compute_stats(self, dataset: Dataset):
        projects = set()
        if "project_name" in dataset.column_names and dataset["project_name"] is not None:
            for project in dataset["project_name"]:
                projects.add(project)
        total_pages = len(set(dataset["filename"]))
        logger.debug(f"Columns of dataset: {dataset.column_names}")

        if "region_id" in dataset.column_names and dataset["region_id"] is not None:
            filename_region = zip(dataset["filename"], dataset["region_id"])
            logger.info("Computing statistics for regions.")
            total_unique_regions = len(set(filename_region))
            total_regions = len(set(dataset["region_id"]))
            logger.info("Computing statistics for lines.")
            if "line_id" in dataset.column_names and dataset["line_id"] is not None:
                filename_lineid = zip(dataset["filename"], dataset["line_id"])
                total_lines = len(dataset)
                logger.info(f"Total lines: {total_lines}")
                total_unique_lines = len(set(filename_lineid))
                logger.info(f"Total unique lines: {total_unique_lines}")
            else:
                total_lines = 0
                total_unique_lines = 0
        else:
            logger.info("Computing statistics for pages (no regions found).")
            total_regions = 0
            total_unique_regions = 0
            total_lines = 0
            total_unique_lines = 0

        self.stats_cache = {
            "total_pages": total_pages,
            "total_regions": total_regions,
            "total_unique_regions": total_unique_regions,
            "total_lines": total_lines,
            "total_unique_lines": total_unique_lines,
            "projects": list(projects),
            "avg_regions_per_page": total_unique_regions / total_pages if total_pages > 0 else 0,
            "avg_lines_per_page": total_unique_lines / total_pages if total_pages > 0 else 0,
            "avg_lines_per_region": total_unique_lines / total_unique_regions if total_unique_regions > 0 else 0,
        }

    def _check_exporter_feature_compatibility(
            self,
            repo_id: str,
            export_mode: str,
            token: str | None = None,
    ) -> None:
        """
        Check if the exporter's post_features are compatible with existing repo.
        This is called BEFORE data processing to save time.

        Args:
            repo_id: Repository ID to check against
            export_mode: Export mode to get the expected features
            token: HuggingFace token

        Raises:
            ValueError: If features are incompatible
        """

        # Check if repo exists
        try:
            exists = repo_exists(repo_id=repo_id, repo_type="dataset", token=token)
            if not exists:
                logger.info(f"Repository {repo_id} does not exist yet - no feature check needed")
                return
        except Exception as e:
            logger.warning(f"Could not check if repo exists: {e}")
            return

        # Get expected features from exporter
        exporter_class = self.EXPORT_MODES[export_mode]
        expected_features = exporter_class.POST_FEATURES

        logger.info(f"Checking feature compatibility with {repo_id} before processing data...")

        # Load existing dataset features
        try:
            existing_ds = load_dataset(repo_id, split="train", streaming=True)
            existing_features = getattr(existing_ds, "features", None)

            if existing_features:
                if str(expected_features) != str(existing_features):
                    logger.error("Feature mismatch detected!")
                    logger.error(f"Existing features: {existing_features}")
                    logger.error(f"Expected features (from {export_mode} exporter): {expected_features}")
                    raise ValueError(
                        f"Feature mismatch! The {export_mode} exporter produces features that "
                        f"don't match the existing dataset.\n"
                        f"Existing: {existing_features}\n"
                        f"Expected: {expected_features}\n"
                        f"Use append=False to overwrite the dataset or choose a different export_mode."
                    )
                logger.info("✓ Features are compatible with existing dataset")
        except ValueError:
            # Re-raise ValueError - this is a critical error that must stop processing
            raise
        except Exception as e:
            logger.warning(f"Could not verify features: {e}")
            logger.info("Proceeding with data processing (feature check inconclusive)")

    def convert(
            self,
            export_mode: str = "text",
            window_size: int = 2,
            overlap: int = 0,
            batch_size: int = 32,
            split_train: float = 1.0,
            split_seed: int = 42,
            split_shuffle: bool = False,
            mask_crop: bool = False,
            min_width: int = 0,
            min_height: int = 0,
            allow_empty: bool = False,
            line_augment: int = 0,
    ) -> Dataset | DatasetDict:
        """
        Convert parsed data to a HuggingFace dataset.

        Args:
            export_mode: Export mode ('raw_xml', 'text', 'region', 'line', 'window')
            window_size: Number of lines per window (only for window mode)
            overlap: Number of lines to overlap between windows (only for window mode)
            batch_size: Batch size for dataset mapping
            split_train: Split train set size (between 0 and 1, e.g. 0.8 for 80% train, 20% test)
            split_seed: Random seed for train/test split and augmentation
            split_shuffle: Whether to shuffle the dataset before splitting
            mask_crop: Whether to crop the mask to polygon (only for region and line mode)
            min_width: Minimum width of the regions/lines to be processed \
                (only for region and line mode)
            min_height: Minimum height of the regions/lines to be processed \
                (only for region and line mode)
            allow_empty: Whether to allow empty elements in the output (default: False)
            line_augment: Amount of random augmentation iterations on the original image (default: None)
        Returns:
            HuggingFace Dataset
        """
        if export_mode not in self.EXPORT_MODES:
            raise ValueError(
                f"Invalid export mode: {export_mode}. \
                    Available modes: {list(self.EXPORT_MODES.keys())}"
            )

        base_dataset = self._create_base_dataset()
        logger.debug("#" * 80)
        logger.debug(f"Base dataset: {base_dataset.info}")
        logger.debug("#" * 80)

        # Handle both zip and folder path for exporters
        exporter_class = self.EXPORT_MODES[export_mode]
        if export_mode == "window":
            logger.info(
                f"Converting to {export_mode} format (window_size={window_size}, overlap={overlap})."
            )
            self.exporter = exporter_class(
                batch_size=batch_size,
                window_size=window_size,
                overlap=overlap,
            )
        else:
            logger.info(f"Converting to {export_mode} format...")
            self.exporter = exporter_class(batch_size=batch_size)

        # Export dataset
        if export_mode == "line":
            if line_augment and line_augment < 0:
                logger.debug("Line augmentation disabled")
                line_augment = 0
            if line_augment and line_augment > 5:
                logger.debug("Reduce amount of line augmentation to 5")
                line_augment = 5
            # For line modes, we can apply mask cropping and augmentation
            dataset = self.exporter.process_dataset(
                dataset=base_dataset,
                mask=mask_crop,
                min_width=min_width,
                min_height=min_height,
                allow_empty=allow_empty,
                line_augment=line_augment,
            )
        elif export_mode == "region":
            # For region modes, we can apply mask cropping
            dataset = self.exporter.process_dataset(
                dataset=base_dataset,
                mask=mask_crop,
                allow_empty=allow_empty,
                min_width=min_width,
                min_height=min_height,
            )
        else:
            dataset = self.exporter.process_dataset(dataset=base_dataset)

        if dataset and self.stats_cache is None:
            logger.debug("Computing statistics...")
            self._compute_stats(dataset)

        logger.info(f"Exported dataset")

        if dataset and split_train and 0.0 < split_train < 1.0:
            logger.info(f"Splitting dataset into train and test sets (train size={split_train})...")
            dataset = dataset.train_test_split(
                train_size=split_train,
                shuffle=split_shuffle,
                seed=split_seed,
            )
            logger.info(
                f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}"
            )

        if dataset:
            return dataset
        else:
            raise ValueError("dataset is None after conversion")

    def upload_to_hub(
            self,
            dataset: Dataset | DatasetDict,
            repo_id: str,
            token: str = "",
            private: bool = False,
            commit_message: str = "Dataset created by pagexml-hf",
            append: bool = False,
            number_of_augmentations: int = 0,
    ) -> str:
        """
        Upload dataset to HuggingFace Hub using parquet shards (memory efficient).

        Data is organized by split and project: /data/<split>/<project_name>/<timestamp>-<shard>.parquet

        Args:
            dataset: The dataset to upload
            repo_id: Repository ID (e.g., "username/dataset-name")
            token: HuggingFace token (if None, will try to get from cache or HF_TOKEN env var)
            private: Whether to make the repo private
            commit_message: Custom commit message
            append: If True, add as new parquet shard (checks feature compatibility).
                   If False, overwrite all existing data. Default: False
            number_of_augmentations: How many augmentations created
        Returns:
            Repository URL
        """
        # Check feature compatibility if appending
        if append and self.exporter is not None:
            # We can infer the export_mode from the exporter instance
            export_mode = None
            for mode, exporter_class in self.EXPORT_MODES.items():
                if isinstance(self.exporter, exporter_class):
                    export_mode = mode
                    break

            if export_mode:
                try:
                    self._check_exporter_feature_compatibility(
                        repo_id=repo_id,
                        export_mode=export_mode,
                        token=token,
                    )
                except ValueError:
                    # Feature mismatch - abort upload
                    raise

        uploader = HubUploader(source_name=self.source_name)
        return uploader.upload_to_hub(
            dataset=dataset,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
            append=append,
            number_of_augmentations=number_of_augmentations,
        )

    def convert_and_upload(
            self,
            repo_id: str,
            export_mode: str = "text",
            token: str = "",
            private: bool = False,
            commit_message: str = "Add new data export",
            batch_size: int = 32,
            mask_crop: bool = False,
            min_width: int = 0,
            min_height: int = 0,
            allow_empty: bool = False,
            window_size: int = 2,
            overlap: int = 0,
            split_train: float = 1.0,
            split_seed: int = 42,
            split_shuffle: bool = False,
            append: bool = False,
            line_augment: int = 0,
    ) -> str:
        """
        Convert and upload in one step using parquet shards (memory efficient).

        Data is organized by split and project: /data/<split>/<project_name>/<timestamp>-<shard>.parquet

        Args:
            repo_id: Repository ID (e.g., "username/dataset-name")
            export_mode: Export mode ('raw_xml', 'text', 'region', 'line', 'window')
            token: HuggingFace token
            private: Whether to make the repo private
            commit_message: Custom commit message
            batch_size: Batch size for dataset mapping
            mask_crop: Whether to crop the mask to polygon (only for region and line mode)
            min_width: Minimum width of the regions/lines to be processed \
                (only for region and line mode)
            min_height: Minimum height of the regions/lines to be processed \
                (only for region and line mode)
            allow_empty: Whether to allow empty elements in the output (default: False)
            window_size: Number of lines per window (only for window mode)
            overlap: Number of lines to overlap between windows (only for window mode)
            split_train: Split train set size (between 0 and 1, e.g. 0.8 for 80% train, 20% test)
            split_seed: Random seed for train/test split
            split_shuffle: Whether to shuffle the dataset before splitting
            append: If True, add as new parquet shard (checks feature compatibility).
                   If False, overwrite all existing data. Default: False
            line_augment: Amount of random augmentations of line images \
                (only for line mode)
        Returns:
            Repository URL
        """
        # Check feature compatibility BEFORE expensive data processing
        logger.debug(f"Convert and upload with export_mode={export_mode}, append={append}")
        if append:
            try:
                self._check_exporter_feature_compatibility(
                    repo_id=repo_id,
                    export_mode=export_mode,
                    token=token,
                )
            except ValueError:
                # Feature mismatch - abort before processing
                raise

        # Convert the data
        dataset = self.convert(
            export_mode=export_mode,
            batch_size=batch_size,
            mask_crop=mask_crop,
            min_width=min_width,
            min_height=min_height,
            allow_empty=allow_empty,
            window_size=window_size,
            overlap=overlap,
            split_train=split_train,
            split_seed=split_seed,
            split_shuffle=split_shuffle,
            line_augment=line_augment,
        )

        # cache_files_deleted = dataset.cleanup_cache_files()
        # logger.debug(f"Number of cache files deleted: {cache_files_deleted}")

        return self.upload_to_hub(
            dataset=dataset,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
            append=append,
            number_of_augmentations=line_augment,
        )
