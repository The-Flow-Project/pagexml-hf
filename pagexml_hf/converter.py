"""
Main converter class for Transkribus to HuggingFace datasets.
"""

from typing import Dict, Any
from pathlib import Path
import os

from datasets import (
    Dataset,
    Features,
    Image as DatasetImage,
    Value,
    List,
)
from huggingface_hub import create_repo, get_token

from .exporters import (
    RawXMLExporter,
    TextExporter,
    RegionExporter,
    LineExporter,
    WindowExporter,
)
from .logger import logger

POLYGON_FEATURE = List(feature=List(feature=Value("int64")))

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


class XmlConverter:
    """
    Main converter class for converting Transkribus
    ZIP/XML-folder files to HuggingFace datasets.
    """

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

    def _create_base_dataset(self) -> Dataset:
        """
        Creates initial raw dataset from the generator.
        Writes the data to disk (Arrow format) to prevent OOM.
        """
        logger.info("Creating dataset (temporary local, page-level).")

        ds = Dataset.from_generator(
            self.gen_func,
            gen_kwargs=self.gen_kwargs,
            features=PRE_FEATURES,
        )
        logger.debug(f"Created dataset ({ds.info}).")
        return ds

    def _compute_stats(self, dataset: Dataset):
        total_regions = 0
        total_lines = 0
        projects = set()
        logger.debug(f"Columns of dataset: {dataset.column_names}")

        if "regions" in dataset.column_names and dataset["regions"] is not None:
            logger.info("Computing statistics for regions.")
            stats_batch = dataset.select_columns(["regions", "project_name"])

            for item in stats_batch:
                if item["project_name"] is not None:
                    projects.add(item["project_name"])

                if item["regions"] is not None:
                    regions_list = item["regions"]
                    total_regions += len(regions_list)

                    for r in regions_list:
                        total_lines += len(r["text_lines"])
        else:
            logger.info("Computing statistics for pages (no regions found).")
            total_regions = 0
            total_lines = 0
            for project in dataset["project_name"]:
                projects.add(project)

        total_pages = len(dataset)

        self.stats_cache = {
            "total_pages": total_pages,
            "total_regions": total_regions,
            "total_lines": total_lines,
            "projects": list(projects),
            "avg_regions_per_page": total_regions / total_pages if total_pages > 0 else 0,
            "avg_lines_per_page": total_lines / total_pages if total_pages > 0 else 0,
            "avg_lines_per_region": total_lines / total_regions if total_regions > 0 else 0,
        }

    def convert(
            self,
            export_mode: str = "text",
            window_size: int = 2,
            overlap: int = 0,
            batch_size: int = 32,
            split_train: float | None = None,
            split_seed: int | None = 42,
            split_shuffle: bool | None = False,
            mask_crop: bool | None = False,
            min_width: int | None = None,
            min_height: int | None = None,
            allow_empty: bool | None = False,
    ) -> Dataset:
        """
        Convert parsed data to a HuggingFace dataset.

        Args:
            export_mode: Export mode ('raw_xml', 'text', 'region', 'line', 'window')
            window_size: Number of lines per window (only for window mode)
            overlap: Number of lines to overlap between windows (only for window mode)
            batch_size: Batch size for dataset mapping
            split_train: Split train set size (between 0 and 1, e.g. 0.8 for 80% train, 20% test)
            split_seed: Random seed for train/test split
            split_shuffle: Whether to shuffle the dataset before splitting
            mask_crop: Whether to crop the mask to polygon (only for region and line mode)
            min_width: Minimum width of the regions/lines to be processed \
                (only for region and line mode)
            min_height: Minimum height of the regions/lines to be processed \
                (only for region and line mode)
            allow_empty: Whether to allow empty elements in the output (default: False)
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

        if self.stats_cache is None:
            logger.debug("Computing statistics...")
            self._compute_stats(base_dataset)

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
        if export_mode in ("line", "region"):
            # For line and region modes, we can apply mask cropping
            dataset = self.exporter.process_dataset(
                dataset=base_dataset,
                mask=mask_crop,
                min_width=min_width,
                min_height=min_height,
                allow_empty=allow_empty,
            )
        else:
            dataset = self.exporter.process_dataset(dataset=base_dataset)

        logger.info(f"Exported dataset")

        if split_train is not None:
            logger.info(f"Splitting dataset into train and test sets (train size={split_train})...")
            if not 0.0 < split_train < 1.0:
                raise ValueError("split_train must be between 0 and 1")
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
            dataset: Dataset,
            repo_id: str,
            token: str | None = None,
            private: bool = False,
            commit_message: str | None = None,
    ) -> str:
        """
        Upload dataset to HuggingFace Hub.

        Args:
            dataset: The dataset to upload
            repo_id: Repository ID (e.g., "username/dataset-name")
            token: HuggingFace token (if None, will try to get from cache or HF_TOKEN env var)
            private: Whether to make the repo private
            commit_message: Custom commit message

        Returns:
            Repository URL
        """
        # Try to get token in this order:
        # 1. Explicit token parameter
        # 2. HF_TOKEN environment variable
        # 3. HuggingFace cache (from huggingface-cli login)
        if token is None:
            token = os.getenv("HF_TOKEN")
            if token is None:
                try:
                    token = get_token()
                except Exception as e:
                    logger.error(f"Error getting token: {e}")

        # If still no token, provide helpful error message
        if token is None:
            raise ValueError(
                "No HuggingFace token found. Please either:\n"
                "1. Run 'huggingface-cli login' to authenticate\n"
                "2. Set HF_TOKEN environment variable\n"
                "3. Pass token parameter directly"
            )

        # Create repository if it doesn't exist
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=token,
                exist_ok=True,
            )
            logger.info(f"Repository {repo_id} created/verified")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise

        # Upload dataset
        if commit_message is None:
            commit_message = f"Upload Transkribus dataset from {self.source_name}"

        logger.info(f"Uploading dataset to {repo_id}...")
        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )

        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Dataset uploaded successfully: {repo_url}")
        return repo_url

    def convert_and_upload(
            self,
            repo_id: str,
            export_mode: str = "text",
            token: str | None = None,
            private: bool = False,
            commit_message: str | None = None,
            window_size: int = 2,
            overlap: int = 0,
            split_train: float | None = None,
            split_seed: int | None = 42,
            split_shuffle: bool | None = False,
    ) -> str:
        """
        Convert and upload in one step.

        Args:
            repo_id: Repository ID (e.g., "username/dataset-name")
            export_mode: Export mode ('raw_xml', 'text', 'region', 'line', 'window')
            token: HuggingFace token
            private: Whether to make the repo private
            commit_message: Custom commit message
            window_size: Number of lines per window (only for window mode)
            overlap: Number of lines to overlap between windows (only for window mode)
            split_train: Split train set size (between 0 and 1, e.g. 0.8 for 80% train, 20% test)
            split_seed: Random seed for train/test split
            split_shuffle: Whether to shuffle the dataset before splitting
        Returns:
            Repository URL
        """
        dataset = self.convert(
            export_mode=export_mode,
            window_size=window_size,
            overlap=overlap,
            split_train=split_train,
            split_seed=split_seed,
            split_shuffle=split_shuffle,
        )
        return self.upload_to_hub(
            dataset=dataset,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
        )
