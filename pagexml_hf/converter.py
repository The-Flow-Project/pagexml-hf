"""
Main converter class for Transkribus to HuggingFace datasets.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from datasets import Dataset
from huggingface_hub import create_repo, get_token
import os

from .parser import XmlParser
from .exporters import (
    RawXMLExporter,
    TextExporter,
    RegionExporter,
    LineExporter,
    WindowExporter,
)


class XmlConverter:
    """Main converter class for converting Transkribus ZIP/XML-folder files to HuggingFace datasets."""

    EXPORT_MODES = {
        "raw_xml": RawXMLExporter,
        "text": TextExporter,
        "region": RegionExporter,
        "line": LineExporter,
        "window": WindowExporter,
    }

    def __init__(
            self,
            zip_path: Optional[str] = None,
            folder_path: Optional[str] = None,
            xmlnamespace: Optional[str] = None,
    ):
        """
        Initialize the converter.

        Args:
            zip_path: Path to the Transkribus ZIP file
            folder_path: Path to the folder containing XML files (if not using ZIP)
        """
        self.zip_path = zip_path
        self.folder_path = Path(folder_path) if folder_path else None
        self.parser = XmlParser(xmlnamespace)
        self.pages = None

    def parse(self) -> None:
        """Parse the ZIP file or folder and extract all page data."""
        if self.zip_path:
            print(f"Parsing ZIP file: {self.zip_path}")
            self.pages = self.parser.parse_zip(self.zip_path)
        elif self.folder_path:
            print(f"Parsing folder: {self.folder_path}")
            self.pages = self.parser.parse_folder(str(self.folder_path))
        else:
            raise ValueError("Either zip_path or folder_path must be provided")
        print(f"Parsed {len(self.pages)} pages")

    def convert(
            self,
            export_mode: str = "text",
            window_size: int = 2,
            overlap: int = 0,
            split_train: Optional[float] = None,
            split_seed: Optional[int] = 42,
            split_shuffle: Optional[bool] = False,
            mask_crop: Optional[bool] = False,
    ) -> Dataset:
        """
        Convert parsed data to a HuggingFace dataset.

        Args:
            export_mode: Export mode ('raw_xml', 'text', 'region', 'line', 'window')
            window_size: Number of lines per window (only for window mode)
            overlap: Number of lines to overlap between windows (only for window mode)
            split_train: Split train set size (between 0 and 1, e.g. 0.8 for 80% train, 20% test)
            split_seed: Random seed for train/test split
            split_shuffle: Whether to shuffle the dataset before splitting
            mask_crop: Whether to crop the mask to polygon (only for region and line mode)
        Returns:
            HuggingFace Dataset
        """
        if self.pages is None:
            self.parse()

        if export_mode not in self.EXPORT_MODES:
            raise ValueError(
                f"Invalid export mode: {export_mode}. Available modes: {list(self.EXPORT_MODES.keys())}"
            )

        exporter_class = self.EXPORT_MODES[export_mode]

        # Handle both zip and folder path for exporters
        if export_mode == "window":
            exporter = exporter_class(
                zip_path=self.zip_path,
                folder_path=self.folder_path,
                window_size=window_size,
                overlap=overlap,
            )
            print(
                f"Converting to {export_mode} format (window_size={window_size}, overlap={overlap})..."
            )
        else:
            exporter = exporter_class(
                zip_path=self.zip_path, folder_path=self.folder_path
            )
            print(f"Converting to {export_mode} format...")

        # Export dataset
        if export_mode == "line" or export_mode == "region":
            # For line and region modes, we can apply mask cropping
            dataset = exporter.export(self.pages, mask=mask_crop)
        else:
            dataset = exporter.export(self.pages)

        if split_train is not None:
            if not (0 < split_train < 1):
                raise ValueError("split_train must be between 0 and 1")
            dataset = dataset.train_test_split(
                train_size=split_train, shuffle=split_shuffle, seed=split_seed
            )
            print(
                f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}"
            )

        return dataset

    def upload_to_hub(
            self,
            dataset: Dataset,
            repo_id: str,
            token: Optional[str] = None,
            private: bool = False,
            commit_message: Optional[str] = None,
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
                    print(f"Error getting token: {e}")
                    pass

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
            print(f"Repository {repo_id} created/verified")
        except Exception as e:
            print(f"Error creating repository: {e}")
            raise

        # Upload dataset
        if commit_message is None:
            commit_message = f"Upload Transkribus dataset from {Path(self.zip_path or self.folder_path).name}"

        print(f"Uploading dataset to {repo_id}...")
        dataset.push_to_hub(repo_id=repo_id, token=token, commit_message=commit_message)

        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        print(f"Dataset uploaded successfully: {repo_url}")
        return repo_url

    def convert_and_upload(
            self,
            repo_id: str,
            export_mode: str = "text",
            token: Optional[str] = None,
            private: bool = False,
            commit_message: Optional[str] = None,
            window_size: int = 2,
            overlap: int = 0,
            split_train: Optional[float] = None,
            split_seed: Optional[int] = 42,
            split_shuffle: Optional[bool] = False,
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the parsed data.

        Returns:
            Dictionary with statistics
        """
        if self.pages is None:
            self.parse()

        total_regions = sum(len(page.regions) for page in self.pages)
        total_lines = sum(
            len(region.text_lines) for page in self.pages for region in page.regions
        )

        projects = set(page.project_name for page in self.pages)

        return {
            "total_pages": len(self.pages),
            "total_regions": total_regions,
            "total_lines": total_lines,
            "projects": list(projects),
            "avg_regions_per_page": total_regions / len(self.pages)
            if self.pages
            else 0,
            "avg_lines_per_page": total_lines / len(self.pages) if self.pages else 0,
        }
