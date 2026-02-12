"""
Utility classes for HuggingFace Hub operations.
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    Image as DatasetImage,
    Sequence,
    List,
)
from huggingface_hub import HfApi, create_repo, get_token, repo_exists

from .logger import logger


class HubUploader:
    """Handles uploading datasets to HuggingFace Hub with parquet shards."""

    def __init__(self, source_name: str):
        """
        Initialize the uploader.

        Args:
            source_name: Name of the data source for commit messages
        """
        self.source_name = source_name

    def upload_to_hub(
            self,
            dataset: Dataset | DatasetDict,
            repo_id: str,
            token: str | None = None,
            private: bool = False,
            commit_message: str | None = None,
            append: bool = False,
    ) -> str:
        """
        Upload dataset to HuggingFace Hub using parquet shards.

        Args:
            dataset: The dataset to upload
            repo_id: Repository ID (e.g., "username/dataset-name")
            token: HuggingFace token
            private: Whether to make the repo private
            commit_message: Custom commit message
            append: If True, add as new parquet shard. If False, overwrite.

        Returns:
            Repository URL
        """
        token = self._get_token(token)
        api = HfApi(token=token)

        # Check/create repository
        dataset_repo_exists = self._check_and_create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
        )

        # Handle overwrite mode - delete existing data
        if not append and dataset_repo_exists:
            self._delete_existing_data(api=api, repo_id=repo_id)

        # Prepare commit message
        if commit_message is None:
            if append and dataset_repo_exists:
                commit_message = f"Append Transkribus dataset from {self.source_name}"
            else:
                commit_message = f"Upload Transkribus dataset from {self.source_name}"

        # Upload as parquet shard(s)
        return self._upload_as_parquet_shard(
            api=api,
            dataset=dataset,
            repo_id=repo_id,
            commit_message=commit_message,
            append=append,
            dataset_repo_exists=dataset_repo_exists,
        )

    @staticmethod
    def _get_token(token: str | None) -> str:
        """Get HuggingFace token from various sources."""
        if token is None:
            token = os.getenv("HF_TOKEN")
            if token is None:
                try:
                    token = get_token()
                except Exception as e:
                    logger.error(f"Error getting token: {e}")

        if token is None:
            raise ValueError(
                "No HuggingFace token found. Please either:\n"
                "1. Run 'huggingface-cli login' to authenticate\n"
                "2. Set HF_TOKEN environment variable\n"
                "3. Pass token parameter directly"
            )

        return token

    @staticmethod
    def _check_and_create_repo(
            repo_id: str,
            private: bool,
            token: str,
    ) -> bool:
        """Check if repo exists and create if needed. Returns True if existed."""
        dataset_repo_exists = False
        try:
            dataset_repo_exists = repo_exists(
                repo_id=repo_id,
                repo_type="dataset",
                token=token
            )
            if dataset_repo_exists:
                logger.info(f"Repository {repo_id} already exists")
            else:
                logger.info(f"Repository {repo_id} does not exist yet")
        except Exception as e:
            logger.debug(f"Could not check repo existence: {e}")

        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=token,
                exist_ok=True,
            )
            if not dataset_repo_exists:
                logger.info(f"Repository {repo_id} created")
            else:
                logger.info(f"Repository {repo_id} verified")
        except Exception as e:
            logger.error(f"Error creating repository: {e}")
            raise

        return dataset_repo_exists

    @staticmethod
    def _delete_existing_data(api: HfApi, repo_id: str) -> None:
        """Delete existing data folder from repository."""
        logger.warning(f"Overwrite mode enabled. Deleting existing data from {repo_id}...")
        try:
            api.delete_folder(
                repo_id=repo_id,
                path_in_repo="data",
                repo_type="dataset",
                commit_message="Clear dataset for overwrite",
            )
            logger.info("Existing data deleted")
        except Exception as e:
            logger.debug(f"Could not delete data folder (might not exist): {e}")

    def _upload_as_parquet_shard(
            self,
            api: HfApi,
            dataset: Dataset | DatasetDict,
            repo_id: str,
            commit_message: str,
            append: bool,
            dataset_repo_exists: bool,
    ) -> str:
        """Upload dataset as parquet shards organized by project."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_projects = set()
        splits_info = {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_root = tmp_path / "data"
            data_root.mkdir(parents=True, exist_ok=True)

            # Normalize to DatasetDict
            if isinstance(dataset, Dataset):
                dataset_dict = DatasetDict({"train": dataset})
            else:
                dataset_dict = dataset

            logger.info(f"Processing dataset with splits: {list(dataset_dict.keys())}")

            # Process each split
            for split_name, split_dataset in dataset_dict.items():
                logger.info(f"Processing split '{split_name}'...")

                # Group by project_name
                projects_in_split = ProjectGrouper.group_by_project(split_dataset)
                all_projects.update(projects_in_split.keys())

                split_total_samples = 0

                for project_name, project_indices in projects_in_split.items():
                    project_dataset = split_dataset.select(project_indices)
                    split_total_samples += len(project_dataset)

                    # Create shards on disk
                    self._upload_project_shards(
                        project_dataset=project_dataset,
                        project_name=project_name,
                        split_name=split_name,
                        timestamp=timestamp,
                        data_root=data_root,
                    )

                splits_info[split_name] = split_total_samples
                logger.info(f"✓ Split '{split_name}' complete ({split_total_samples} samples)")

            # Upload all shards in a single batch
            logger.info("Uploading parquet shards with upload_folder...")
            api.upload_folder(
                folder_path=str(data_root),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            logger.info("✓ Parquet shards uploaded")

            # Create or update README
            logger.info("Creating/updating README.md with dataset configuration...")
            ReadmeGenerator.create_or_update_readme(
                api=api,
                repo_id=repo_id,
                dataset=dataset_dict,
                splits_info=splits_info,
                all_projects=list(all_projects),
                append=append,
                dataset_repo_exists=dataset_repo_exists,
            )

        repo_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Dataset uploaded successfully: {repo_url}")
        logger.info("Note: The Hub will automatically merge all parquet files when loading.")
        return repo_url

    @staticmethod
    def _upload_project_shards(
            project_dataset: Dataset,
            project_name: str,
            split_name: str,
            timestamp: str,
            data_root: Path,
    ) -> None:
        """Write shards for a single project under the temp data root."""
        shard_size = 10000
        num_shards = max(1, (len(project_dataset) + shard_size - 1) // shard_size)

        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(project_dataset))
            shard_dataset = project_dataset.select(range(start_idx, end_idx))

            # Create directory structure
            project_dir = data_root / split_name / project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # Save as parquet
            if num_shards > 1:
                parquet_file = project_dir / f"{timestamp}-{shard_idx:04d}.parquet"
            else:
                parquet_file = project_dir / f"{timestamp}-0000.parquet"

            logger.info(
                f"Saving split '{split_name}', project '{project_name}', "
                f"shard {shard_idx + 1}/{num_shards} ({len(shard_dataset)} samples)..."
            )
            shard_dataset.to_parquet(str(parquet_file))
            logger.info(f"✓ Saved shard {shard_idx + 1}/{num_shards}")


class ProjectGrouper:
    """Groups dataset samples by project name."""

    @staticmethod
    def group_by_project(dataset: Dataset) -> Dict[str, list]:
        """
        Group dataset indices by project_name.

        Args:
            dataset: The dataset to group

        Returns:
            Dictionary mapping project_name to list of indices
        """
        projects = {}

        for idx, project_name in enumerate(dataset["project_name"]):
            # Ensure project_name is set
            if project_name is None or project_name == "":
                project_name = "unknown_project"

            if project_name not in projects:
                projects[project_name] = []
            projects[project_name].append(idx)

        logger.info(f"Found {len(projects)} project(s): {list(projects.keys())}")
        return projects


class ReadmeGenerator:
    """Generates and updates README.md for dataset repositories."""

    @staticmethod
    def create_or_update_readme(
            api: HfApi,
            repo_id: str,
            dataset: DatasetDict,
            splits_info: Dict[str, int],
            all_projects: list,
            append: bool,
            dataset_repo_exists: bool,
    ) -> None:
        """Create or update README.md with dataset card configuration."""
        try:
            existing_readme_content = None
            existing_splits_info = {}
            existing_projects = []

            # Try to download existing README if appending
            if append and dataset_repo_exists:
                try:
                    existing_readme_path = api.hf_hub_download(
                        repo_id=repo_id,
                        filename="README.md",
                        repo_type="dataset",
                    )
                    with open(existing_readme_path, encoding='utf-8') as f:
                        existing_readme_content = f.read()
                    logger.info("Found existing README.md, will update it")

                    existing_splits_info = ReadmeParser.parse_splits_from_readme(
                        existing_readme_content
                    )
                    existing_projects = ReadmeParser.parse_projects_from_readme(
                        existing_readme_content
                    )
                except Exception as e:
                    logger.debug(f"Could not load existing README: {e}")

            # Extract features from dataset
            first_split = list(dataset.keys())[0]
            features = dataset[first_split].features

            # Merge projects if appending
            merged_projects = list(all_projects)
            if append and existing_projects:
                # Merge existing projects with new ones (remove duplicates)
                all_projects_set = set(all_projects) | set(existing_projects)
                merged_projects = sorted(list(all_projects_set))
                logger.info(
                    f"Merged {len(existing_projects)} existing + {len(all_projects)} new "
                    f"= {len(merged_projects)} total projects"
                )

            # Calculate total size
            total_splits_info = {}
            if append and existing_splits_info:
                for split_name in set(list(splits_info.keys()) + list(existing_splits_info.keys())):
                    total_splits_info[split_name] = (
                            splits_info.get(split_name, 0) + existing_splits_info.get(split_name, 0)
                    )
            else:
                total_splits_info = splits_info.copy()

            # Calculate dataset size
            total_samples = sum(total_splits_info.values())
            first_sample = dataset[first_split][0]
            approx_size_per_sample = len(str(first_sample)) * 2
            approx_total_size_mb = (total_samples * approx_size_per_sample) / (1024 * 1024)

            # Generate content
            yaml_config = YamlGenerator.generate_dataset_card_yaml(
                splits_info=total_splits_info,
                features=features,
                dataset_size_mb=approx_total_size_mb,
            )

            new_readme = ReadmeGenerator._build_readme_content(
                yaml_config=yaml_config,
                repo_id=repo_id,
                total_splits_info=total_splits_info,
                all_projects=merged_projects,
                total_samples=total_samples,
                approx_total_size_mb=approx_total_size_mb,
                features=features,
            )

            # Upload README
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False,
                                             encoding='utf-8') as tmp_file:
                tmp_file.write(new_readme)
                tmp_readme_path = tmp_file.name

            try:
                api.upload_file(
                    path_or_fileobj=tmp_readme_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message="Update dataset card",
                )
                logger.info("✓ README.md updated successfully")
            finally:
                Path(tmp_readme_path).unlink(missing_ok=True)

        except Exception as e:
            logger.warning(f"Could not create/update README.md: {e}")
            logger.info("Dataset files uploaded, but README may need manual update")

    @staticmethod
    def _build_readme_content(
            yaml_config: str,
            repo_id: str,
            total_splits_info: Dict[str, int],
            all_projects: list,
            total_samples: int,
            approx_total_size_mb: float,
            features: Features,
    ) -> str:
        """Build README content."""
        # Always generate fresh content with updated statistics
        # The YAML frontmatter is updated with merged data
        # The body is regenerated with updated project lists and sample counts

        readme = f"""---
{yaml_config}
---

# Dataset Card for {repo_id.split('/')[-1]}

This dataset was created using pagexml-hf converter from Transkribus PageXML data.

## Dataset Summary

This dataset contains {total_samples:,} samples across {len(total_splits_info)} split(s).

## Dataset Structure

### Data Splits

"""
        for split_name, count in total_splits_info.items():
            readme += f"- **{split_name}**: {count:,} samples\n"

        readme += f"""
### Dataset Size

- Approximate total size: {approx_total_size_mb:,.2f} MB
- Total samples: {total_samples:,}

### Features

"""
        for feature_name, feature_type in features.items():
            readme += f"- **{feature_name}**: `{feature_type}`\n"

        readme += f"""
## Data Organization

Data is organized as parquet shards by split and project:
```
data/
├── <split>/
│   └── <project_name>/
│       └── <timestamp>-<shard>.parquet
```

The HuggingFace Hub automatically merges all parquet files when loading the dataset.

## Usage

```python
from datasets import load_dataset

# Load entire dataset
dataset = load_dataset("{repo_id}") 

# Load specific split
train_dataset = load_dataset("{repo_id}", split="train")
```

### Projects Included

"""
        readme += f"{', '.join(sorted(all_projects))}\n"
        readme += """
"""
        return readme


class ReadmeParser:
    """Parses information from README files."""

    @staticmethod
    def parse_splits_from_readme(readme_content: str) -> Dict[str, int]:
        """Parse split information from existing README."""
        splits = {}
        pattern = r'-\s+name:\s+(\w+)\s+num_examples:\s+(\d+)'
        matches = re.findall(pattern, readme_content)

        for split_name, count in matches:
            splits[split_name] = int(count)

        if splits:
            logger.info(f"Parsed existing splits from README: {splits}")

        return splits

    @staticmethod
    def parse_projects_from_readme(readme_content: str) -> list:
        """Parse project list from existing README."""
        projects = []

        # Look for the "### Projects Included" section
        if "### Projects Included" in readme_content:
            # Find content between "### Projects Included" and next section
            start_marker = "### Projects Included"
            start_idx = readme_content.find(start_marker)
            if start_idx != -1:
                # Find the next section (starts with ##)
                content_after = readme_content[start_idx + len(start_marker):]
                end_idx = content_after.find("\n##")
                if end_idx != -1:
                    projects_section = content_after[:end_idx]
                else:
                    projects_section = content_after

                # Extract project names
                for project in projects_section.split(','):
                    project_name = project.strip()
                    if project_name:
                        projects.append(project_name)

        if projects:
            logger.info(f"Parsed {len(projects)} existing project(s) from README: {projects}")

        return projects


class YamlGenerator:
    """Generates YAML configurations for dataset cards."""

    @staticmethod
    def generate_dataset_card_yaml(
            splits_info: Dict[str, int],
            features: Features,
            dataset_size_mb: float,
    ) -> str:
        """Generate YAML frontmatter for dataset card."""
        yaml_lines = [
            "dataset_info:",
            "  config_name: default",
            "  features:",
        ]

        # Add feature definitions
        for feature_name, feature_type in features.items():
            yaml_lines.extend(
                FeatureYamlConverter.feature_to_yaml(feature_name, feature_type, indent=2)
            )

        yaml_lines.append("  splits:")

        for split_name in splits_info.keys():
            yaml_lines.append(f"  - name: {split_name}")
            yaml_lines.append(f"    num_examples: {splits_info[split_name]}")
            yaml_lines.append(
                f"    num_bytes: {int(dataset_size_mb * 1024 * 1024 / len(splits_info))}"
            )

        total_bytes = int(dataset_size_mb * 1024 * 1024)
        yaml_lines.extend([
            f"  download_size: {total_bytes}",
            f"  dataset_size: {total_bytes}",
        ])

        yaml_lines.extend([
            "configs:",
            "- config_name: default",
            "  data_files:",
        ])

        for split_name in splits_info.keys():
            yaml_lines.append(f"  - split: {split_name}")
            yaml_lines.append(f"    path: data/{split_name}/**/*.parquet")

        yaml_lines.extend([
            "tags:",
            "  - image-to-text",
            "  - htr",
            "  - trocr",
            "  - transcription",
            "  - pagexml",
            "license: mit",
        ])

        return '\n'.join(yaml_lines)


class FeatureYamlConverter:
    """Converts dataset features to YAML format."""

    @staticmethod
    def feature_to_yaml(name: str, feature: Any, indent: int = 0) -> list:
        """Convert a dataset feature to YAML lines."""
        lines = []
        spaces = " " * indent

        lines.append(f"{spaces}- name: {name}")

        # Handle different feature types
        if isinstance(feature, Value):
            lines.append(f"{spaces}  dtype: {feature.dtype}")

        elif isinstance(feature, DatasetImage):
            lines.append(f"{spaces}  dtype:")
            lines.append(f"{spaces}    image:")
            if hasattr(feature, 'decode') and not feature.decode:
                lines.append(f"{spaces}      decode: false")

        elif isinstance(feature, (Sequence, List)):
            lines.append(f"{spaces}  dtype:")

            if hasattr(feature, 'feature'):
                inner_feature = feature.feature

                if isinstance(inner_feature, Value):
                    lines.append(f"{spaces}    sequence: {inner_feature.dtype}")

                elif isinstance(inner_feature, (Sequence, List)):
                    if hasattr(inner_feature, 'feature') and isinstance(inner_feature.feature, Value):
                        lines.append(f"{spaces}    sequence:")
                        lines.append(f"{spaces}      sequence: {inner_feature.feature.dtype}")

                elif isinstance(inner_feature, dict):
                    lines.append(f"{spaces}    sequence:")
                    for sub_name, sub_feature in inner_feature.items():
                        sub_lines = FeatureYamlConverter.feature_to_yaml(
                            sub_name, sub_feature, indent + 3
                        )
                        if sub_lines:
                            lines.append(f"{spaces}      - name: {sub_name}")
                            for sub_line in sub_lines[1:]:
                                lines.append(f"  {sub_line}")

        elif isinstance(feature, dict):
            lines.append(f"{spaces}  dtype:")
            lines.append(f"{spaces}    struct:")
            for sub_name, sub_feature in feature.items():
                sub_lines = FeatureYamlConverter.feature_to_yaml(
                    sub_name, sub_feature, indent + 3
                )
                lines.extend(sub_lines)

        else:
            lines.append(f"{spaces}  dtype: {str(feature)}")

        return lines

