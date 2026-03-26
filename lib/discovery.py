"""
Model Discovery Module

Automatically discover new models from:
1. TT-Forge repositories (test files, examples)
2. HuggingFace model hub (by family/architecture)

Usage:
    from lib.discovery import discover_forge_models, discover_huggingface_models

    # Find models in forge repos
    forge_models = discover_forge_models()

    # Find ResNet models on HuggingFace
    hf_models = discover_huggingface_models(family='resnet')
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class DiscoveredModel:
    """Represents a discovered model"""
    name: str
    family: str
    source: str  # 'forge', 'huggingface', 'timm'
    framework: str  # 'pytorch', 'jax', 'tensorflow'
    location: str  # File path or HF model ID
    confidence: float  # 0.0-1.0, how confident we are this will work
    metadata: Dict = None

    def to_dict(self):
        return asdict(self)


class ForgeRepoScanner:
    """Scan TT-Forge repositories for model usage"""

    def __init__(self, forge_fe_path: Optional[str] = None):
        self.forge_fe_path = forge_fe_path or os.path.expanduser("~/tt-forge-fe")

    def scan(self) -> List[DiscoveredModel]:
        """Scan forge repos for model instantiations"""
        models = []

        if not os.path.exists(self.forge_fe_path):
            print(f"Warning: tt-forge-fe not found at {self.forge_fe_path}")
            return models

        # Scan test directories
        test_dirs = [
            "forge/test/models",
            "forge/test/mlir",
            "forge/test/benchmark",
        ]

        for test_dir in test_dirs:
            dir_path = os.path.join(self.forge_fe_path, test_dir)
            if os.path.exists(dir_path):
                models.extend(self._scan_directory(dir_path))

        return models

    def _scan_directory(self, directory: str) -> List[DiscoveredModel]:
        """Scan a directory for Python files with model instantiations"""
        models = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    models.extend(self._scan_file(file_path))

        return models

    def _scan_file(self, file_path: str) -> List[DiscoveredModel]:
        """Scan a Python file for model patterns"""
        models = []

        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Pattern 1: torchvision.models.resnet50()
            pattern1 = r'torchvision\.models\.(\w+)\('
            for match in re.finditer(pattern1, content):
                model_name = match.group(1)
                models.append(DiscoveredModel(
                    name=model_name,
                    family=self._extract_family(model_name),
                    source='forge',
                    framework='pytorch',
                    location=file_path,
                    confidence=0.8,
                    metadata={'pattern': 'torchvision.models'}
                ))

            # Pattern 2: timm.create_model('resnet50')
            pattern2 = r'timm\.create_model\([\'"]([^\'"]+)[\'"]\)'
            for match in re.finditer(pattern2, content):
                model_name = match.group(1)
                models.append(DiscoveredModel(
                    name=model_name,
                    family=self._extract_family(model_name),
                    source='timm',
                    framework='pytorch',
                    location=file_path,
                    confidence=0.9,
                    metadata={'pattern': 'timm.create_model'}
                ))

            # Pattern 3: transformers.AutoModel.from_pretrained('bert-base')
            pattern3 = r'(?:AutoModel|AutoModelForSequenceClassification|AutoModelForCausalLM)\.from_pretrained\([\'"]([^\'"]+)[\'"]\)'
            for match in re.finditer(pattern3, content):
                model_id = match.group(1)
                models.append(DiscoveredModel(
                    name=model_id,
                    family=self._extract_family(model_id),
                    source='huggingface',
                    framework='pytorch',
                    location=file_path,
                    confidence=0.7,
                    metadata={'pattern': 'transformers.AutoModel', 'model_id': model_id}
                ))

        except Exception as e:
            # Skip files that can't be read
            pass

        return models

    def _extract_family(self, model_name: str) -> str:
        """Extract model family from model name"""
        name_lower = model_name.lower()

        # Common families
        families = [
            'resnet', 'vgg', 'efficientnet', 'densenet', 'mobilenet',
            'inception', 'squeezenet', 'alexnet', 'googlenet',
            'bert', 'gpt', 'xlm', 'roberta', 't5', 'bart',
            'vit', 'deit', 'swin', 'convnext',
            'yolo', 'unet', 'segformer'
        ]

        for family in families:
            if family in name_lower:
                return family

        # Extract prefix before numbers
        match = re.match(r'([a-z]+)', name_lower)
        if match:
            return match.group(1)

        return 'unknown'


class HuggingFaceDiscoverer:
    """Discover models from HuggingFace model hub"""

    def __init__(self):
        try:
            from huggingface_hub import HfApi
            self.api = HfApi()
            self.available = True
        except ImportError:
            print("Warning: huggingface_hub not installed. Run: pip install huggingface_hub")
            self.available = False

    def search(self,
               family: Optional[str] = None,
               task: Optional[str] = None,
               library: str = 'pytorch',
               limit: int = 20) -> List[DiscoveredModel]:
        """Search HuggingFace for models"""

        if not self.available:
            return []

        models = []

        try:
            # Use search parameter for family (text search works better than tags)
            search_query = family.lower() if family else None

            # Build search filter
            # Note: HuggingFace API has issues combining filter + search
            # When doing text search, skip the library filter and filter results afterward
            search_filter = {}
            if not search_query:
                # Only use filter when NOT doing text search
                if library:
                    search_filter['library'] = library
                if task:
                    search_filter['task'] = task

            # Query HuggingFace
            model_infos = self.api.list_models(
                filter=search_filter if search_filter else None,
                search=search_query,
                sort="downloads",
                direction=-1,
                limit=limit * 2 if search_query else limit  # Get more when filtering post-query
            )

            count = 0
            for model_info in model_infos:
                # If we did text search, filter for desired library post-query
                if search_query and library:
                    tags = model_info.tags or []
                    if library not in tags:
                        continue

                # Check against limit (in case we're filtering post-query)
                if count >= limit:
                    break

                model_id = model_info.modelId

                # Extract family from model ID or tags
                detected_family = family or self._detect_family(model_id, model_info.tags or [])

                models.append(DiscoveredModel(
                    name=model_id,
                    family=detected_family,
                    source='huggingface',
                    framework=library,
                    location=model_id,
                    confidence=0.6,  # Lower confidence for HF models (may need config)
                    metadata={
                        'downloads': getattr(model_info, 'downloads', 0),
                        'tags': model_info.tags or [],
                        'pipeline_tag': getattr(model_info, 'pipeline_tag', None)
                    }
                ))
                count += 1

        except Exception as e:
            print(f"Error searching HuggingFace: {e}")

        return models

    def _detect_family(self, model_id: str, tags: List[str]) -> str:
        """Detect model family from ID and tags"""

        # Check tags first
        known_families = [
            'resnet', 'vgg', 'efficientnet', 'densenet', 'mobilenet',
            'bert', 'gpt2', 'gpt-neo', 't5', 'bart', 'roberta',
            'vit', 'deit', 'swin', 'convnext'
        ]

        for tag in tags:
            tag_lower = tag.lower()
            for family in known_families:
                if family in tag_lower:
                    return family

        # Check model ID
        model_id_lower = model_id.lower()
        for family in known_families:
            if family in model_id_lower:
                return family

        return 'unknown'


def discover_forge_models(forge_fe_path: Optional[str] = None) -> List[DiscoveredModel]:
    """
    Discover models from TT-Forge repositories

    Args:
        forge_fe_path: Path to tt-forge-fe repo (default: ~/tt-forge-fe)

    Returns:
        List of discovered models
    """
    scanner = ForgeRepoScanner(forge_fe_path)
    return scanner.scan()


def discover_huggingface_models(family: Optional[str] = None,
                                 task: Optional[str] = None,
                                 limit: int = 20) -> List[DiscoveredModel]:
    """
    Discover models from HuggingFace model hub

    Args:
        family: Model family (e.g., 'resnet', 'bert')
        task: Task type (e.g., 'image-classification', 'text-generation')
        limit: Maximum number of models to return

    Returns:
        List of discovered models
    """
    discoverer = HuggingFaceDiscoverer()
    return discoverer.search(family=family, task=task, limit=limit)


def deduplicate_models(models: List[DiscoveredModel]) -> List[DiscoveredModel]:
    """Remove duplicate models, keeping highest confidence"""
    seen = {}

    for model in models:
        key = (model.name.lower(), model.framework)

        if key not in seen or model.confidence > seen[key].confidence:
            seen[key] = model

    return list(seen.values())


def filter_by_family(models: List[DiscoveredModel], family: str) -> List[DiscoveredModel]:
    """Filter models by family"""
    family_lower = family.lower()
    return [m for m in models if family_lower in m.family.lower()]


def save_discovered_models(models: List[DiscoveredModel], output_file: str):
    """Save discovered models to JSON file"""
    data = [m.to_dict() for m in models]

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(models)} models to {output_file}")


def load_discovered_models(input_file: str) -> List[DiscoveredModel]:
    """Load discovered models from JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    models = [DiscoveredModel(**item) for item in data]
    return models


if __name__ == "__main__":
    # Demo usage
    print("=== Discovering Models from TT-Forge ===")
    forge_models = discover_forge_models()
    print(f"Found {len(forge_models)} models in Forge repos")

    # Group by family
    families = {}
    for model in forge_models:
        families.setdefault(model.family, []).append(model.name)

    for family, names in sorted(families.items()):
        print(f"{family}: {len(names)} models")

    print("\n=== Discovering ResNet Models from HuggingFace ===")
    hf_models = discover_huggingface_models(family='resnet', limit=10)
    print(f"Found {len(hf_models)} ResNet models on HuggingFace")

    for model in hf_models[:5]:
        print(f"  {model.name} (downloads: {model.metadata.get('downloads', 0)})")
