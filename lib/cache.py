#!/usr/bin/env python3
"""
Model cache management for TT-Forge Compiletron.
Simplified version - uses PyTorch's built-in caching.
"""

from pathlib import Path
from typing import Dict, List, Optional


class ModelCache:
    """Manages model weight downloads and caching."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Custom cache directory (default: ~/.cache/compiletron)
        """
        self.cache_dir = cache_dir or Path.home() / ".cache/compiletron"
        self.torch_cache = Path.home() / ".cache/torch/hub/checkpoints"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with total_size_mb, num_files, oldest timestamp
        """
        if not self.torch_cache.exists():
            return {'total_size_mb': 0, 'num_files': 0}

        total_size = 0
        num_files = 0
        for file in self.torch_cache.glob('*'):
            if file.is_file():
                total_size += file.stat().st_size
                num_files += 1

        return {
            'total_size_mb': total_size / (1024 * 1024),
            'num_files': num_files,
        }

    def clear_cache(self, confirm: bool = False):
        """
        Clear cached model weights.

        Args:
            confirm: Must be True to actually delete files (safety)
        """
        if not confirm:
            print("⚠️  Dry run mode. Use confirm=True to actually delete.")
            stats = self.get_cache_stats()
            print(f"   Would free {stats['total_size_mb']:.1f} MB ({stats['num_files']} files)")
            return

        if self.torch_cache.exists():
            import shutil
            shutil.rmtree(self.torch_cache)
            print(f"✓ Cleared cache: {self.torch_cache}")


# Example usage
if __name__ == '__main__':
    cache = ModelCache()
    stats = cache.get_cache_stats()
    print(f"PyTorch cache: {stats['total_size_mb']:.1f} MB ({stats['num_files']} files)")
