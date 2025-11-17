"""Minimal usage example for the echemistpy package."""

from importlib import metadata


def main() -> None:
    """Print the installed package version."""
    try:
        version = metadata.version("echemistpy")
    except metadata.PackageNotFoundError:
        version = "(local)"
    print(f"echemistpy version: {version}")


if __name__ == "__main__":
    main()
