"""Preprocessing entry point for Ark-generated complaint-title pseudo labels."""

from __future__ import annotations

from src.summarization.generate_pseudo_titles import (
    generate_pseudo_titles,
    main,
    parse_args,
)

__all__ = ["generate_pseudo_titles", "main", "parse_args"]


if __name__ == "__main__":
    main()
