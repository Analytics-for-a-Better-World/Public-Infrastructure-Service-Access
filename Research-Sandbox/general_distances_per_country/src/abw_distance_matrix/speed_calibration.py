"""Console entry point for travel-time speed calibration."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Run the historical ``calibrate_speeds.py`` CLI through the package."""
    from calibrate_speeds import main as calibrate_speeds_main

    calibrate_speeds_main(argv)


if __name__ == "__main__":
    main()
