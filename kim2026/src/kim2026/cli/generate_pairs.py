"""Generate deterministic vacuum/turbulence NPZ caches."""

from __future__ import annotations

import argparse

from kim2026.cli.common import apply_runtime_environment, load_config
from kim2026.turbulence.channel import generate_pair_cache
from kim2026.utils.seed import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_runtime_environment(cfg["runtime"])
    set_global_seed(int(cfg["runtime"]["seed"]), strict_reproducibility=bool(cfg["runtime"]["strict_reproducibility"]))
    generate_pair_cache(cfg)


if __name__ == "__main__":
    main()
