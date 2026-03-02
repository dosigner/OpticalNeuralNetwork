"""Export trained D2NN phase masks as physical height maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from d2nn.cli.common import build_model_from_config, load_checkpoint, load_config
from d2nn.export.heightmap import export_height_map
from d2nn.export.lumerical import LumericalBuilder, LumericalConfig
from d2nn.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export phase masks to height maps")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--out-dir", default=None, help="Output directory override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model_from_config(cfg)
    load_checkpoint(args.checkpoint, model)

    export_cfg = cfg.get("export", {})
    physics = cfg.get("physics", {})
    q_levels = export_cfg.get("quantization_levels", None)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent.parent / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_n = float(export_cfg.get("delta_n", 0.7227))
    wavelength = float(physics.get("wavelength", 0.00075))

    height_paths: list[Path] = []
    for idx, layer in enumerate(model.layers):
        phase = layer.phase_constraint(layer.raw_phase).detach().cpu().numpy()
        out_path = out_dir / f"height_layer_{idx + 1}.npy"
        export_height_map(
            out_path,
            phase,
            wavelength,
            delta_n,
            quantization_levels=q_levels,
        )
        height_paths.append(out_path)

    lcfg = export_cfg.get("lumerical", {})
    lumerical_enabled = bool(lcfg.get("enabled", False))
    if lumerical_enabled:
        sim_cfg = LumericalConfig(
            N=int(physics.get("N", 200)),
            dx=float(physics.get("dx", 0.0004)),
            z_layer=float(physics.get("z_layer", 0.03)),
            z_out=float(physics.get("z_out", 0.03)),
            wavelength=wavelength,
            material_name=str(lcfg.get("material_name", "VeroBlackPlus_RGD875")),
            refractive_index=float(lcfg.get("refractive_index", 1.7227)),
            extinction_k=float(lcfg.get("extinction_k", 0.0)),
            temp_dir=str(lcfg.get("temp_dir", out_dir / "lumerical")),
            base_fsp=str(lcfg.get("base_fsp", "base.fsp")),
            out_fsp=str(lcfg.get("out_fsp", "final.fsp")),
            hide_gui=bool(lcfg.get("hide_gui", True)),
            mock_mode=bool(lcfg.get("mock_mode", False)),
        )
        builder = LumericalBuilder(sim_cfg)
        base = builder.build_base_simulation()
        layer_fsps = []
        for i, hp in enumerate(height_paths, start=1):
            hm = np.load(hp)
            layer_fsps.append(builder.build_layer(i, hm))
        merged = builder.merge_layers(base, layer_fsps)
        save_json(out_dir / "lumerical_manifest.json", {"base": str(base), "layers": [str(x) for x in layer_fsps], "merged": str(merged), "mock": builder.is_mock})

    print(str(out_dir))


if __name__ == "__main__":
    main()
