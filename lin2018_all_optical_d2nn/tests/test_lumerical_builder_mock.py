from __future__ import annotations

import numpy as np

from d2nn.export.lumerical import LumericalBuilder, LumericalConfig


def test_lumerical_builder_mock(tmp_path) -> None:
    cfg = LumericalConfig(
        N=32,
        dx=1e-3,
        z_layer=0.01,
        z_out=0.01,
        wavelength=0.75e-3,
        material_name="mock_material",
        refractive_index=1.7,
        temp_dir=str(tmp_path),
        mock_mode=True,
    )

    builder = LumericalBuilder(cfg)
    assert builder.is_mock

    base = builder.build_base_simulation()
    l1 = builder.build_layer(1, np.ones((32, 32), dtype=np.float32))
    merged = builder.merge_layers(base, [l1])

    assert base.exists()
    assert l1.exists()
    assert merged.exists()
