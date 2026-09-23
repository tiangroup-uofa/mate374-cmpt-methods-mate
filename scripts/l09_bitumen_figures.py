"""Run the two bitumen notebooks and save their static fallbacks at 300 dpi.

    uv run --locked python scripts/l09_bitumen_figures.py

The instructor's finalized under/overfitting figures are never regenerated.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = spec_from_file_location(name, ROOT / "activities" / f"{name}.edit.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    _, definitions = module.app.run()
    return definitions


def main():
    linear = load("l09_bitumen_linearized")
    direct = load("l09_bitumen_direct")
    np.testing.assert_array_equal(linear["T"], direct["T"])
    np.testing.assert_array_equal(linear["mu"], direct["mu"])
    assert direct["result"].success, direct["result"].message
    np.testing.assert_allclose(direct["result"].x, direct["parameters"], rtol=1e-5)
    assert linear["log_sse"] < direct["log_sse"]
    assert direct["sse"] < linear["sse"]
    for label, values in [("linearized", linear), ("direct", direct)]:
        path = ROOT / "assets" / f"L09-bitumen-{label}.png"
        values["figure"].savefig(path, dpi=300, bbox_inches="tight")
        plt.close(values["figure"])
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
