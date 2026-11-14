"""Measure cold JIT and warm dispatch separately using a new cache in /tmp."""

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

# Set before importing Numba. No previously compiled code is reused.
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="elisa-clipping-jit-", dir="/tmp")

import numpy as np
from elisa.binary_system.surface.clipping import sutherland_hodgman_intersection_areas

p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
start = perf_counter()
sutherland_hodgman_intersection_areas(p, p[:3][None])
cold = perf_counter() - start
samples = []
for _ in range(10):
    start = perf_counter()
    sutherland_hodgman_intersection_areas(p, p[:3][None])
    samples.append(perf_counter() - start)
Path(__file__).with_name("compilation.json").write_text(
    json.dumps(dict(cold_first_call_s=cold, warm_samples_s=samples), indent=2)
)
