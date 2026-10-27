# vkmmc benchmark inputs

This folder holds benchmark inputs that check vkmmc's absorbed fraction against
the reference values of MCX and MMC. `testbenchmarks.sh` runs each input listed
in `golden.txt` with its reference's photon number and a fixed seed, and passes
a run when its absorbed fraction lies within 5 binomial standard errors plus
`RELTOL` (default 2%) of the reference.

```
cd tests/benchmarks
./testbenchmarks.sh                       # ../../bin/vkmmc
./testbenchmarks.sh ../../bin/vkmmc_optix
MMC_BUILTIN=/path/to/mmc/test/builtin ./testbenchmarks.sh
ONLY='^csg/' ./testbenchmarks.sh
```

See the header of `testbenchmarks.sh` for the other environment variables.

## Sources

- `mcx/`: the 33 MCX built-in examples that define their domain with `Shapes`
  constructs, copied verbatim from MCX's `test/builtin` (MCX commit ac88f59).
  The references are MCX's absorbed fractions at 1e6 photons, from MCX's
  `test/builtin/golden.txt`.
- `mmc/`: MMC's built-in examples from MMC's `test/builtin` (exported after MMC
  commit 5369342). Only the 17 inputs under 100 kB are kept here; the other 51
  (up to 40 MB each) are read from `$MMC_BUILTIN` and skipped when it is unset.
  The references are MMC's absorbed fractions at 1e5 photons, from MMC's
  `test/builtin/golden.txt`.
- `csg/`: 11 inputs converting MMC mesh examples to vkmmc's CSG shapes; each
  names its source in its `Comment` key, and `mkcsg.py` regenerates them. Seven
  are checked against MMC at 1e7 photons; four without an MMC counterpart are
  regression references from vkmmc at 1e6 photons.

## Skipped inputs

`golden.txt` gives the reason for each skipped input. They use features vkmmc
does not support: Tag 0 (background) shapes and grids, `UpperSpace`,
`OriginType` 0, MCX's painter-order overlaps, implicit MMC (`Shapes.MeshROI`)
and adjoint outputs.
