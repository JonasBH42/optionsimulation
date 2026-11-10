The American price basis stores a deterministic binary spatial decision tree under schema `PRICE_AABB_TREE_V1`. It indexes already accepted cells and does not change their numerical qualification.

Two NumPy artifacts are included in the pack manifest and its byte-hash receipts:

- `price_tree_nodes`: little-endian int64, shape `(2*N-1, 3)`, containing left child, right child, and accepted-cell index. A leaf has children `-1`; a branch has cell index `-1`.
- `price_tree_bounds`: little-endian float64, shape `(2*N-1, 4, 2)`, containing inclusive lower/upper bounds for log moneyness, total volatility, rate times maturity, and carry times maturity.

The offline builder balances cells by normalized spatial spread and deterministic median ordering. Original accepted-cell indices are preserved. Loading verifies dtypes, dimensions, finite positive-width boxes, one leaf per accepted cell, exact leaf bounds, canonical child ordering, complete reachability without cycles/shared children, exact parent bounding unions, and a depth limit of 64. Every saved artifact also passes the pack's SHA-256 checks. A self-consistent hash receipt cannot excuse a structurally corrupt tree.

Queries traverse only intersecting node bounds. Every face is inclusive, including coarse/fine faces and corners. Matching cells return in original index order: the lowest index chooses the interpolant, and neighboring-price/Greek checks still inspect every matching face cell. Outside-domain holes remain unavailable. Lookup stops with `PRICE_TREE_LOOKUP_LIMIT` after 4,096 visited nodes or 256 matches; it never falls back to a complete scan or rebuilds the tree online.

Explicitly unqualified legacy fixtures retain scan compatibility. A pack claiming qualified price capability must contain the validated saved tree. The production pipeline writes the tree with its accepted price arrays and declares its schema in the manifest. Numerical price, Greek, inverse-IV, model-error and production release gates remain unchanged.

`tests/test_price_tree.py` compares tree and scan results on interiors, missing cells, exact faces/corners and neighboring floating-point values; verifies lower-index ties and coarse/fine discontinuity rejection; checks bounded pruning over 2,048 cells without calling the scan; and rejects malformed indices, duplicate leaves, bounds corruption, incompatible dtypes/schema, missing artifacts and modified bytes. These structural tests do not establish numerical price accuracy.
