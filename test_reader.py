from pathlib import Path
from echemistpy.io import load
import xarray as xr
import time

file_path = Path(r"e:\Desktop\echemistpy\docs\examples\TEM\EDS\0020 - 1406 Pristine MnO2 HAADF 10000 x\0020 - 1406 Pristine MnO2 HAADF 10000 x.emd")
print(f"File exists: {file_path.exists()}")

if file_path.exists():
    start_time = time.time()
    raw_data, raw_info = load(file_path)
    end_time = time.time()
    print(f"Load successful in {end_time - start_time:.2f} seconds!")
    print(f"Sample: {raw_info.sample_name}")

    print("\nDataTree structure:")
    for node in raw_data.data.subtree:
        print(f"Node: {node.path}")
        if node.has_data:
            print(f"  Dims: {list(node.dims)}")
            print(f"  Vars: {list(node.data_vars)}")
