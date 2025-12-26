import h5py
from pathlib import Path

file_path = Path(r"e:\Desktop\echemistpy\docs\examples\TEM\EDS\0020 - 1406 Pristine MnO2 HAADF 10000 x\0020 - 1406 Pristine MnO2 HAADF 10000 x.emd")

with h5py.File(file_path, "r") as f:
    print("Groups in root:")
    for name in f:
        print(f"  {name}")

    if "Data" in f:
        print("Groups in Data:")
        for name in f["Data"]:
            print(f"    {name}")
            for sub in f["Data"][name]:
                print(f"      {sub}")
                if "Data" in f["Data"][name][sub]:
                    print(f"        Shape: {f['Data'][name][sub]['Data'].shape}")
