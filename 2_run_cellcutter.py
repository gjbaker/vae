import os
import pandas as pd
from subprocess import call
from subprocess import run

# save directory
save_dir = '/Users/greg/projects/vae_sardana-097/2_cellcutter_output_win30'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

###############################################################################
# run cellcutter

for name in ['test', 'validate', 'train']:
    print()
    print(f'Cutting {name} data...')
    run(
        ["cut_cells", "-z", "--window-size", "30", "--cells-per-chunk", "200",
         "--cache-size", "57711", "/Volumes/My Book/cylinter_input/sardana-097/tif/WD-76845-097.ome.tif",
         "/Volumes/My Book/cylinter_input/sardana-097/mask/nucleiRingMask.tif",
         f"/Users/greg/projects/vae_sardana-097/1_cellcutter_input/{name}.csv",
         f"/Users/greg/projects/vae_sardana-097/2_cellcutter_output_win30/{name}_thumbnails_30.zarr",
         "--channels", "10", "12", "15", "16", "18", "19", "20", "22", "23", "24", "26", "27", "28", "30", "31", "32", "34", "35", "36", "38", "40"
         ]
         )
