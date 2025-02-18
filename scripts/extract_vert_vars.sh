#!/usr/bin/env sh
DATA_PATH="/lustre/orion/cli900/world-shared/projects/XNR1K/data/hipq-tc1/global/O8000/"
GRID_FILE="gulf_grid.txt"

#cdo remapnn,gulf_grid.txt /lustre/orion/cli900/world-shared/projects/XNR1K/data/hipq-tc1/global/O8000/ml-dgg/t/t_130_hipq-tc1_104400.ml-dgg.global.grib t_130_hipq-tc1_104400.ml-dgg.gom.nc

remap_grib_timestep() {
    local input_grib=$1      # Input GRIB file
    local timestep=$2        # Time step to extract (1-based index)
    local output_nc=$3     # Output GRIB file


    # Step 1: Extract the specified time step
    echo "${DATA_PATH}${input_grib} ${output_nc} ${timestep}"
    cdo seltimestep,${timestep} "${DATA_PATH}${input_grib}" "${output_nc}"

    # Step 2: Apply nearest-neighbor remapping
    cdo remapnn,"${GRID_FILE}" "${output_nc}" "${output_nc}.tmp"

    # Step 3: Clean up temporary file
    mv "${output_nc}.tmp" "${output_nc}"
    echo "Remapped time step ${timestep} saved to ${output_grib}"
}

remap_grib_timestep sfc-dgg/skt/skt_235_hipq-tc1_097935-116000.sfc-dgg.global.grib 408 skt_235_hipq-tc1_097935-116000.sfc-dgg.gom.grib
remap_grib_timestep sfc-dgg/10u/10u_165_hipq-tc1_097935-116000.sfc-dgg.global.grib 408 10u_165_hipq-tc1_097935-116000.sfc-dgg.gom.grib
remap_grib_timestep sfc-dgg/10v/10v_166_hipq-tc1_097935-116000.sfc-dgg.global.grib 408 10v_166_hipq-tc1_097935-116000.sfc-dgg.gom.grib
remap_grib_timestep sfc-dgg/2t/2t_167_hipq-tc1_097935-116000.sfc-dgg.global.grib 408 2t_167_hipq-tc1_097935-116000.sfc-dgg.gom.grib
