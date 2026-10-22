#!/bin/bash
nline_per_set=2000
[[ $# -ge 1 ]] && nline_per_set=$1
rm -fr set*
if ! [ -f box.raw ]; then
    echo "Error: box.raw not found. Exiting."
    exit 1
fi
split box.raw -l $nline_per_set -d -a 3 box.raw.
split coord.raw -l $nline_per_set -d -a 3 coord.raw.
for f in energy force virial atom_ener fparam atomic_dipole atomic_polarizability; do
    test -f ${f}.raw && split ${f}.raw -l $nline_per_set -d -a 3 ${f}.raw.
done
nset=$(ls -1 box.raw.* | wc -l)
nset_1=$((nset - 1))
for ii in $(seq 0 $nset_1); do
    pi=$(printf "%03d" $ii)
    mkdir -p "set.$pi"
    mv "box.raw.$pi" "set.$pi/box.raw"
    mv "coord.raw.$pi" "set.$pi/coord.raw"
    for f in energy force virial atom_ener fparam atomic_dipole atomic_polarizability; do
        test -f "${f}.raw.$pi" && mv "${f}.raw.$pi" "set.$pi/${f}.raw"
    done
    cd "set.$pi"
    python3 <<END
import os
import numpy as np
def c(f, n):
    if os.path.isfile(f):
        try:
            d = np.loadtxt(f, ndmin=2).astype(np.float32)
            np.save(n, d)
        except: pass
c('box.raw','box'); c('coord.raw','coord'); c('energy.raw','energy');
c('force.raw','force'); c('virial.raw','virial'); c('atom_ener.raw','atom_ener');
c('fparam.raw','fparam'); c('dipole.raw','dipole'); c('polarizability.raw','polarizability');
c('atomic_dipole.raw','atomic_dipole'); c('atomic_polarizability.raw','atomic_polarizability');
END
    rm -f *.raw
    cd ..
done
