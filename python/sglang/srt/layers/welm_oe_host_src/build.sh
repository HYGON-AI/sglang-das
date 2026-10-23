#!/bin/bash
# Build oe_host_gather.so (host-mapped OE row gather) next to welm_oe_host.py.
# flags.txt / lflags.txt: same compile/link flags as welm_pa_part_src.
set -e
cd "$(dirname "$0")"
F=../attention/welm_pa_part_src
eval /opt/dtk/bin/aicc -x hip -c oe_host_gather.cpp -o /tmp/oe_host_gather.o $(sed 's#-I/[^ ]*fa_src[^ ]*##g' $F/flags.txt) -DTORCH_EXTENSION_NAME=oe_host_gather -fvisibility=hidden
eval /opt/dtk/bin/aicc -shared -o ../oe_host_gather.so /tmp/oe_host_gather.o $(cat $F/lflags.txt)
