#!/bin/bash
cd /root/welm_opt/pa_part
eval /opt/dtk/bin/aicc -x hip -c welm_pa_part.cpp -o welm_pa_part.o -I. $(cat flags.txt) -DTORCH_EXTENSION_NAME=welm_pa_part -fvisibility=hidden && eval /opt/dtk/bin/aicc -shared -o welm_pa_part.so welm_pa_part.o $(cat lflags.txt)
