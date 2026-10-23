#!/bin/bash
cd /root/welm_opt/fa256
eval /opt/dtk/bin/aicc -c welm_fa256.cpp -o welm_fa256.o -I/root/welm_opt/fa_src/csrc/flash_attn_hipc $(cat flags.txt) -DTORCH_EXTENSION_NAME=welm_fa256 -fvisibility=hidden && eval /opt/dtk/bin/aicc -shared -o welm_fa256.so welm_fa256.o $(cat lflags.txt)
