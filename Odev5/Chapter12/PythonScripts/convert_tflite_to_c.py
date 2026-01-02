#!/usr/bin/env python3
"""
convert_tflite_to_c.py
Basit TFLite->C array çevirici.
Kullanım:
 python convert_tflite_to_c.py models_kws\kws_cnn_int8.tflite kws_model
Çıktılar: kws_model.c, kws_model.h
"""
import sys, os
def to_c_array(bytes_data, varname):
    arr = ",".join(str(b) for b in bytes_data)
    content_c = f'#include "{varname}.h"\n\nconst unsigned char {varname}[] = {{{arr}}};\nconst unsigned int {varname}_len = {len(bytes_data)};\n'
    header = f'#pragma once\nextern const unsigned char {varname}[];\nextern const unsigned int {varname}_len;\n'
    return header, content_c

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: convert_tflite_to_c.py <tflite_path> <varname>")
        sys.exit(1)
    tflite_path = sys.argv[1]
    varname = sys.argv[2]
    with open(tflite_path, "rb") as f:
        b = f.read()
    header, content = to_c_array(b, varname)
    with open(varname + ".h", "w") as hf:
        hf.write(header)
    with open(varname + ".c", "w") as cf:
        cf.write(content)
    print("Wrote", varname + ".c and " + varname + ".h")