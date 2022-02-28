import sys
import os
import shutil
import datetime
import mritopng

file_path = sys.argv[1]
os.makedirs("./eval/xray", exist_ok=True)
os.makedirs("./eval/json", exist_ok=True)
coun = 0
num = 1
for (path, dir, files) in os.walk(file_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.dcm':
            full_path = path + '/' + filename
            dst_dir = "./eval/xray/" + filename
            shutil.copy(full_path,dst_dir)
            print(full_path)
        if ext == '.json':
            full_path = path + '/' + filename
            dst_dir = "./eval/json/" + filename
            shutil.copy(full_path,dst_dir)
            print(full_path)

print("dicom image extraction")
# Convert a whole folder recursively
mritopng.convert_folder('./eval/xray/', './eval/png')
