import os
import sys
import json
set_path = sys.argv[1]

print "path:",set_path
type_index = 1
files = {}

if os.path.isdir(set_path):
    for child_dir in os.listdir(set_path):
        child_dir_full = os.path.join(set_path,child_dir)
        if os.path.isdir(child_dir_full):
            info = {"type":type_index,"list":[]}
            type_index += 1
            files[child_dir] = info
            for child_file in os.listdir(child_dir_full):
                child_file_full = os.path.join(child_dir_full,child_file)
                info["list"].append(child_file_full)

json_str = json.dumps(files)
output_file = open(sys.argv[2]+"_detail.json",'w')
output_file.write(json_str)
output_file.flush()
output_file.close()
output_file = open(sys.argv[2]+".txt",'w')
for key in files:
    info = files[key]
    for file in info["list"]:
        output_file.write(file)
        output_file.write("\r\n")
        output_file.write(str(info["type"]))
        output_file.write("\r\n")
output_file.flush()
output_file.close()
