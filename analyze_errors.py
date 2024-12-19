import os
from tqdm import tqdm


paths = []
for root, dirs, files in os.walk("./multirun"):
    for file in files:
        if file.endswith("log.err"):
            paths.append(os.path.join(root, file))
paths = sorted(paths, reverse=True)
error_dict = {}
path_dict = {}
for path in tqdm(paths):
    with open(path, "r") as f:
        lines = f.readlines()
    if not any("Traceback" in line for line in lines):
        continue
    print(path, "has errors!")
    with open(path.replace("_log.err", "_log.out")) as f2:
        out_lines = f2.readlines()
        key = []
        for i, line in enumerate(out_lines):
            if line.startswith(("id: ", "idx: ")):
                # print(line, end="")
                key.append(line.rstrip())
            elif line.startswith("type: "):
                # print(out_lines[i-1:i+10], end="")
                key.append(out_lines[i-1].rstrip())
                if len(key) == 3:
                    break

    for i, line in enumerate(lines):
        if "Traceback" in line:
            # print("".join(lines[i:]), end="")
            error_dict[tuple(key)] = lines[-1].rstrip()
            path_dict[tuple(key)] = path
            break

# print(error_dict)

from collections import defaultdict
def merge_error_dict(error_dict):
    merged_dict = defaultdict(list)

    # Organize by (id, name, error message)
    for (id_name_idx, error_message) in error_dict.items():
        if len(id_name_idx) != 3:
            print(id_name_idx, error_message + error_dict[id_name_idx])
        else:
            id_val, idx_val, name_val = id_name_idx
            idx = int(idx_val.split(': ')[1])  # Extract numerical index
            merged_dict[(id_val, name_val, error_message)].append(idx)

    # Create new merged dictionary
    final_dict = {}
    for (id_val, name_val, error_message), idx_list in merged_dict.items():
        idx_list.sort()
        final_dict[(id_val, f"idx: {idx_list}", name_val)] = error_message + path_dict[(id_val, f'idx: {idx_list[0]}', name_val,)]

    return final_dict

final_dict = merge_error_dict(error_dict)
for key, value in final_dict.items():
    print(key, value)