import os

scan_dirs = os.listdir('scans')
print(len(scan_dirs))

for scan_dir in scan_dirs:
    scan_files = os.listdir(os.path.join('scans', scan_dir))
    if scan_files.__len__() != 5: # see if the files are complete
        print(scan_dir, scan_files)
        print('\n')
    for scan_file in scan_files: # see if the files are correct
        scan_id = scan_file[:12]
        if scan_id != scan_dir:
            print(scan_dir, scan_files)