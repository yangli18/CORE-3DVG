"""
Customized script to download only the necessary ScanNet files.

You still need to get the original script from the authors of ScanNet.
"""

import os
import json

from tqdm import tqdm


def get_scan_ids():
    """Load the .csv files and return a set of scan_ids."""
    scan_ids = []
    # for name in ['train', 'val']:
    for name in ['test']:
        with open('data/meta_data/scannetv2_' + name + '.txt') as fid:
            scan_ids += fid.readlines()

    scan_ids = [_sid.strip('\n') for _sid in scan_ids]

    # with open('DATA_ROOT/scanrefer/ScanRefer_filtered_test.json') as f:
    #     test_json = json.load(f)
    # scan_ids = set([ann["scene_id"] for ann in test_json])

    exist_scan_ids = set([name for name in os.listdir('DATA_ROOT/scans/') if name.startswith('scene')])
    scan_ids = set.difference(set(scan_ids), exist_scan_ids)

    return sorted(list(scan_ids))


def download_scan_id(scan_id):
    """Download files for a specifed scan_id."""
    # command = 'python download-scannet.py -o . --id %s' % scan_id
    command = '/usr/bin/python scripts/download-scannet.py -o . --id %s' % scan_id
    to_download = [
        '.aggregation.json',
        '.txt',
        '_vh_clean_2.0.010000.segs.json',
        '_vh_clean_2.ply',
        '_vh_clean_2.labels.ply'
    ]
    for filetype in to_download:
        os.system(command + ' --type ' + filetype)


def main():
    """Download all necessary files for all scan_ids."""
    scan_ids = get_scan_ids()
    print(scan_ids)
    for scan_id in tqdm(scan_ids):
        download_scan_id(scan_id)


if __name__ == "__main__":
    main()
