'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file genScanNetData.py

    \brief Ply binary reader. This code is a modification from the file ply.py 
        from the project pyntcloud of David de la Iglesia Castro 
        (https://github.com/daavoo/pyntcloud)

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as np
from collections import defaultdict

sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


def read_points_binary_ply(filename):
    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start whith the word ply')

        fmt = ply.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file format is ascii not binary')

        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b'property' in line:
                line = line.split()
                if b'list' not in line:
                    dtypes[name].append((line[2], ext + ply_dtypes[line[1]]))
            count += 1

        end_header = ply.tell()

    with open(filename, 'rb') as ply:
        ply.seek(end_header)
        points_np = np.fromfile(ply, dtype=np.dtype(dtypes["vertex"]), count=points_size)
        if ext != sys_byteorder:
            points_np = points_np.byteswap().newbyteorder()

    return points_np