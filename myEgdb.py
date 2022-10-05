#!/usr/bin/env python

import os
import sys
from subprocess import run
import random
import struct
import numpy
import zipfile
from functools import *


def SetEnv(iget_path='', lib_path=''):
    import platform
    current_path = os.environ['PATH']
    platform = platform.system() 
    if platform == 'Windows':
        os.environ['PATH'] = iget_path + ';' + current_path
        pass
    elif platform == 'Linux':
        os.environ['PATH'] = iget_path + ':' + current_path
        current_lpath = os.environ['LD_LIBRARY_PATH']
        os.environ['LD_LIBRARY_PATH'] = lib_path + ':' + current_lpath

    elif platform == 'Darwin':
        os.environ['PATH'] = iget_path + ':' + current_path
        current_lpath = os.environ['DYLD_LIBRARY_PATH']
        os.environ['DYLD_LIBRARY_PATH'] = lib_path + ':' + current_lpath
    
    os.environ['CLIENT_INI2']='client2.ini'
    os.environ['INDEXSERVERNAME']= 'dasindex.lhd.nifs.ac.jp'


def igetfile(diagname, shotno, subshot, outputname):
    cwd = os.getcwd()
    outpath = os.path.join(cwd, outputname)
    cmdstr = ['igetfile', '-s', f'{shotno:d}', '-m', f'{subshot:d}', '-d', f'{diagname:s}', '-o', f'{outpath:s}']
    os.chdir(os.path.join('C:', os.sep, 'eg', 'GUIclient2'))
    # print(os.getcwd())
    run(cmdstr, shell=True)
    os.chdir(cwd)
    # print(os.getcwd())

def iregist(diagname, shotno, subshot, targetname, user_pw=None):
    if user_pw is None:
        cmdstr = "iregist -s %d -m %d -d %s -p %s" % (shotno, subshot, diagname, targetname)
    else:
        cmdstr = "iregist -s %d -m %d -d %s -p %s -u %s -w %s" % (
                shotno, subshot, diagname, targetname, user_pw[0], user_pw[1])
     #print cmdstr
    os.system(cmdstr)

def idelete(diagname, shotno, subshot):
    cmdstr = "idelete -s %d -m %d -d %s" % (shotno, subshot, diagname)
    #print cmdstr
    os.system(cmdstr)

def makeTempfilename(filename):
    return MakeUniqueFilename(filename, ".")

def MakeUniqueFilename(filename, folderpath=None, rlen=10):
    if not folderpath:
        folderpath = os.getcwd()
    base, ext = os.path.splitext(filename)
    filenames = os.listdir(folderpath)
    filename = base + '_%s' % RandomStr(rlen) + ext
    while (filename in filenames):
        filename = base + '_%s' % RandomStr(rlen) + ext
    return filename

def RandomStr(length=10):
    source = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join([random.choice(source) for i in range(length)])

def ReadFromZipfile(filepath):
    zf = zipfile.ZipFile(filepath, 'r')
    target = zf.namelist()[0]
    cbuff = zf.read(target)
    zf.close()
    return cbuff

def OpenFile(filepath):
    if filepath[-3:].upper() == 'ZIP':
        zf = zipfile.ZipFile(filepath, 'r')
        target = zf.namelist()[0]
        return zf, zf.open(target, 'r')
    else:
        return None, open(filepath, "r")

def CloseFile(zf, fp):
    if zf:
        fp.close()
        zf.close()
    else:
        fp.close()

def LoadEG(diagname, sn, sub=1, flg_remove=True):
    filepath = f'{diagname:s}@{sn:06d}_{sub:03d}.txt'
    igetfile(diagname, sn, sub, filepath)
    if not os.path.exists(filepath):
        return None
    eg = EG(filepath)
    if flg_remove:
        os.unlink(filepath)
    return eg



#------------------------------------------------------
class EG(object):
    def __init__(self, filepath, flg_bin=False):
        """
        self.data[idx]
        iline => self.line_index() 
        size = self.dimno + self.valno
        idx = self.line_size * iline + vidx
        """
        self.filepath = filepath
        self.keyvalue = {}
        if flg_bin:
            self.load_binary(filepath)
        else:
            self.load(filepath)

    def save_binary(self, filepath):
        serializer = Serializer()
        data = self.get_data_binary()
        line = serializer.pack(data)
        with open(filepath, 'wb') as fpo:
            fpo.write(line)
        return filepath

    def load_binary(self, filepath):
        serializer = Serializer()
        with open(filepath, 'rb') as fpi:
            line = fpi.read()
        bin_data = serializer.unpack(line)
        self.set_data_binary(bin_data)

    def set_data_binary(self, bin_data):
        data, head, dim_access_idx, weights = bin_data
        self.data = numpy.array(data)
        self.dim_access_idx = numpy.array(dim_access_idx)
        self.weights = numpy.array(weights)

        self.head = head[0].split('\n')
        self.set_headder(self.head)

    def get_data_binary(self):
        #self.dim_access_idx = [i for seq, i in sorted_seq_i]
        #print self.dim_access_idx
        #self.weights = self.calc_weights(self.dimsize)
        bin_data = [['double', self.data],
                ['string', "\n".join(self.head)],
                ['int', self.dim_access_idx],
                ['int', self.weights]]
        return bin_data


    def set_headder(self, buff):
        for l in buff:
            l = l.strip()
            if l[0] == '#':
                clm = l[1:].split('=')
                if len(clm) == 2:
                    self.keyvalue[clm[0].strip()] = clm[1].strip()
                continue

        cdimno = self.keyvalue.get('DimNo')
        cdimsize = self.keyvalue.get('DimSize')
        cvalno = self.keyvalue.get('ValNo')
        cdimnames = self.keyvalue.get('DimName')
        cdimunits = self.keyvalue.get('DimUnit')
        cvalnames = self.keyvalue.get('ValName')
        cvalunits = self.keyvalue.get('ValUnit')
        
        dimno = int(cdimno)
        valno = int(cvalno)
        line_size = dimno + valno
            
        dimsize = [int(c.strip()) for c in cdimsize.split(',')]
        line_size = dimno + valno
        data_size = line_size * reduce((lambda x, y: x * y), dimsize, 1)
        
        # make Headder info
        self.dimnames = [c.strip("' ") for c in cdimnames.split(',')] 
        self.dimunits = [c.strip("' ") for c in cdimunits.split(',')] 
        self.valnames = [c.strip("' ") for c in cvalnames.split(',')] 
        self.valunits = [c.strip("' ") for c in cvalunits.split(',')] 
        self.dimno = dimno
        self.valno = valno
        self.data_size = data_size
        self.line_size = line_size
        self.dimsize = dimsize
        #t_size, self.r_size = dimsize

    def load(self, filepath):
        zf, fp = OpenFile(filepath)
        l = fp.readline()
        self.head = []
        while l:
            l = l.strip()
            if len(l) == 0:
                l = fp.readline()
                continue
            if l[0] == '#':
                self.head.append(l)
                clm = l[1:].split('=')
                if len(clm) == 2:
                    self.keyvalue[clm[0].strip()] = clm[1].strip()
                l = fp.readline()
                continue
            break

        self.set_headder(self.head)

        # allocate memory
        self.data = numpy.zeros(self.data_size)
        num_seq = numpy.array([1] * (self.dimno+self.valno)) # dim value change cycle
        num_seq_max = numpy.array([1] * (self.dimno+self.valno)) # keep max dim value change cycle

        idx = 0
        values = [None] * self.line_size
        while l:
            l = l.strip()
            l = l.strip(',')
            if len(l) == 0:
                l = fp.readline()
                continue
            #print idx
            cidx = 0
            for c in l.split(','):
                try:
                    v = float(c)
                except ValueError:
                    v = numpy.nan

                if values[cidx] is None:
                    num_seq[cidx] = 0
                    values[cidx] = v
                else:
                    if values[cidx] == v:
                        num_seq[cidx] = num_seq[cidx] + 1
                    else:
                        if num_seq[cidx] > num_seq_max[cidx]:
                            num_seq_max[cidx] = num_seq[cidx]
                        num_seq[cidx] = 1
                        values[cidx] = v

                self.data[idx] = v
                idx = idx+1
                cidx = cidx+1
                if idx >= self.data_size:
                    break
            l = fp.readline()

        dim_seq = num_seq_max[:self.dimno]
        #print dim_seq
        sorted_seq_i = sorted([(seq, i) for i, seq in enumerate(dim_seq)], reverse=True)
        self.dim_access_idx = [i for seq, i in sorted_seq_i]
        #print self.dim_access_idx
        self.weights = self.calc_weights(self.dimsize)

        CloseFile(zf, fp)
        #self.timelist = self.times()
        #self.rlist = self.rs()

    def calc_weights(self, lst):
        def multiall(lst):
            return reduce((lambda x, y: x * y), lst, 1) 

        #ws = [multiall(lst[i+1:]) for i in range(len(lst)-1)]
        ws = [multiall([self.dimsize[idx] for idx in idxs]) 
                for idxs in [self.dim_access_idx[i:] for i in range(1, self.dimno)]]
        ws.append(1)
        ws = [w for i, w in sorted([(i, w) for i, w in zip(self.dim_access_idx, ws)])]
        #print ws
        return ws

    def get_line_index(self, idxs):
        #weights = self.weights(self.dimsize)
        return sum([i*w for i, w in zip(idxs, self.weights)])

    def get_line(self, idxs):
        idx = self.line_size * self.get_line_index(idxs)
        return self.data[idx: idx+self.line_size]

    def dim_indices(self, dim):
        pre = [0] * dim
        post = [0] * (self.dimno - dim -1)
        idxs = [pre + [i] + post for i in range(self.dimsize[dim])]
        #print idxs
        return idxs
    
    def dim_indices_at(self, dim, other_idxs):
        pre = other_idxs[0: dim]
        post = other_idxs[dim:]
        return [pre + [i] + post for i in range(self.dimsize[dim])]

    def dims_indices_2d(self, dimxy):
        dxidx = [i for i in range(self.dimsize[dimxy[0]])]
        dyidx = [i for i in range(self.dimsize[dimxy[1]])]
        gridxi, gridyi = numpy.meshgrid(dxidx, dyidx)
        return numpy.stack([numpy.ravel(gridxi.T), numpy.ravel(gridyi.T)]).T

    def dims(self, dim):
        seq_indices = self.dim_indices(dim)
        lidxs = numpy.array([self.get_line_index(idxs) for idxs in seq_indices])
        didxs = lidxs*self.line_size + dim
        return self.data[didxs]
        #return [self.get_line(indices)[dim] for indices in self.dim_indices(dim)]

    def dims_2d(self, dimxy):
        dimsx = self.dims(dimxy[0])
        dimsy = self.dims(dimxy[1])
        gridx, gridy = numpy.meshgrid(dimsx, dimsy)
        return numpy.stack([numpy.ravel(gridx.T), numpy.ravel(gridy.T)]).T
    
    def vdata(self, idxs, idx_v):
        lidx = self.get_line_index(idxs)
        idx = lidx*self.line_size + self.dimno + idx_v
        return self.data[idx]

    def data_values(self, seq_indices, idx_v):
        lidxs = numpy.array([self.get_line_index(idxs) for idxs in seq_indices])
        vidxs = lidxs*self.line_size + self.dimno + idx_v
        return self.data[vidxs]
        #return self.get_line(indices)[self.dimno+idx_v]

    def vidx(self, name):
        #print name, self.valnames
        if name in self.valnames:
            return self.valnames.index(name)
        else:
            return -1

    def trace_of_by_vidx(self, vidx, dim, other_idxs):
        seq_indices = self.dim_indices_at(dim, other_idxs)
        return self.data_values(seq_indices, vidx)

    def trace_of(self, name, dim, other_idxs):
        vidx = self.vidx(name)
        if vidx == -1:
            return []
        #print "TRACE_OF", other_idxs
        seq_indices = self.dim_indices_at(dim, other_idxs)
        return self.data_values(seq_indices, vidx)
        #return [self.get_line(indices)[self.dimno + vidx] 
        #        for indices in self.dim_indices_at(dim, other_idxs)]

    def trace_of_2d(self, name, dimxy):
        vidx = self.vidx(name)
        if vidx == -1:
            return []
        seq_indices = self.dims_indices_2d(dimxy)
        return self.data_values(seq_indices, vidx)   # 1d array which size = Ndimx * Ndimy


class EG_Writer(object):
    def __init__(self, sn, diagname):
        self.sn = sn
        self.diagname = diagname
        self.filepath = "%s@%d.txt" % (diagname, sn)

        self.dims = [] # [[name, unit, [values...]], [name, unit, [values...]], ...]
        self.values = []
        self.dimnames = []
        self.dimunits = []
        self.valnames = []
        self.valunits = []
        self.comments = []
    
    def weights(self, lst):
        if len(lst) == 1:
            return [1]
        w = self.weights(lst[1:])
        return [lst[1] * w[0]] + w

    def get_line_index(self, indices):
        weights = self.weights(self.dimsizes)
        idx = sum([i*w for i, w in zip(indices, weights)])
        return idx

    def set_dimno(self, n):
        for i in range(n):
            self.dims.append(("dim%d" % (i+1), "a.u.", [], None))
        self.make_dim_state()

    def set_dim(self, dimidx, name, unit, values, fmt=None):
        self.dims[dimidx] = (name, unit, values, fmt)
        self.make_dim_state()

    def append_dim(self, name, unit, values, fmt=None):
        self.dims.append((name, unit, values, fmt))
        self.make_dim_state()

    def make_dim_state(self):
        self.dimno = len(self.dims)
        self.dimsizes = [len(vs) for n, u, vs, fmt in self.dims]

    def valsize(self):
        return reduce((lambda x, y: x * y), self.dimsizes, 1) 

    def set_valno(self, n):
        for i in range(n):
            self.values.append(("'value%d'" % (i+1), "'a.u.'", [], None))
        self.valno = n

    def set_val(self, validx, name, unit, fmt=None):
        self.values[validx] = (name, unit, [0.0]*self.valsize(), fmt)

    def append_val(self, name, unit, fmt=None):
        self.values.append((name, unit, [0.0]*self.valsize(), fmt))
        self.valno = len(self.values)

    def set_values(self, validx, dimidx, other_indices, values):
        """
        set trace data on dim(dimidx) as the values for valname(validx).
        """
        dname, dunit, dvalues, fmt = self.dims[dimidx]
        indices = [other_indices[:dimidx] + [i] + other_indices[dimidx:] 
                for i, v in enumerate(dvalues)]
        for idxs, v in zip(indices, values):
            idx = self.get_line_index(idxs)
            self.values[validx][2][idx] = v

    def make_head(self):
        import datetime

        buff = []
        buff.append("# [Parameters]")
        buff.append("# Name = '%s'" % self.diagname)
        buff.append("# ShotNo = %d" % self.sn)
        dt = datetime.datetime.now() 
        buff.append("# Date = '%02d/%02d/%04d %02d:%02d'" % 
                (dt.month, dt.day, dt.year, dt.hour, dt.minute))

        buff.append("# DimNo = %d" % len(self.dims))

        dimnames = []
        units = []
        for name, unit, values, fmt in self.dims:
            dimnames.append("'%s'" % name)
            units.append("'%s'" % unit)

        buff.append("# DimSize = %s" % ", ".join(["%d" % v for v in self.dimsizes]))
        buff.append("# DimName = %s" % ", ".join(dimnames))
        buff.append("# DimUnit = %s" % ", ".join(units))

        buff.append("# ValNo = %d" % (self.valno))

        valnames = []
        units = []
        for name, unit, values, fmt in self.values:
            valnames.append("'%s'" % name)
            units.append("'%s'" % unit)

        buff.append("# ValName = %s" % ", ".join(valnames))
        buff.append("# ValUnit = %s" % ", ".join(units))
        buff.append("# ")
        buff.append("# [Comments]")
        for l in self.comments:
            buff.append("# %s" % l)
        buff.append("# ")
        buff.append("# [data]")
        return buff

    """
    def make_indices(self, indices):
        if (len(indices) == 1):
            return [[i] for i in range(indices[0])]
        else:
            idxs = self.make_indices(indices[1:])
            return [[j] + i for j in range(indices[0]) for i in idxs]
    """

    def make_indices(self, indices):
        if (len(indices) == 1):
            for i in range(indices[0]):
                yield [i]
        else:
            for j in range(indices[0]):
                for idxs in self.make_indices(indices[1:]):
                    yield [j] + idxs
  
    def save(self):
        buff = self.make_head()

        for idxs in self.make_indices(self.dimsizes):
            #print idxs
            line = []
            for idx, dim in zip(idxs, self.dims):
                dname, dunit, dvalues, fmt = dim
                if fmt is None:
                    line.append("%g" % dvalues[idx]) 
                else:
                    line.append(fmt % dvalues[idx]) 

            idx = self.get_line_index(idxs)
            for name, unit, values, fmt in self.values:
                if fmt is None:
                    line.append("%g" % values[idx]) 
                else:
                    line.append(fmt % values[idx]) 
            buff.append(", ".join(line))

        with open(self.filepath, 'w') as fp:
            fp.write("\n".join(buff))
            fp.write("\n")
        return self.filepath

class Serializer(object):
    """
    data serializer
    """
    def __init__(self, **args):
        self.datatypechars = ['x', 'c', 'b', 'B', 'h', 'H', 'i', 'I',
                'l', 'L', 'q', 'Q', 'f', 'd', 's', 'p', 'P'] 
        self.datatypedict = {'pad byte':'x',
                'char':'c',
                'signed char':'b',
                'unsigned char':'B',
                'short':'h',
                'unsigned short':'H',
                'int':'i',
                'unsigned int':'I',
                'long':'l',
                'unsigned long':'L',
                'long long':'q',
                'unsigned long long':'Q',
                'float':'f',
                'double':'d',
                'string':'s',
                'char []':'p',
                'void *':'P'}
        self.endian = args.get('endian', '>') # '>' Big endian '<' Little endian

    def pack(self, blocksOfValues):
        """
        return strings of list of lists
        """
        head = ""
        chars = ""
        for dtype, values in blocksOfValues:
            if dtype in self.datatypechars:
                datatype = dtype
            else:
                datatype = self.datatypedict.get(dtype, '')
            l = len(values)
            fmt = "%c%d%c" % (self.endian, l, datatype)
            if ( datatype == 's' ):
                dat = struct.pack(fmt, values)
            else:
                dat = struct.pack(fmt, *values)
            head = head + "%c,%d,%d,%c:" % (self.endian, l, len(dat), datatype)
            chars = chars + dat
        blocksize = len(blocksOfValues)
        headsize = len(head)
        eightbyte = struct.pack(">LL", headsize, blocksize)
        return eightbyte + head + chars
    
    def unpack(self, chars):
        """
        return list of lists for string
        """
        headsize, blocksize = struct.unpack('>LL', chars[:8])
        info = chars[8: 8+headsize]
        clm = info.split(':')[:-1]
        if blocksize != len(clm):
            return None
        idx = 8 + headsize
        values = []
        for c in clm:
            endian, lstr, size, datatype = c.split(',')
            l = int(lstr)
            size = int(size)
            fmt = "%c%d%c" % (endian, l, datatype)
            string = chars[idx: idx + size]
            idx = idx + size 
            values.append(list(struct.unpack(fmt, string)))
        return values

if __name__ == '__main__':
    egw = EG_Writer(1234, "test")
    times = [0.1*i for i in range(10)]
    egw.append_dim('time', 'sec', times)
    import math
    val1 = [math.sin(x) for x in times]
    val2 = [math.cos(x) for x in times]
    egw.append_val('val1', '.')
    egw.append_val('val2', '.')
    egw.set_values(0, 0, [], val1)
    egw.set_values(1, 0, [], val2)
    filepath = egw.save()
    eg = EG(filepath)
    print( eg.valnames )
    eg.save_binary('test.bin')
    egbin = EG('test.bin', flg_bin=True)
    print( egbin.valnames)
    eg = EG('tsmesh_v@131663.txt')
    eg.save_binary('tsmesh_v@131663.bin')







