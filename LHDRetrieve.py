import os
import subprocess
import sys
import numpy

def ReadDatafile(filepath):
    data = numpy.fromfile(filepath, dtype=numpy.dtype('<f8'))
    return data

def ReadParameterfile(filepath, start=1):
    fpi = open(filepath, "r")
    lines = [l.strip() for l in fpi.readlines()]
    fpi.close()
    dic = {}
    for l in lines:
        clm = l.split(',')
        dic[clm[start]] = clm[start+1:]
    return dic

def Datafile(diagname, sn, subsn, ch, ext='.dat'):
    return f"{diagname:s}-{sn:d}-{subsn:d}-{ch:d}{ext:s}"

def Parameterfile(diagname, sn, subsn, ch, ext='.prm'):
    return f"{diagname:s}-{sn:d}-{subsn:d}-{ch:d}{ext:s}"

def Retrieve(diagname, sn, subsn, ch, force=False):
    if force or not os.path.exists(Datafile(diagname, sn, subsn, ch)):
        cmd = ['retrieve', f'{diagname:s}', f'{sn:d}', f'{subsn:d}', f'{ch:d}', '-V8']
        subprocess.run(cmd)

def Retrieve_ss(diagname, sn, subsn, ch, ss, force=False):
    if force or not os.path.exists(Datafile(diagname, sn, subsn, ch)):
        cmd = ['retrieve', f'{diagname:s}', f'{sn:d}', f'{subsn:d}', f'{ch:d}', '--ss', f'{ss[0]}:{ss[1]}', '-V8']
        subprocess.run(cmd)

def Retrieve_et(diagname, sn, subsn, ch, et, force=False):
    if force or not os.path.exists(Datafile(diagname, sn, subsn, ch)):
        cmd = ['retrieve', f'{diagname:s}', f'{sn:d}', f'{subsn:d}', f'{ch:d}', '--et', f'{et[0]}:{et[1]}s', '-V8']
        subprocess.run(cmd)

def Retrieve_t(diagname, sn, subsn, ch, force=False):
    print(type(ch))
    if force or not os.path.exists(Datafile(sn, subsn, ch, '.time')):
        print(type(ch))
        cmd = f'retrieve_t {diagname:s} {sn:d} {subsn:d} {ch:d} -D'
        os.system(cmd)

def RetrieveData(diagname, sn, subsn, ch, flg_remove=True):
    Retrieve(diagname, sn, subsn, ch, True)
    prmfile = Parameterfile(diagname, sn, subsn, ch)
    prms = ReadParameterfile(prmfile)
    datafile = Datafile(diagname, sn, subsn, ch)
    data = ReadDatafile(datafile)
    if flg_remove:
        os.unlink(prmfile)
        os.unlink(datafile)

    return data, prms

def RetrieveData_ss(diagname, sn, subsn, ch, ss, flg_remove=True):
    Retrieve_ss(diagname, sn, subsn, ch, ss, True)
    prmfile = Parameterfile(diagname, sn, subsn, ch)
    prms = ReadParameterfile(prmfile)
    datafile = Datafile(diagname, sn, subsn, ch)
    data = ReadDatafile(datafile)
    if flg_remove:
        os.unlink(prmfile)
        os.unlink(datafile)

    return data, prms

def RetrieveData_et(diagname, sn, subsn, ch, et, flg_remove=True):
    Retrieve_et(diagname, sn, subsn, ch, et, True)
    prmfile = Parameterfile(diagname, sn, subsn, ch)
    prms = ReadParameterfile(prmfile)
    datafile = Datafile(diagname, sn, subsn, ch)
    data = ReadDatafile(datafile)
    if flg_remove:
        os.unlink(prmfile)
        os.unlink(datafile)

    return data, prms

def RetrieveData_only_et(diagname, sn, subsn, ch, et, flg_remove=True):
    Retrieve_et(diagname, sn, subsn, ch, et, True)
    datafile = Datafile(diagname, sn, subsn, ch)
    data = ReadDatafile(datafile)
    if flg_remove:
        os.unlink(datafile)

    return data

def RetrieveTime(diagname, sn, subsn, ch, flg_remove=True):
    print(type(ch))
    Retrieve_t(diagname, sn, subsn, ch, True)
    tprmfile = Parameterfile(diagname, sn, subsn, ch, '.tprm')
    tprms = ReadParameterfile(tprmfile, start=0)
    datafile = Datafile(diagname, sn, subsn, ch, '.time')
    data = ReadDatafile(datafile)
    if flg_remove:
        os.unlink(tprmfile)
        os.unlink(datafile)
    
    return data, tprms
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    diagname = 'PCIFINE'
    sn = 153669
    subsn = 1
    ch = 20
    data, prms = RetrieveData(diagname, sn, subsn, ch)
    tdata, tprms = RetrieveTime(diagname, sn, subsn, ch)

    dt = float(tprms.get('ClockCycle')[0].replace('sec', ''))
    t0 = float(tprms.get('StartTiming')[0].replace('sec', ''))
    print (f"Start: {t0:g} (sec) Sampling: {dt:g} (sec)")
        
    plt.plot(tdata, data, '-')
    plt.show()


    



