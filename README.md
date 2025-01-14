Genral Analysis Tool kits made and modified by Tatsuhiro Nasu
Note: Make config.json by yourself to teach working server

- __init__.py
- README.md : this file
- LHDRetieve.py : modules to retrieve LHD raw data from LABCOM
- Retrieve_MWRM : modules to especially retrieve MWRM_** data as class object with class functions to process them.
- calc.py : general modules for calculation
- calib_table.csv : table including IQ amp. ratio, IQ phase difference, and offsets of IQ voltages for calibration of DBS comb2 signal.
- const.py : definitions of physical constants
- eg_mwrm.py : modules to get eg data of mwrm signal and according modules to process them
- getShotInfo.py : to get shot informations, Shot #, sub shot #, Bt, Rax, datetime, ...
- get_eg.py : modules to get eg data generally and to process them using myEgdb.py.
- myEgdb.py : modules to get eg data
- plot.py : modules to plot data generally
- proc.py : modules to process data, especially treating with array
- qmc.py : modules to read w7x data.
- read.py : modules to read data using LHD Retrieve.py
- read_w7x.py : modules to read w7x data.
- get_d3d.py : modules to retrieve d3d data and brief processing
- timetrace.py : general processing of timetrace signal
- system.py
- get_labcom.py
- get_eg_timetrace.py
- gadata.py

example of config.json
{
    "working_server": "Precision3450"
}