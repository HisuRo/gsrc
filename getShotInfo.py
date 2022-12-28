#!/usr/bin/env python
# coding: shift_jis

import psycopg2

def cycles():
    connection = psycopg2.connect(
        database='db1', 
        user='guest', 
        password='guest', 
        host='egdb.lhd.nifs.ac.jp', 
        port=5432)

    cursor = connection.cursor()
    sql = (
        "select "
        "cycleno, startno, endno, startdate, enddate "
        "from exp_cycle"
        ) 
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    return rows

def cycle_of(target_sn):
    rows = cycles()
    for cn, sn, en, sd, ed in rows:
         if sn <= target_sn <= en:
             return cn
    return 0

def info(sn):
    connection = psycopg2.connect(
        database='db1', 
        user='guest', 
        password='guest', 
        host='egdb.lhd.nifs.ac.jp', 
        port=5432)

    cursor = connection.cursor()

    sql = (
        "select "
        "magneticfield, magneticaxis, quadruple, gamma, dDatacreationTime "
        "from explog2 where nshotnumber=%d"
        ) % sn
    cursor.execute(sql)
    rows = cursor.fetchall()

    BT, Rax, Bq, Gamma, date0 = rows[0]
    datet = (date0.year, date0.month, date0.day, date0.hour, date0.minute, date0.second, 0, 0, 0)

    cursor.close()
    connection.close()
    return BT, Rax, Bq, Gamma, datet, cycle_of(sn)

if __name__ == '__main__':
    import time
    import sys
    sn = int(sys.argv[1])
    bt, rax, bq, gamma, dt, cycle = info(sn)
    time_in_sec = time.mktime(dt)
    localtime = time.localtime(time_in_sec)
    print (time_in_sec, localtime, cycle)



