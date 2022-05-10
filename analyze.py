from common import _list_files
import collections
root_RawData = r"E:\Data\RawData"

def invalid_data_check():
    files = _list_files(root_RawData)

    res = {}
    for f in files:
        fl = f.split("_")
        if fl[3][:4] != "2021":
            continue
        print(fl[2],fl[3][4:],fl[4][4:],fl[5][4:])
        continue
        year = fl[3][:4]
        month = fl[3][4:6]
        loc = fl[2]

        # if loc != "133033":
        #     continue
        # print(f)
        if year in res:
            res[year].append([loc,month])
        else:
            res[year]=[[loc,month]]

        # if fl[3][4:6] not in ["06", "07", "08"]:
        #     if year in res:
        #         res[year].append([loc,month])
        #     else:
        #         res[year]=[[loc,month]]


    res = collections.OrderedDict(sorted(res.items()))
    for k,v in res.items():
        print(k)
        print(v)


changed = r"""2001 132034
2003 132034
2004 134033
2005 134033
2006 134033
2008 133033 134033
2009 134033
2010 133034 134033 135033
2011 133034
2012 134033
2013 134033
2014 134033
2015 133034
2017 134033
2018 133034
2020 134033
2021 134033"""


def new_release():
    new_root = r"E:\Data\new"
    res={}
    for dt in changed.split('\n'):
        fl = dt.split(" ")
        k = fl[0]
        v =fl[1:]
        # print(k,v) 
        if k in res:
            res[k].append(v)
        else:
            res[k] = v

    f_res = {}    
    files = _list_files(new_root)
    for f in files:
        fl = f.split('_')
        k = fl[3][:4]
        v = fl[2]
        if k in f_res:
            f_res[k].append(v)
        else:
            f_res[k] = [v]
    print(len(res),len(f_res))
    for k,v in res.items():
        if k in f_res:
            if res[k] != f_res[k]:
                print(k, res[k], "*", f_res[k])
            f_res.pop(k, None)
        else:
            print(k,"files not found")
        
    print(f_res)
    # print(len(res), len(files))

# new_release()

invalid_data_check()