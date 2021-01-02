# -*- coding=utf-8 -*-
import random
import numpy as np
import tools.command_tools as ctools
import tools.file_tools as ftools
import tools.list_tools as ltools

def distance(m1, m2):
    res = 0
    for i in range(len(m1)):
        if m1[i] != m2[i]:
            res += 1
    return res


def exploreMask1(d, c):
    res0 = [0 for i in range(d)]
    res = []
    res.append(res0[:])
    while len(res) < c:
        res0 = [random.randint(0, 1) for i in range(d)]
        tmp_res0 = [res0 for i in range(len(res))]
        tmp_res0 = np.array(tmp_res0)
        tmp_res = np.array(res)
        distMap = tmp_res ^ tmp_res0
        dist = np.sum(distMap, 1)
        thisMin = np.min(dist)
        minDist = thisMin
        for i in range(len(res0)):
            res0[i] ^= 1
            distMap[:, i] ^= 1
            dist = np.sum(distMap, 1)
            thisMin = np.min(dist)
            if thisMin < minDist:
                res0[i] ^= 1
                distMap[:, i] ^= 1
            else:
                minDist = thisMin
        print(minDist)
        res.append(res0[:])
    return res


def getOpts(LLVM_PATH):
    cmd = LLVM_PATH + 'llvm-as < /dev/null | ' + LLVM_PATH + 'opt -O3 -disable-output -debug-pass=Arguments 2>&1'
    optList = ctools.execmd(cmd).split('\n')[:2]
    optList = [optList[i][optList[i].index(':') + 3:] for i in range(len(optList))]
    optList = ' '.join(optList).split(' ')
    return optList


def getFullOpts(GCC):
    opt3 = ctools.execmd(GCC + ' -O3 -Q --help=optimizers').split('\n')
    opt3 = ltools.extract(opt3, '[enabled]')
    opt3 = ltools.strip(opt3)
    opt3 = ltools.get_first_word(opt3)
    return opt3


if __name__ == '__main__':
    GCC = '/home/suocy/bin/gcc-4.5.0/bin/gcc'
    ALL_OPT = getFullOpts(GCC)
    maskl = exploreMask1(len(ALL_OPT), 2000)
    for i in range(len(maskl)):
        for j in range(len(maskl[i])):
            maskl[i][j] = str(maskl[i][j])
    ftools.put_file_content('mask.txt', '\n'.join([','.join(maskl[i]) for i in range(len(maskl))]))
