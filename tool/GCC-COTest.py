# -*- coding=utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import tools.command_tools as ctools
import tools.list_tools as ltools
import tools.file_tools as ftools
import xgboost as xgb
import numpy as np
import random
import time
import os

GCC_LOW = '/home/suocy/bin/gcc-4.4.0/bin/gcc'
GCC_HIGH = '/home/suocy/bin/gcc-4.6.0/bin/gcc'
GCC = GCC_HIGH

# must use csmith in HICOND
CSMITH = '/home/suocy/bin/csmith_exnted0/bin/csmith'
LIB = '/home/suocy/bin/csmith-2.3.0/include/csmith-2.3.0/'

# use seed file to provide seed(csmith)
SEED_FILE = '/home/suocy/code/tmp-code/model-related/seed.txt'
maskFile = '/home/suocy/code/tmp-code/model-related/GCC-4.4.0_4.6.0-mask.txt'
MODEL = '/home/suocy/code/tmp-code/model-related/GCC-4.4.0.model'
TEST_BASE = '/home/suocy/data/tmp/tmp-code/'

# used to normalization.
CSMITH_ANNOTATION = '/home/suocy/code/tmp-code/model-related/GCC-4.4.0-normalization.csv'

cmd = 'pwd'
pwd = ctools.execmd(cmd).split('\n')[0]
pwdInfo = pwd.split('/')
FEATURE_FILE = '/'.join(pwdInfo[:-1]) + '/training_prob_t1.csv'

TIME_LIMIT = 24 * 60 * 60
EXE_FLAGS = 4
TRUNCATE_PROB = 0.5
programNum = 0

SEED_LIST = ftools.get_file_content(SEED_FILE).split('\n')
SEED_LIST = [int(SEED_LIST[i]) for i in range(len(SEED_LIST))]

maskList = ftools.get_file_lines(maskFile)
maskList = [maskList[i].split(',') for i in range(len(maskList))]
for i in range(len(maskList)):
    for j in range(len(maskList[i])):
        maskList[i][j] = int(maskList[i][j])
maskList = np.array(maskList)

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(MODEL)

SCALAR = MinMaxScaler()
content = np.loadtxt(CSMITH_ANNOTATION, delimiter=',')
SCALAR.fit(content)

startTime = time.time()
endTime = time.time()

def predict(xgb_clf, feature):
    import numpy as np
    x = np.array(feature)
    res = xgb_clf.predict_proba(x)
    return [[res[i][1], i] for i in range(len(res))]


# hard coding, statistic from 40,000 program annotation.
def extractAnnotationFeature(src):
    types = ['XXX    times read thru a pointer', 'XXX    times written thru a pointer', 'XXX average alias set size',
             'XXX backward jumps', 'XXX const bitfields defined in structs', 'XXX forward jumps',
             'XXX full-bitfields structs in the program', 'XXX max block depth', 'XXX max dereference level',
             'XXX max expression depth', 'XXX max struct depth', 'XXX non-zero bitfields defined in structs',
             'XXX number of pointers point to pointers', 'XXX number of pointers point to scalars',
             'XXX number of pointers point to structs', 'XXX percent of pointers has null in alias set',
             'XXX percentage a fresh-made variable is used', 'XXX percentage an existing variable is used',
             'XXX percentage of non-volatile access', 'XXX stmts', 'XXX structs with bitfields in the program',
             'XXX times a bitfields struct on LHS', 'XXX times a bitfields struct on RHS',
             "XXX times a bitfields struct's address is taken", 'XXX times a non-volatile is read',
             'XXX times a non-volatile is write', 'XXX times a pointer is compared with address of another variable',
             'XXX times a pointer is compared with another pointer', 'XXX times a pointer is compared with null',
             'XXX times a pointer is dereferenced on LHS', 'XXX times a pointer is dereferenced on RHS',
             'XXX times a pointer is qualified to be dereferenced', 'XXX times a single bitfield on LHS',
             'XXX times a single bitfield on RHS', 'XXX times a variable address is taken',
             'XXX times a volatile is available for access', 'XXX times a volatile is read',
             'XXX times a volatile is write', 'XXX total number of pointers', 'XXX total union variables',
             'XXX volatile bitfields defined in structs', 'XXX zero bitfields defined in structs']
    content = ftools.get_file_content(src).split('\n')
    inx = content.index('/************************ statistics *************************')
    content = content[inx:]
    content = ltools.extract(content, 'XXX')
    content = [content[i].split(':') for i in range(len(content))]
    tmp_type = {}
    for t in types:
        tmp_type[t] = 0
    for c in content:
        t = c[0]
        v = c[1][1:]
        v = float(v)
        tmp_type[t] = v
    return [tmp_type[key] for key in types]


def doCompile(gcc, lib, opt, src, out, error, timetoutReport, crashReport, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    cmd = gcc + ' -I ' + lib + ' ' + opt + ' ' + src + ' -o ' + out + ' 2>' + error
    timein = ctools.execmd_limit_time(cmd, limit)
    if not timein:
        errorMsg = ftools.get_file_content(error)
        errorMsg = opt + '\n' + errorMsg
        ftools.put_file_content(timetoutReport, errorMsg)
        return 'compile_timeout'
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        errorMsg = ftools.get_file_content(error)
        errorMsg = opt + '\n' + errorMsg
        ftools.put_file_content(crashReport, errorMsg)
        return 'compile_crash'
    return 'success'


def doExec(out, res, opt, error, crashReport, timeoutReport, limit):
    ftools.delete_if_exists(res)
    ftools.delete_if_exists(error)
    cmd = out + ' 1>' + res + ' 2>' + error
    timein = ctools.execmd_limit_time(cmd, limit)
    if not timein:
        errorMsg = ftools.get_file_content(error)
        errorMsg = opt + '\n' + errorMsg
        ftools.put_file_content(timeoutReport, errorMsg)
        return 'exec_timeout'
    if not os.path.exists(res) or os.path.getsize(res) == 0:
        errorMsg = ftools.get_file_content(error)
        errorMsg = opt + '\n' + errorMsg
        ftools.put_file_content(crashReport, errorMsg)
        return 'exec_crash'
    return 'success'


def doDiff(file1, file2, opt, miscompileReport):
    cmd = 'diff ' + file1 + ' ' + file2
    diff = ctools.execmd(cmd)
    if len(diff) != 0:
        ftools.put_file_content(miscompileReport, opt + '\n' + diff)
        return 'miscompile'
    return 'success'


def testPass(gcc, lib, opt, src, out, compileTimeoutReport, compileCrashReport,
             res, execTimeoutReport, execCrashReport,
             oriRes, miscompileReport,
             error, limit):
    passed = doCompile(gcc, lib, opt, src, out, error, compileTimeoutReport, compileCrashReport, limit)
    if passed != 'success':
        return passed
    passed = doExec(out, res, opt, error, execCrashReport, execTimeoutReport, limit)
    if passed != 'success':
        return passed
    if len(oriRes) != 0:
        passed = doDiff(res, oriRes, opt, miscompileReport)
        if passed != 'success':
            return passed
    return passed


def getRandomSequence(l):
    import random
    res = l[:]
    for i in range(len(res)):
        inx = random.randint(0, len(res) - 1)
        tmp = res[i]
        res[i] = res[inx]
        res[inx] = tmp
    return res


def generateProgram(CSMITH, seed, src):
    cmd = CSMITH + ' --seed ' + str(seed) + ' > ' + src
    ctools.execmd_limit_time(cmd, 180)


def getFullOpts(GCC):
    opt3 = ctools.execmd(GCC + ' -O3 -Q --help=optimizers').split('\n')
    opt3 = ltools.extract(opt3, '[enabled]')
    opt3 = ltools.strip(opt3)
    opt3 = ltools.get_first_word(opt3)
    return opt3


def get_negation(flag):
    if '-fno-' not in flag:
        if '-fweb-' not in flag:
            return flag[:2] + 'no-' + flag[2:]
        else:
            return flag[6:7] + 'no-' + flag[7:]
    else:
        return flag[:2] + flag[5:]


def getNegativeOptStringList(GCC):
    opt3 = ctools.execmd(GCC + ' -O2 -Q --help=optimizers').split('\n')
    opt3 = ltools.extract(opt3, '[enabled]')
    opt3 = ltools.strip(opt3)
    opt3 = ltools.get_first_word(opt3)
    opt3 = [get_negation(opt3[i]) for i in range(len(opt3))]
    return opt3


if __name__ == '__main__':
    # mapping optimization settings
    lowList = getFullOpts(GCC_LOW)
    highList = getFullOpts(GCC_HIGH)
    lowFlagName2Index = {lowList[i]: i for i in range(len(lowList))}
    highFlagName2Index = {highList[i]: i for i in range(len(highList))}
    highId2LowId = [i for i in range(len(highList))]
    for flagName in highFlagName2Index:
        if flagName not in lowFlagName2Index:
            highId2LowId[highFlagName2Index[flagName]] = -1
        else:
            highId2LowId[highFlagName2Index[flagName]] = lowFlagName2Index[flagName]

    # prepare enable flags and "-fno-" disable flags
    ALL_OPT = getFullOpts(GCC)
    ALL_NEGATIVE_OPT = getNegativeOptStringList(GCC)

    while endTime - startTime <= TIME_LIMIT:
        # create test environment
        ftools.delete_if_exists(FEATURE_FILE)
        seed = SEED_LIST[programNum]
        testPath = TEST_BASE + str(programNum) + '-' + str(seed) + '/'
        code = testPath + 'code' + '/'
        find = testPath + 'find' + '/'
        res = testPath + 'res' + '/'
        ftools.create_dir_if_not_exist(testPath)
        ftools.create_dir_if_not_exist(code)
        ftools.create_dir_if_not_exist(find)
        ftools.create_dir_if_not_exist(res)
        src = code + 'a.c'
        generateProgram(CSMITH, seed, src)

        # generate optimization settings from mask
        randomFeature = np.array([random.randint(0, 1) for i in range(len(ALL_OPT))])
        featureLists = []
        for i in range(len(maskList)):
            featureLists.append((randomFeature ^ maskList[i]).tolist())

        # generate optimization flag sequences
        optLists = []
        for optNum in range(len(featureLists)):
            feature = featureLists[optNum]
            opt = []
            opt.extend(ALL_NEGATIVE_OPT)
            for i in range(len(feature)):
                if feature[i] == 1:
                    opt.append(ALL_OPT[i])
            if '-funit-at-a-time' not in opt and '-ftoplevel-reorder' in opt:
                opt.append('-fno-toplevel-reorder')
                featureLists[optNum][ALL_OPT.index('-ftoplevel-reorder')] = 0
            optLists.append(' '.join(opt))

        # convert high version optimization setting to low version(test version => train version)
        tmpFeatureList = featureLists[:]
        featureLists = []
        for i in range(len(tmpFeatureList)):
            mapedFeature = [0.0 for j in range(len(lowList))]
            for j in range(len(tmpFeatureList[i])):
                if highId2LowId[j] != -1:
                    mapedFeature[highId2LowId[j]] = tmpFeatureList[i][j]
            featureLists.append(mapedFeature[:])

        # delete some program features that always be constant
        programFeature = ftools.get_file_content(FEATURE_FILE)[:-1].split(',')
        programFeature = [float(programFeature[i]) for i in range(len(programFeature))]
        reduce_list = [5, 10, 18, 19, 21, 22, 23, 25, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                       48, 71, 82, 92, 97, 105, 108, 109, 110]
        reducedProgramFeature = []
        for c in range(len(programFeature)):
            if c not in reduce_list:
                reducedProgramFeature.append(programFeature[c])
        reducedProgramFeature = np.array(reducedProgramFeature, dtype=float)

        # extract features in csmith's annotation
        annotationFeature = extractAnnotationFeature(src)
        annotationFeature = np.array(annotationFeature, dtype=float)

        # combine
        programAnnotationFeature = np.hstack((reducedProgramFeature, annotationFeature))
        programAnnotationFeature = programAnnotationFeature.reshape(1, -1)

        # normalization
        programAnnotationChangedFeature = SCALAR.transform(programAnnotationFeature)
        programAnnotationChangedFeature = programAnnotationChangedFeature.tolist()
        programAnnotationChangedFeature = programAnnotationChangedFeature[0]

        # combine program feature and optimization settings
        mlFeature = [programAnnotationChangedFeature + featureLists[i] for i in range(len(featureLists))]
        mlFeature = np.array(mlFeature)

        # predict and sort
        predictProbs = predict(xgb_clf, mlFeature)
        predictProbs = sorted(predictProbs, reverse=True)
        optLists = [optLists[predictProbs[i][1]] for i in range(EXE_FLAGS)]

        print(seed)
        print(predictProbs[0][0])
        print(predictProbs[0][1])

        # threshold
        if predictProbs[0][0] < TRUNCATE_PROB:
            programNum += 1
            endTime = time.time()
            ftools.delete_if_exists(testPath)
            continue

        # start -O0
        out = code + 'a.o'
        error = code + 'error.txt'
        compileTimeoutReport = find + 'ori_compile_timeout.txt'
        compileCrashReport = find + 'ori_compile_crash.txt'
        tmpRes = res + 'ori_res.txt'
        exeTimeoutReport = find + 'ori_exe_timeout.txt'
        exeCrashReport = find + 'ori_exe_crash.txt'
        bugType = testPass(GCC, LIB, '-O0', src, out, compileTimeoutReport, compileCrashReport, tmpRes,
                           exeTimeoutReport, exeCrashReport,
                           '', '',
                           error, 180)
        # no need more test since it crashes at -O0
        if bugType != 'success':
            endTime = time.time()
            programNum += 1
            ftools.delete_if_exists(testPath)
            continue

        # start optimization setting
        oriRes = tmpRes
        BugMapping = {'compile_timeout': 0, 'compile_crash': 0, 'exec_timeout': 0, 'exec_crash': 0, 'miscompile': 0}
        resList = []
        for optList in optLists:
            optList = ' -O2 ' + optList
            out = code + 'a.o'
            error = code + 'error.txt'
            compileTimeoutReport = find + 'flags_compile_timeout' + str(BugMapping['compile_timeout']) + '.txt'
            compileCrashReport = find + 'flags_compile_crash' + str(BugMapping['compile_crash']) + '.txt'
            tmpRes = res + 'tmp_res.txt'
            exeTimeoutReport = find + 'flags_exe_timeout' + str(BugMapping['exec_timeout']) + '.txt'
            exeCrashReport = find + 'flags_exe_crash' + str(BugMapping['exec_crash']) + '.txt'
            miscompileReport = find + 'miscompile' + str(BugMapping['miscompile']) + '.txt'
            bugType = testPass(GCC, LIB, optList, src, out, compileTimeoutReport, compileCrashReport, tmpRes,
                               exeTimeoutReport, exeCrashReport,
                               oriRes, miscompileReport,
                               error, 180)
            if bugType != 'success':
                break

        programNum += 1
        endTime = time.time()
        ftools.delete_if_exists(code)
        ftools.delete_if_exists(res)