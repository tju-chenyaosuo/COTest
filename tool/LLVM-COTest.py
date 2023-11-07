# -*- coding=utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
import tools.command_tools as ctools
import tools.LLVM_processor as LLVM
import tools.file_tools as ftools
import tools.list_tools as ltools
import xgboost as xgb
import numpy as np
import random
import time
import os

LLVM_PATH_OLD = '/home/suocy/bin/llvm+clang-4.0.0/bin/'
LLVM_PATH_NEW = '/home/suocy/bin/llvm+clang-5.0.0/bin/'

LLVM_PATH = LLVM_PATH_NEW
CLANG = LLVM_PATH + 'clang'
OPT = LLVM_PATH + 'opt'
LLC = LLVM_PATH + 'llc'

CSMITH = '/home/suocy/bin/csmith_exnted0/bin/csmith'

# csmith include files
LIB = '/home/suocy/bin/csmith-2.3.0/include/csmith-2.3.0/'

MODEL = '/home/suocy/code/tmp-code/model-related/LLVM-4.model'
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(MODEL)

# We used same randomly generated program for all group in second study.
SEED_FILE = '/home/suocy/code/tmp-code/model-related/seed.txt'

# mask files
# Note that: there must be more efficient method to generate mask file, such as GA
maskFile = '/home/suocy/code/tmp-code/model-related/LLVM-4.0.0_5.0.0-mask.txt'

# test dir
TEST_BASE = '/home/suocy/data/tmp/tmp-code/'

# file use to normalization
# It contains only program features, but for our one-hot encoding for optimization settings,
# it is no need for the normalization of optimization settings.
CSMITH_ANNOTATION = '/home/suocy/code/tmp-code/model-related/LLVM-4.0.0-normalization.csv'

TRUNCATE_PROB = 0.5
EXE_FLAGS = 4
TIME_LIMIT = 24 * 60 * 60

cmd = 'pwd'
pwd = ctools.execmd(cmd).split('\n')[0]
pwdInfo = pwd.split('/')
FEATURE_FILE = '/'.join(pwdInfo[:-1]) + '/training_prob_t1.csv'

SEED_LIST = ftools.get_file_content(SEED_FILE).split('\n')
SEED_LIST = [int(SEED_LIST[i]) for i in range(len(SEED_LIST))]

maskList = ftools.get_file_lines(maskFile)
maskList = [maskList[i].split(',') for i in range(len(maskList))]
for i in range(len(maskList)):
    for j in range(len(maskList[i])):
        maskList[i][j] = int(maskList[i][j])
maskList = np.array(maskList)

SCALAR = MinMaxScaler()
content = np.loadtxt(CSMITH_ANNOTATION, delimiter=',')
SCALAR.fit(content)

startTime = time.time()
endTime = time.time()
programNum = 0


def predict(xgb_clf, feature):
    import numpy as np
    x = np.array(feature)
    res = xgb_clf.predict_proba(x)
    return [[res[i][1], i] for i in range(len(res))]


def compileExec(clang, clang2BcOpts, lib, clang2BcSrc, clang2BcOut, clang2BcError, clang2BcTimeout, clang2BcCrash,
                opt, optOpts, optOut, optError, optTimeout, optCrash,
                clang2ExeOpt, clang2ExeOut, clang2ExeError, clang2ExeTimeout, clang2ExeCrash,
                exeOut, exeError, exeTimeout, exeCrash,
                originalRes, miscompile,
                limit):
    if clang2BcSrc != '':
        timein = LLVM.clang_c_emit_llvm(clang, clang2BcOpts, lib, clang2BcSrc, clang2BcOut, clang2BcError, limit)
        if not timein:
            errorMsg = ftools.get_file_content(clang2BcError)
            ftools.put_file_content(clang2BcTimeout, errorMsg)
            return 'clang_timeout'
        if not os.path.exists(clang2BcOut) or os.path.getsize(clang2BcOut) == 0:
            errorMsg = ftools.get_file_content(clang2BcError)
            ftools.put_file_content(clang2BcCrash, errorMsg)
            return 'clang_crash'

    clang2ExeSrc = clang2BcOut

    flagInfo = ''
    if opt != '':
        flagInfo = ' '.join(optOpts) + '\n'
        optSrc = clang2BcOut
        timein = LLVM.opt(opt, optOpts, optSrc, optOut, optError, limit)
        if not timein:
            errorMsg = ftools.get_file_content(optError)
            ftools.put_file_content(optTimeout, flagInfo + errorMsg)
            return 'opt_timeout'
        if not os.path.exists(optOut) or os.path.getsize(optOut) == 0:
            errorMsg = ftools.get_file_content(optError)
            ftools.put_file_content(optCrash, flagInfo + errorMsg)
            return 'opt_crash'
        clang2ExeSrc = optOut

    timein = LLVM.clang_direct(clang, clang2ExeOpt, '', clang2ExeSrc, clang2ExeOut, clang2ExeError, limit)
    if not timein:
        errorMsg = ftools.get_file_content(clang2ExeError)
        ftools.put_file_content(clang2ExeTimeout, flagInfo + errorMsg)
        return 'clang2exe_timeout'
    if not os.path.exists(clang2ExeOut) or os.path.getsize(clang2ExeOut) == 0:
        errorMsg = ftools.get_file_content(clang2ExeError)
        ftools.put_file_content(clang2ExeCrash, flagInfo + errorMsg)
        return 'clang2exe_crash'

    exeSrc = clang2ExeOut
    timein = LLVM.exe(exeSrc, exeOut, exeError, limit)
    if not timein:
        errorMsg = ftools.get_file_content(exeError)
        ftools.put_file_content(exeTimeout, flagInfo + errorMsg)
        return 'exe_timeout'
    if not os.path.exists(exeOut) or os.path.getsize(exeOut) == 0:
        errorMsg = ftools.get_file_content(exeError)
        ftools.put_file_content(exeCrash, flagInfo + errorMsg)

    if originalRes != '':
        cmd = 'diff ' + originalRes + ' ' + exeOut
        diff = ctools.execmd(cmd)
        if len(diff) != 0:
            ftools.put_file_content(miscompile, flagInfo + diff)
            return 'miscompile'

    return 'success'


def getRandomSequence(l):
    import random
    res = l[:]
    for i in range(len(res)):
        inx = random.randint(0, len(res) - 1)
        tmp = res[i]
        res[i] = res[inx]
        res[inx] = tmp
    return res


def getOxOpt(level, llvmPath):
    cmd = llvmPath + 'llvm-as < /dev/null | ' + llvmPath + 'opt ' + level + ' -disable-output -debug-pass=Arguments 2>&1'
    optList = ctools.execmd(cmd).split('\n')[:2]
    optList = [optList[i][optList[i].index(':') + 3:] for i in range(len(optList))]
    optList = [optList[i].split(' ') for i in range(len(optList))]
    return '\n'.join(['\n'.join(optList[i]) for i in range(len(optList))])


def getClearDiff(diff):
    res = []
    for d in diff:
        if '>' not in d and '<' not in d and '-' not in d:
            res.append(d)
    return res


# Note that: All mappings are checked when we conducted second study in the paper.
def changeMapping(highId2LowId, diff):
    if 'd' in diff:
        splitInfo = diff.split('d')
        prefix = splitInfo[0]
        if ',' in prefix:
            prefixSplitInfo = prefix.split(',')
            n1 = int(prefixSplitInfo[0])
            n2 = int(prefixSplitInfo[1])
            for i in range(n1 - 1, n2):
                highId2LowId[i] = -1
            for i in range(n2, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] -= (n2 - n1) + 1
        else:
            n1 = int(prefix)
            highId2LowId[n1 - 1] = -1
            for i in range(n1, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] -= 1
    elif 'a' in diff:
        splitInfo = diff.split('a')
        prefix = splitInfo[0]
        suffix = splitInfo[1]
        if ',' in suffix:
            n1 = int(prefix)
            suffixSplitInfo = suffix.split(',')
            n2 = int(suffixSplitInfo[0])
            n3 = int(suffixSplitInfo[1])
            for i in range(n1, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] += (n3 -n2) + 1
        else:
            n1 = int(prefix)
            for i in range(n1, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] += 1
    elif 'c' in diff:
        splitInfo = diff.split('c')
        prefix = splitInfo[0]
        suffix = splitInfo[1]
        if ',' in prefix and ',' in suffix:
            prefixSplitInfo = prefix.split(',')
            suffixSplitInfo = suffix.split(',')
            n1 = int(prefixSplitInfo[0])
            n2 = int(prefixSplitInfo[1])
            n3 = int(suffixSplitInfo[2])
            n4 = int(suffixSplitInfo[3])
            for i in range(n1 - 1, n2):
                highId2LowId[i] = -1
            for i in range(n2, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] += (n4 - n3) - (n2 - n1)
        elif ',' in prefix and ',' not in suffix:
            prefixSplitInfo = prefix.split(',')
            n1 = int(prefixSplitInfo[0])
            n2 = int(prefixSplitInfo[1])
            for i in range(n1 - 1, n2):
                highId2LowId[i] = -1
            for i in range(n2, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] -= (n2 - n1)
        elif ',' not in prefix and ',' in suffix:
            n1 = int(prefix)
            suffixSplitInfo = suffix.split(',')
            n2 = int(suffixSplitInfo[0])
            n3 = int(suffixSplitInfo[1])
            highId2LowId[n1 - 1] = -1
            for i in range(n1, len(highId2LowId)):
                if highId2LowId[i] != -1:
                    highId2LowId[i] += (n3 - n2)
        elif ',' not in prefix and ',' not in suffix:
            n1 = int(prefix)
            highId2LowId[n1 - 1] = -1


def mapping(highId2LowId, diffContent):
    diffInx = len(diffContent) - 1
    while diffInx >= 0:
        changeMapping(highId2LowId, diffContent[diffInx])
        diffInx -= 1


def generateProgram(CSMITH, seed, src):
    cmd = CSMITH + ' --seed ' + str(seed) + ' > ' + src
    ctools.execmd_limit_time(cmd, 180)


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


def getOpts(LLVM_PATH):
    cmd = LLVM_PATH + 'llvm-as < /dev/null | ' + LLVM_PATH + 'opt -O3 -disable-output -debug-pass=Arguments 2>&1'
    optList = ctools.execmd(cmd).split('\n')[:2]
    optList = [optList[i][optList[i].index(':') + 3:] for i in range(len(optList))]
    optList = ' '.join(optList).split(' ')
    return optList


if __name__ == '__main__':
    # mapping
    OLD_OPTS = getOxOpt('-O3', LLVM_PATH_OLD)
    NEW_OPTS = getOxOpt('-O3', LLVM_PATH_NEW)
    ftools.delete_if_exists('old_opts.txt')
    ftools.delete_if_exists('new_opts.txt')
    ftools.put_file_content('old_opts.txt', OLD_OPTS)
    ftools.put_file_content('new_opts.txt', NEW_OPTS)
    cmd = 'diff new_opts.txt old_opts.txt'
    diffContent = ctools.execmd(cmd).split('\n')[:-1]
    NEW_OPTS = NEW_OPTS.split('\n')
    OLD_OPTS = OLD_OPTS.split('\n')
    highId2LowId = [i for i in range(len(NEW_OPTS))]
    diffContent = getClearDiff(diffContent)
    mapping(highId2LowId, diffContent)
    ftools.delete_if_exists('old_opts.txt')
    ftools.delete_if_exists('new_opts.txt')
    print(highId2LowId)

    # prepare for flags
    FULL = getOpts(LLVM_PATH)

    while endTime - startTime <= TIME_LIMIT:
        # prepare test environment
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

        # generate optimization setting from mask
        randomFeature = np.array([random.randint(0, 1) for i in range(len(FULL))])
        allFeature = []
        allOptList = []
        for i in range(len(maskList)):
            allFeature.append((randomFeature ^ maskList[i]).tolist())
        for i in range(len(allFeature)):
            optList = []
            for j in range(len(allFeature[i])):
                if allFeature[i][j] == 1:
                    optList.append(FULL[j])
            allOptList.append(optList)

        # mapping
        tmpFeature = allFeature[:]
        allFeature = []
        for flagsNum in range(len(tmpFeature)):
            mappingFeature = [0.0 for i in range(len(OLD_OPTS))]
            for colNum in range(len(tmpFeature[flagsNum])):
                if highId2LowId[colNum] != -1:
                    mappingFeature[highId2LowId[colNum]] = tmpFeature[flagsNum][colNum]
            allFeature.append(mappingFeature[:])

        # delete some program features that always be constant
        programFeature = ftools.get_file_content(FEATURE_FILE)[:-1].split(',')
        programFeature = [float(programFeature[i]) for i in range(len(programFeature))]
        reduce_list = [5, 10, 18, 19, 21, 22, 23, 25, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                       48, 71, 82, 92, 97, 105, 108, 109, 110]
        reducedProgramFeature = []
        for c in range(len(programFeature)):
            if c not in reduce_list:
                reducedProgramFeature.append(programFeature[c])
        reducedProgramFeature = np.array(reducedProgramFeature)

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
        mlFeature = [programAnnotationChangedFeature + allFeature[i] for i in range(len(allFeature))]
        mlFeature = np.array(mlFeature)

        # predict and sort
        predictProbs = predict(xgb_clf, mlFeature)
        predictProbs = sorted(predictProbs, reverse=True)
        allOptList = [allOptList[predictProbs[i][1]] for i in range(EXE_FLAGS)]

        if predictProbs[0][0] < TRUNCATE_PROB:
            programNum += 1
            endTime = time.time()
            ftools.delete_if_exists(testPath)
            continue

        clang2BcSrc = code + 'a.c'
        clang2BcOut = code + 'a.bc'
        codeError = code + 'error.txt'
        clang2BcTimeout = find + 'clang2Bc_timeout.txt'
        clang2BcCrash = find + 'clang2Bc_crash.txt'
        clang2ExeOut = code + 'a.out'
        clang2ExeTimeout = find + 'clang2Exe_timeout.txt'
        clang2ExeCrash = find + 'clang2Exe_crash.txt'
        exeOut = res + 'res0'
        resError = res + 'error.txt'
        exeTimeout = find + 'exe_timeout.txt'
        exeCrash = find + 'exe_crash.txt'
        bug = compileExec(CLANG, ['-O3', '-mllvm', '-disable-llvm-optzns'], LIB,
                          clang2BcSrc, clang2BcOut, codeError, clang2BcTimeout, clang2BcCrash,
                          '', '', '', '', '', '',
                          '', clang2ExeOut, codeError, clang2ExeTimeout, clang2ExeCrash,
                          exeOut, resError, exeTimeout, exeCrash,
                          '', '',
                          180)
        if bug != 'success':
            endTime = time.time()
            programNum += 1
            ftools.delete_if_exists(testPath)
            continue

        originalRes = exeOut

        bugCountsMap = {'miscompile': 0, 'exe_timeout': 0, 'clang2exe_crash': 0, 'clang2exe_timeout': 0,
                        'llc_crash': 0, 'llc_timeout': 0, 'opt_crash': 0, 'opt_timeout': 0}

        for flagsList in allOptList:
            optList = flagsList

            clang2BcOut = code + 'a.bc'
            codeError = code + 'error.txt'
            optOut = code + 'a.opt.bc'
            optTimeout = find + 'flags_opt_timeout' + str(bugCountsMap['opt_timeout']) + '.txt'
            optCrash = find + 'flags_opt_crash' + str(bugCountsMap['opt_crash']) + '.txt'
            clang2ExeOut = code + 'a.out'
            clang2ExeTimeout = find + 'flags_clang2Exe_timeout' + str(bugCountsMap['clang2exe_timeout']) + '.txt'
            clang2ExeCrash = find + 'flags_clang2Exe_crash' + str(bugCountsMap['clang2exe_crash']) + '.txt'
            exeOut = res + 'res1'
            resError = res + 'error.txt'
            exeTimeout = find + 'flags_exe_timeout' + str(bugCountsMap['exe_timeout']) + '.txt'
            exeCrash = find + 'flags_exe_crash-' + str(time.time()) + '.txt'
            miscompile = find + 'flags_miscompile' + str(bugCountsMap['miscompile']) + '.txt'
            bug = compileExec(CLANG, '', '', '', clang2BcOut, '', '', '',
                              OPT, optList, optOut, codeError, optTimeout, optCrash,
                              '', clang2ExeOut, codeError, clang2ExeTimeout, clang2ExeCrash,
                              exeOut, resError, exeTimeout, exeCrash,
                              originalRes, miscompile,
                              180)
            if bug != 'success':
                break
        programNum += 1
        endTime = time.time()
        ftools.delete_if_exists(code)
        ftools.delete_if_exists(res)
