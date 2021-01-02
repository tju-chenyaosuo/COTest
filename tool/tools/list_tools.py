def extract(l, s):
    res = []
    for elem in l:
        if s in elem:
            res.append(elem)
    return res


def strip(l):
    res = []
    for elem in l:
        res.append(elem.strip())
    return res


def get_first_word(l):
    res = []
    for elem in l:
        res.append(elem[:elem.find(' ')])
    return res


def get_diff(l1, l2):
    diff1 = []
    diff2 = []
    common = []
    for s1 in l1:
        if s1 not in l2:
            diff1.append(s1)
    for s2 in l2:
        if s2 not in l1:
            diff2.append(s2)
    for s1 in l1:
        if s1 in l2:
            common.append(s1)
    return [diff1, common, diff2]


def trim_empty_strings(l):
    res = []
    for string in l:
        if string != '':
            res.append(string)
    return res


def shuffle(l):
    import random
    res = l[:]
    for i in range(len(res)):
        inx = random.randint(0, len(res) - 1)
        tmp = res[i]
        res[i] = res[inx]
        res[inx] = tmp
    return res