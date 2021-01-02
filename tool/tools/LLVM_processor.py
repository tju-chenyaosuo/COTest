# -*- coding=utf-8 -*-
import tools.command_tools as ctools
import tools.file_tools as ftools


def clang_c_emit_llvm(clang, opts, lib, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_lib = ''
    if len(lib) != 0:
        str_lib = '-I ' + lib
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = clang + ' ' + str_opt + ' -c -emit-llvm ' + str_lib + ' ' + src + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


def clang_direct(clang, opts, lib, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_lib = ''
    if len(lib) != 0:
        str_lib = '-I ' + lib
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = clang + ' ' + str_opt + ' ' + str_lib + ' ' + src + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


def opt(opt_, opts, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = opt_ + ' ' + str_opt + ' ' + src + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


@DeprecationWarning
def lli(lli_, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    cmd = lli_ + ' ' + src + ' 1>' + out + ' 2>' + error
    return ctools.execmd_limit_time(cmd, limit)


def llc_direct(llc_, opts, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = llc_ + ' ' + str_opt + ' ' + src + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


def llc_filetype_obj(llc_, opts, src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = llc_ + ' ' + str_opt + ' -filetype=obj ' + src + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


def clang_lm(clang, opts, srcs, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_opt = ''
    if len(opts) != 0:
        str_opt = ' '.join(opts)
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = clang + ' ' + str_opt + ' -lm ' + ' '.join(srcs) + ' -o ' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)


def exe(src, out, error, limit):
    ftools.delete_if_exists(out)
    ftools.delete_if_exists(error)
    str_error = ''
    if len(error) != 0:
        str_error = '2>' + error
    cmd = src + ' 1>' + out + ' ' + str_error
    return ctools.execmd_limit_time(cmd, limit)