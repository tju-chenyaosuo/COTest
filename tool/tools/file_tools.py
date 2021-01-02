# -*- coding=utf-8 -*-
import os
import tools.command_tools as ct


def delete_if_exists(path):
    if path == '' or path == '/' or path == '/*':
        return
    if os.path.exists(path):
        ct.execmd('rm -rf ' + path)


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        ct.execmd('mkdir ' + path)


def get_file_content(path):
    f = open(path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(path):
    c = get_file_content(path)
    if c == '':
        return ''
    if c[-1] == '\n':
        return c[:-1].split('\n')
    else:
        return c.split('\n')


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.close()