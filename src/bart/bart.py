# Copyright 2016. The Regents of the University of California.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2016 Siddharth Iyer <sid8795@gmail.com>
# 2018 Soumick Chatterjee <soumick.chatterjee@ovgu.de> , WSL Support

# based on https://github.com/mrirecon/bart/blob/master/python/bart.py @ commit 3664133
# changes are marked with MV

import subprocess as sp
import tempfile as tmp
from . import cfl  # MV
import os
from .wslsupport import PathCorrection  # MV

def bart(nargout, cmd, *args, suppress_stdouterr: bool = False, **kwargs):  # MV

    if type(nargout) != int or nargout < 0:
        print("Usage: bart(<nargout>, <command>, <arguments...>)")
        return

    try:
        bart_path = os.environ['TOOLBOX_PATH']
    except:
        bart_path = None
    isWSL = False

    if not bart_path:
        if os.path.isfile('/usr/local/bin/bart'):
            bart_path = '/usr/local/bin'
        elif os.path.isfile('/usr/bin/bart'):
            bart_path = '/usr/bin'
        else:
            try:  # MV
                sp.check_call(['wsl', 'bart', 'version', '-V'], stdout=sp.DEVNULL, stderr=sp.DEVNULL)  # MV
                bart_path = '/usr/bin'
                isWSL = True
            except (sp.CalledProcessError, FileNotFoundError):  # MV
                raise Exception('Environment variable TOOLBOX_PATH is not set.')

    tmpdir = None  # MV
    if os.path.ismount('/tmp/share'):  # MV
        # indicates chroot in fire wip. In this case, the temporary file would automatically be located in /tmp/share which does not work  # MV
        tmpdir = '/tmp'  # MV
    with tmp.NamedTemporaryFile(dir=tmpdir) as tf:  # MV
        name = tf.name  # MV

    nargin = len(args)
    infiles = [name + 'in' + str(idx) for idx in range(nargin)]

    for idx in range(nargin):
        cfl.writecfl(infiles[idx], args[idx])

    args_kw = [("--" if len(kw)>1 else "-") + kw for kw in kwargs]
    infiles_kw = [name + 'in' + kw for kw in kwargs]
    for idx, kw in enumerate(kwargs):
        cfl.writecfl(infiles_kw[idx], kwargs[kw])

    outfiles = [name + 'out' + str(idx) for idx in range(nargout)]

    cmd = cmd.split(" ")

    if os.name =='nt':
        if isWSL:
            #For WSL and modify paths
            infiles_wsl = [PathCorrection(item) for item in infiles]  # MV
            infiles_kw_wsl = [PathCorrection(item) for item in infiles_kw]  # MV
            outfiles_wsl = [PathCorrection(item) for item in outfiles]  # MV
            # cmd = [PathCorrection(item) for item in cmd]  # MV
            args_infiles_kw = [item for pair in zip(args_kw, infiles_kw_wsl) for item in pair]
            shell_cmd = ['wsl', 'bart', *cmd, *args_infiles_kw, *infiles_wsl, *outfiles_wsl]  # MV
        else:
            #For cygwin use bash and modify paths
            infiles_cw = [item.replace(os.path.sep, '/') for item in infiles]  # MV
            infiles_kw_cw = [item.replace(os.path.sep, '/') for item in infiles_kw]  # MV
            outfiles_cw = [item.replace(os.path.sep, '/') for item in outfiles]  # MV
            cmd = [item.replace(os.path.sep, '/') for item in cmd]
            args_infiles_kw = [item for pair in zip(args_kw, infiles_kw_cw) for item in pair]
            shell_cmd = ['bash.exe', '--login',  '-c', os.path.join(bart_path, 'bart'), *cmd, *args_infiles_kw, *infiles_cw, *outfiles_cw]  # MV
            #TODO: Test with cygwin, this is just translation from matlab code
    else:
        args_infiles_kw = [item for pair in zip(args_kw, infiles_kw) for item in pair]
        shell_cmd = [os.path.join(bart_path, 'bart'), *cmd, *args_infiles_kw, *infiles, *outfiles]

    # run bart command
    ERR, stdout, stderr = execute_cmd(shell_cmd, suppress_stdouterr)  # MV

    # store error code, stdout and stderr in function attributes for outside access
    # this makes it possible to access these variables from outside the function (e.g "print(bart.ERR)")
    bart.ERR, bart.stdout, bart.stderr = ERR, stdout, stderr

    for elm in infiles:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    for elm in infiles_kw:
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    output = []
    for idx in range(nargout):
        elm = outfiles[idx]
        if not ERR:
            output.append(cfl.readcfl(elm))
        if os.path.isfile(elm + '.cfl'):
            os.remove(elm + '.cfl')
        if os.path.isfile(elm + '.hdr'):
            os.remove(elm + '.hdr')

    if ERR:
        print(f"Command exited with error code {ERR}.")
        return

    if nargout == 0:
        return
    elif nargout == 1:
        return output[0]
    else:
        return output


def clear_stdouterr():  # MV
    if hasattr(bart, 'ERR'):  # MV
        del bart.ERR  # MV
    if hasattr(bart, 'stdout'):  # MV
        del bart.stdout  # MV
    if hasattr(bart, 'stderr'):  # MV
        del bart.stderr  # MV


def execute_cmd(cmd, suppress_stdouterr):  # MV
    """
    Execute a command in a shell.
    Print and catch the output.
    """

    errcode = 0
    stdout = ""
    stderr = ""

    # remove empty strings from cmd
    cmd = [item for item in cmd if len(item)]

    # execute cmd
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)

    # print to stdout
    for stdout_line in iter(proc.stdout.readline, ""):
        stdout += stdout_line
        if not suppress_stdouterr:  # MV
            print(stdout_line, end="")  # MV
    proc.stdout.close()

    # in case of error, print to stderr
    errcode = proc.wait()
    if errcode:
        stderr = "".join(proc.stderr.readlines())
        if not suppress_stdouterr:  # MV
            print(stderr)  # MV
    proc.stderr.close()

    return errcode, stdout, stderr