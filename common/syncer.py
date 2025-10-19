'''
Prerequisites:
# add this to ~/.bash_profile
export KRB5_CONFIG=/mnt/hephy/cms/Tools/krb5.conf
Do 
kinit -fp <CERNUSER>@CERN.CH
not more than 24hrs before syncing.
The code wraps output functions and intercepts filenames.
All you have to do in your script is
import common.syncer
and the filenames written via
    TCanvas::Print
    pickle.dump
will be synced.
'''

import os, uuid
import subprocess 

def has_kerberos_token():
    try:
        # Run the `klist` command
        result = subprocess.run(['klist'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the output contains "No credentials cache found"
        if "No credentials cache found" in result.stderr:
            return False
        
        # If there's no error, and the command executed successfully, a token is present
        return result.returncode == 0
    except FileNotFoundError:
        # `klist` command is not available on the system
        print("Error: `klist` command not found. Ensure Kerberos is installed.")
        return False

# module singleton to keep track of files
file_sync_storage = []

## Wrap TCanvas class Print function
import ROOT
#class myTCanvas( ROOT.TCanvas ):
#    # recall the argument
#    def Print( self, *args):
#        logger.debug( "Appending file %s", args[0] )
#        file_sync_storage.append( args[0] )
#        # call original Print method 
#        super(myTCanvas, self).Print(*args)
## what could possibly go wrong.
#ROOT.TCanvas = myTCanvas 

_print = ROOT.TCanvas.Print 
def myPrint( self, *args):
    print( "Appending file %s"% args[0] )
    if not os.path.exists(os.path.dirname( args[0] ) ):
        os.makedirs( os.path.dirname( args[0] ) )
    file_sync_storage.append( args[0] )
    # call original Print method 
    _print(self, *args)

ROOT.TCanvas.Print = myPrint 

_saveAs = ROOT.TCanvas.SaveAs 
def mySaveAs( self, *args):
    print( "Appending file %s"% args[0] )
    if not os.path.exists(os.path.dirname( args[0] ) ):
        os.makedirs( os.path.dirname( args[0] ) )
    file_sync_storage.append( args[0] )
    # call original Print method 
    _saveAs(self, *args)

ROOT.TCanvas.SaveAs = mySaveAs 

from matplotlib import pyplot as plt
from functools import wraps
import os

## Save the original plt.savefig function before wrapping
#_savefig = plt.savefig
#
#@wraps(_savefig)  # Decorate after _savefig is defined
#def my_savefig(*args, **kwargs):
#    if args:  # Ensure args is not empty
#        file_path = args[0]
#        if not os.path.exists(os.path.dirname(file_path)):
#            os.makedirs(os.path.dirname(file_path))
#        file_sync_storage.append(file_path)
#    _savefig(*args, **kwargs)  # Call the original savefig function
#
## Overwrite plt.savefig with your custom function
#plt.savefig = my_savefig


from matplotlib import pyplot as plt

_savefig = plt.savefig
def my_savefig( *args, **kwargs):
    if not os.path.exists(os.path.dirname( args[0] ) ):
        os.makedirs( os.path.dirname( args[0] ) )
    file_sync_storage.append( args[0] )
    _savefig(*args, **kwargs)

plt.savefig = my_savefig

## Wrap pickle dump
#import pickle
## that's the old dump method
#pickle._dump = pickle.dump
#def syncer_pickle_dump( *args ):
#    # second argument is file handle!
#    if len(args)>1:
#        file_sync_storage.append( args[1].name )
#    else:
#        logger.warning( "Pickle dump called with less than two arguments... shouldn't happen." )
#    pickle._dump(*args)
## that's the new dump method
#pickle.dump = syncer_pickle_dump

# What happens on exit 
def write_sync_files_txt(output_filename = 'file_sync_storage.txt'):
    # No logger here, since it is already unloaded!
    # write file_sync_storage.txt because we can't scp in the container with kerberos authentication
    
    # find php files and copy these also
    for filename in list(set(file_sync_storage)):
        dir_path = os.path.dirname(os.path.realpath(filename))
        phpfiles = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.php')]
        for f in phpfiles:
            if f not in file_sync_storage:
                file_sync_storage.append(f)
    n_files = 0
    if len(file_sync_storage)>0:
        with open( output_filename, 'w' ) as outfile:
            for filename in file_sync_storage:
                if 'www/' in filename:
                    # for rsync cmd with relative path
                    outfile.write('{filename}\n'.format(filename=os.path.expanduser(os.path.expandvars(filename)).replace('www/','www/./')))
                    n_files+=1
                else:
                    print(("Will not sync %s" % filename))
        print(("Analysis.Tools.syncer: Written %i files to %s for rsync." % (n_files, output_filename)))
    return n_files

gif_cmds = []
def makeRemoteGif(directory, pattern, name, delay=50):
    if "CERN_USER" not in os.environ: 
        print("To sync with CERN www directory, you need to set $CERN_USER")
        return
    if '/www/' not in directory:
        print("makeRemoteGif: /www/ not found. Do nothing.")
        return
    directory_ = '/'+(directory.split('/www/')[-1])
    cern_user = os.environ["CERN_USER"]
    remotedir = "/eos/user/{INITIAL}/{CERN_USER}/www/{directory}".format( CERN_USER=cern_user, INITIAL=cern_user[0],directory=directory_)
    cmd = "ssh {CERN_USER}@lxplus.cern.ch 'convert -delay {delay} -loop 0 {remotedir}/{pattern} {remotedir}/{name}.gif'".format(CERN_USER=cern_user, delay=delay, remotedir=remotedir, pattern=pattern, name=name)
    if cmd not in gif_cmds:
        gif_cmds.append( cmd )

def make_gifs( cmds=gif_cmds ):
    ret = []
    for cmd in cmds:
        print("make gif:", cmd )
        output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
        if error:
            ret.append(cmd)
    return ret

#def sync(gifs=False):
#
#    if "CERN_USER" not in os.environ: 
#        print("To sync with CERN www directory, you need to set $CERN_USER")
#        return
#
#    #if not has_kerberos_token():
#    #    print("No kerberos token. Do nothing.")
#    #    return 
#    
#    global file_sync_storage
#    global gif_cmds 
#
#    if len(file_sync_storage)==0:
#        print("No files for syncing.")
#        return
#
#    filename = '/tmp/%s.txt'%uuid.uuid4()
#
#    if write_sync_files_txt(filename)==0: return 
#
#    cmd = "rsync -avR  `cat %s` ${CERN_USER}@lxplus.cern.ch:/eos/user/$(echo ${CERN_USER} | head -c 1)/${CERN_USER}/www/" % filename
#    #cmd = "xrdcp -f  `cat %s` root://eosuser.cern.ch//eos/user/$(echo ${CERN_USER} | head -c 1)/${CERN_USER}/www/" % filename
#    print(cmd)
#    output,error = subprocess.Popen(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
#    #os.remove(filename)
#    file_sync_storage = []
#    if gifs:
#        gif_cmds = make_gifs(gif_cmds) 
#
#    return #output, error
import os
import uuid
import shlex
import pathlib
import subprocess

# ----------------- CONFIG -----------------
EOS_HOST = "root://eosuser.cern.ch"
WWW_MARKER = "/www/"
# ------------------------------------------

def _run(cmd):
    """Run a shell command and print stderr on failure."""
    p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}"
        )
    return p.stdout

def _rel_under_www(src_abs):
    """
    Strip everything up to and including '/www/'.
    '/groups/.../www/t_sch/a.txt' -> 't_sch/a.txt'
    Fallback: if no '/www/' present, drop the leading slash to avoid mirroring root.
    """
    s = os.path.abspath(src_abs)
    if WWW_MARKER in s:
        return s.split(WWW_MARKER, 1)[1]
    return str(pathlib.Path(s).as_posix()).lstrip("/")

def _eos_user_base(user):
    return f"/eos/user/{user[0]}/{user}/www"

def _eos_url(abs_eos_path):
    # xrdcp requires the double slash before absolute EOS path
    return f"{EOS_HOST}//{abs_eos_path}"

def sync(gifs=False):
    """
    Sequential upload to EOS using xrdfs/xrdcp, driven by your file list.
    Path mapping: everything AFTER '/www/' in the source path is mirrored under .../www on EOS.
    """
    if "CERN_USER" not in os.environ:
        print("To sync with CERN www directory, you need to set $CERN_USER")
        return

    global file_sync_storage
    global gif_cmds

    if len(file_sync_storage) == 0:
        print("No files for syncing.")
        return

    # Use your existing writer for the file list
    filename = f"/tmp/{uuid.uuid4()}.txt"
    if write_sync_files_txt(filename) == 0:
        return

    user = os.environ["CERN_USER"]
    eos_base = _eos_user_base(user)

    # Read file list
    with open(filename, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    # Build copy plan and collect directories
    plan = []            # (src_abs, dest_abs)
    dirs_needed = set()
    for src in files:
        src_abs = os.path.abspath(src)
        rel = _rel_under_www(src_abs)
        dest_abs = f"{eos_base}/{rel}"
        plan.append((src_abs, dest_abs))
        dirs_needed.add(os.path.dirname(dest_abs))

    # Create directories on EOS (sequential)
    for d in sorted(dirs_needed):
        cmd = f'xrdfs {EOS_HOST} mkdir -p {shlex.quote(d)}'
        try:
            _run(cmd)
        except RuntimeError as e:
            # If directory exists, xrdfs -p is usually fine; otherwise show the error and continue
            print(e)

    # Copy files (sequential)
    for src_abs, dest_abs in plan:
        dest_spec = _eos_url(dest_abs)
        cmd = f'xrdcp -f {shlex.quote(src_abs)} {shlex.quote(dest_spec)}'
        print(cmd)
        _run(cmd)

    # Cleanup / post
    # os.remove(filename)
    file_sync_storage = []
    if gifs:
        gif_cmds = make_gifs(gif_cmds)

    return


import atexit
atexit.register( sync )
atexit.register( make_gifs )
