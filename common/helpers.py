import   numpy as np
import   array

def make_TH1F( h, ignore_binning = False):
    import   ROOT
    # remove infs from thresholds
    vals, thrs = h
    if ignore_binning:
        histo = ROOT.TH1F("h","h",len(vals),0,len(vals))
    else:
        histo = ROOT.TH1F("h","h",len(thrs)-1,array.array('d', thrs))
    for i_v, v in enumerate(vals):
        if v<float('inf'): # NAN protection
            histo.SetBinContent(i_v+1, v)
    return histo

def make_TGraph( coords ):
    import   ROOT
    tgraph = ROOT.TGraph(len(coords), array.array('d', [c[0] for c in coords]), array.array('d', [c[1] for c in coords]))
    return tgraph

def make_TH2F( h, ignore_binning = False):
    import   ROOT
    # remove infs from thresholds
    vals, thrs_x, thrs_y = h
    if ignore_binning:
        histo = ROOT.TH2F("h","h",len(vals[0]),0,len(vals[0]),len(vals),0,len(vals))
    else:
        histo = ROOT.TH2F("h","h",len(thrs_x)-1,array.array('d', thrs_x),len(thrs_y)-1,array.array('d', thrs_y))
    for ix, _ in enumerate(vals):
        for iy, v in enumerate(vals[ix]):
            if v<float('inf'): # NAN protection
                histo.SetBinContent(histo.FindBin(thrs_x[ix], thrs_y[iy]), v)
    return histo

import os, shutil
def copyIndexPHP( directory ):
    ''' Copy index.php to directory
    '''
    index_php = os.path.join( directory, 'index.php' )
    if not os.path.exists( directory ): os.makedirs( directory )
    shutil.copyfile( os.path.join(os.path.dirname(__file__), 'scripts/php/index.php'), index_php )

def _binning_equal(names_a, edges_a, names_b, edges_b, rtol=0.0, atol=0.0):
    """
    Strict equality by default; you can relax with rtol/atol if needed.
    names_*: tuple[str,...]
    edges_*: list[np.ndarray]
    """
    import numpy as _np
    if tuple(names_a) != tuple(names_b):
        return False
    if len(edges_a) != len(edges_b):
        return False
    for ea, eb in zip(edges_a, edges_b):
        if ea.shape != eb.shape:
            return False
        if not _np.allclose(ea, eb, rtol=rtol, atol=atol):
            return False
    return True


#import os, glob, subprocess
#
#def _derive_eos_remote_dir(local_dir: str) -> str:
#    """
#    Map a local dir like
#      /groups/hephy/cms/<user>/www/<rest>
#    to
#      root://eosuser.cern.ch///eos/user/<initial>/<user>/www/<rest>
#    using $CERN_USER (fallback to $USER).
#    """
#    user = os.environ.get("CERN_USER") or os.environ.get("USER")
#    if not user:
#        raise RuntimeError("Cannot determine username: set $CERN_USER or $USER.")
#
#    # split on 'www' and keep the suffix (including any leading '/')
#    parts = local_dir.split(os.sep + "www" + os.sep, 1)
#    if len(parts) == 2:
#        suffix = parts[1]  # e.g. 'SBIPDF/TFMC/tfmc_toy'
#        suffix = suffix.lstrip("/")
#        remote = f"root://eosuser.cern.ch///eos/user/{user[0]}/{user}/www/{suffix}"
#    else:
#        # no '/www/' in path → default to the user's www root
#        remote = f"root://eosuser.cern.ch///eos/user/{user[0]}/{user}/www"
#
#    return remote.rstrip("/")
#
#
#def make_and_push_gifs(
#    local_dir: str,
#    items = (("epoch_*.png", "epoch.gif"), ("norm_epoch_*.png", "norm_epoch.gif")),
#    delay: int = 10,
#    optimize: bool = True,
#):
#    """
#    Create animated GIFs from PNG sequences (locally) and xrdcp them to EOS.
#    remote_dir is auto-derived from local_dir by truncating at 'www'.
#    """
#    # sanity: tools
#    for tool in ("convert", "xrdcp"):
#        if subprocess.call(["which", tool], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
#            raise RuntimeError(f"Required tool '{tool}' not found in PATH.")
#
#    os.makedirs(local_dir, exist_ok=True)
#    remote_dir = _derive_eos_remote_dir(os.path.abspath(local_dir))
#
#    made = []
#    for pattern, gif_name in items:
#        frames = sorted(glob.glob(os.path.join(local_dir, pattern)))
#        if not frames:
#            print(f"[make_and_push_gifs] No frames for '{pattern}' in {local_dir}. Skipping.")
#            continue
#
#        out_gif = os.path.join(local_dir, gif_name)
#        cmd = ["convert", "-delay", str(delay), "-loop", "0"]
#        if optimize:
#            cmd += ["-dispose", "previous", "-layers", "Optimize"]
#        cmd += frames + [out_gif]
#
#        print(f"[make_and_push_gifs] Creating {out_gif} from {len(frames)} frames…")
#        subprocess.check_call(cmd)
#
#        dst = f"{remote_dir}/{gif_name}"
#        print(f"[make_and_push_gifs] xrdcp -> {dst}")
#        subprocess.check_call(["xrdcp", "-f", out_gif, dst])
#
#        made.append(out_gif)
#
#    return made
#
