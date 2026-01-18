import pandas as pd
import numpy as np
from itertools import combinations


def compute_deltaf(
    emgfile,
    smoothfits,
    average_method="test_unit_average",
    normalisation="False",
    recruitment_difference_cutoff=1.0,
    corr_cutoff=0.7,
    controlunitmodulation_cutoff=0.5,
    clean=True,
):
    """
    Quantify delta-F via paired-motor-unit analysis.
    (docstring unchanged for brevity)
    """

    # ------------------------------------------------------------------ #
    #  early exits / init
    # ------------------------------------------------------------------ #
    if emgfile["NUMBER_OF_MUS"] < 2:
        return pd.DataFrame(
            {"MU": np.nan * np.ones((1, 2)), "dF": np.nan}
        )

    combs        = combinations(range(emgfile["NUMBER_OF_MUS"]), 2)
    r_ret        = []
    dfret        = []
    testmu       = []
    ctrl_mod     = []
    rcrt_diff    = []
    controlmu    = []
    mucombo      = []

    # --------------------------------------------------------------- #
    #  helper to keep bookkeeping in ONE place
    # --------------------------------------------------------------- #
    def _store_pair(controlU: int):
        """append reporter/test MU indices that generated the current dfret"""
        controlmu.append(mucombo[-1][controlU - 1])
        testmu.append   (mucombo[-1][1 - controlU // 2])

    # ------------------------------------------------------------------ #
    #  main loop over MU pairs
    # ------------------------------------------------------------------ #
    for mu1_id, mu2_id in combs:

        # skip pairs when either MU has fewer than 2 firings
        if (len(emgfile["MUPULSES"][mu1_id]) < 2 or
            len(emgfile["MUPULSES"][mu2_id]) < 2):
            dfret     = np.append(dfret,     np.nan)
            r_ret     = np.append(r_ret,     np.nan)
            rcrt_diff = np.append(rcrt_diff, np.nan)
            ctrl_mod  = np.append(ctrl_mod,  np.nan)
            continue

        mucombo.append((mu1_id, mu2_id))

        mu1_times  = np.where(emgfile["BINARY_MUS_FIRING"][mu1_id] == 1)[0]
        mu2_times  = np.where(emgfile["BINARY_MUS_FIRING"][mu2_id] == 1)[0]
        mu1_rcrt, mu1_drcrt = mu1_times[0], mu1_times[-1]
        mu2_rcrt, mu2_drcrt = mu2_times[0], mu2_times[-1]

        # overlap region
        muoverlap = range(max(mu1_rcrt, mu2_rcrt), min(mu1_drcrt, mu2_drcrt))

        if len(muoverlap) < 2:         # too little overlap
            dfret     = np.append(dfret,     np.nan)
            r_ret     = np.append(r_ret,     np.nan)
            rcrt_diff = np.append(rcrt_diff, np.nan)
            ctrl_mod  = np.append(ctrl_mod,  np.nan)
            continue

        # correlation between the two smoothed discharge profiles
        r = pd.DataFrame(
            zip(smoothfits[mu1_id][muoverlap], smoothfits[mu2_id][muoverlap])
        ).corr()
        r_ret = np.append(r_ret, r.iloc[0, 1])

        # recruitment-time difference (s)
        rcrt_diff = np.append(
            rcrt_diff, np.abs(mu1_rcrt - mu2_rcrt) / emgfile["FSAMP"]
        )

        # ------------------------------------------------------------------ #
        #  work out which is reporter (control) and which is test
        # ------------------------------------------------------------------ #
        if mu1_rcrt < mu2_rcrt:                         # MU-1 recruited first
            controlU = 1
            if mu1_drcrt < mu2_drcrt:                   # shorten to overlap
                mu2_drcrt = mu1_drcrt
            df  = (smoothfits[mu1_id][mu2_rcrt] -
                   smoothfits[mu1_id][mu2_drcrt])

            mod = (np.nanmax(smoothfits[mu1_id][mu2_rcrt:mu2_drcrt]) -
                   np.nanmin(smoothfits[mu1_id][mu2_rcrt:mu2_drcrt]))
            ctrl_mod = np.append(ctrl_mod, mod)

            if normalisation == "ctrl_max_desc":
                k  = (smoothfits[mu1_id][mu2_rcrt] -
                      smoothfits[mu1_id][mu1_drcrt])
                df = df / k

            dfret = np.append(dfret, df)
            _store_pair(controlU)                 # ▲ new line

        elif mu1_rcrt > mu2_rcrt:                       # MU-2 recruited first
            controlU = 2
            if mu1_drcrt > mu2_drcrt:
                mu1_drcrt = mu2_drcrt
            df  = (smoothfits[mu2_id][mu1_rcrt] -
                   smoothfits[mu2_id][mu1_drcrt])

            mod = (np.nanmax(smoothfits[mu2_id][mu1_rcrt:mu1_drcrt]) -
                   np.nanmin(smoothfits[mu2_id][mu1_rcrt:mu1_drcrt]))
            ctrl_mod = np.append(ctrl_mod, mod)

            if normalisation == "ctrl_max_desc":
                k  = (smoothfits[mu2_id][mu1_rcrt] -
                      smoothfits[mu2_id][mu2_drcrt])
                df = df / k

            dfret = np.append(dfret, df)
            _store_pair(controlU)                 # ▲ new line

        else:                                            # same recruitment
            if mu1_drcrt > mu2_drcrt:                    # MU-1 fires longer
                controlU = 1
                df  = (smoothfits[mu1_id][mu2_rcrt] -
                       smoothfits[mu1_id][mu2_drcrt])

                mod = (np.nanmax(smoothfits[mu1_id][mu2_rcrt:mu2_drcrt]) -
                       np.nanmin(smoothfits[mu1_id][mu2_rcrt:mu2_drcrt]))
                ctrl_mod = np.append(ctrl_mod, mod)

                if normalisation == "ctrl_max_desc":
                    k  = (smoothfits[mu1_id][mu2_rcrt] -
                          smoothfits[mu1_id][mu1_drcrt])
                    df = df / k
            else:                                        # MU-2 fires longer
                controlU = 2
                df  = (smoothfits[mu2_id][mu1_rcrt] -
                       smoothfits[mu2_id][mu1_drcrt])

                mod = (np.nanmax(smoothfits[mu2_id][mu1_rcrt:mu1_drcrt]) -
                       np.nanmin(smoothfits[mu2_id][mu1_rcrt:mu1_drcrt]))
                ctrl_mod = np.append(ctrl_mod, mod)

                if normalisation == "ctrl_max_desc":
                    k  = (smoothfits[mu2_id][mu1_rcrt] -
                          smoothfits[mu2_id][mu2_drcrt])
                    df = df / k

            dfret = np.append(dfret, df)
            _store_pair(controlU)                 # ▲ new line

    # ------------------------------------------------------------------ #
    #  clean-up filters
    # ------------------------------------------------------------------ #
    if clean:
        keep = ((rcrt_diff > recruitment_difference_cutoff) &
                (r_ret      > corr_cutoff) &
                (ctrl_mod   > controlunitmodulation_cutoff))
        keep = np.asarray(keep, dtype=bool)  # ✅ force boolean array
        dfret[~keep] = np.nan

    # ------------------------------------------------------------------ #
    #  averaging or “all pairs” reporting
    # ------------------------------------------------------------------ #
    if average_method == "test_unit_average":
        dfret_ret = []
        mucombo_ret = []
        for test in range(emgfile["NUMBER_OF_MUS"]):
            idx = [i for i, t in enumerate(testmu) if t == test]
            dfret_ret.append(np.nanmean(dfret[idx]) if idx else np.nan)
            mucombo_ret.append(test)
    else:                                             # return all pairs
        dfret_ret   = dfret
        mucombo_ret = mucombo

    return pd.DataFrame({"MU": mucombo_ret, "dF": dfret_ret})
