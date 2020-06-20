#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_track.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/22 9:20 PM   Haohe Liu      1.0         None
'''

# !/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate_track.py    
@Contact :   liu.8948@buckeyemail.osu.edu
@License :   (C)Copyright 2020-2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/4/22 8:04 PM   Haohe Liu      1.0         None
'''

import os.path as op
import numpy as np
import os
from evaluate.museval.aggregate import TrackStore
from evaluate.museval.mus_eval import bss_eval


def eval_mus_track(
        track,
        user_estimates,
        track_name="temp",
        output_dir=None,
        sample_rate=44100,
        mode='v4',
        win=1.0,
        hop=1.0
):
    """Compute all bss_eval metrics for the musdb track and estimated signals,
    given by a `user_estimates` dict.

    Parameters
    ----------
    track : Track
        dictionary, containing the ground truth as np.arrays.
    estimated_sources : Dict
        dictionary, containing the user estimates as np.arrays.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in

    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """

    # make sure to always build the list in the same order
    # therefore track.targets is an OrderedDict
    eval_targets = ['vocals', 'accompaniment']
    path = op.join(output_dir, track_name) + '.json'
    data = TrackStore(win=win, hop=hop, track_name=track_name,frames_agg="mean") # todo
    if (op.exists(path)):
        data.load_stored_result(path)
        return data
    # check if vocals and accompaniment is among the targets
    has_acc = all(x in eval_targets for x in ['vocals', 'accompaniment'])

    # add vocal accompaniment targets later
    if has_acc:
        # add vocals and accompaniments as a separate scenario
        audio_estimates = []
        audio_reference = []

        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(track[target])

        SDR, ISR, SIR, SAR = evaluate(
            audio_reference,
            audio_estimates,
            win=int(win * sample_rate),
            hop=int(hop * sample_rate),
            mode=mode
        )

        # iterate over all targets
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )

    if output_dir:
        # validate against the schema
        # data.validate() # todo

        try:
            # subset_path = op.join(
            #     output_dir,
            #     track.subset
            # )
            if not op.exists(output_dir):
                os.makedirs(output_dir)

            with open(
                    op.join(output_dir, track_name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except (IOError):
            pass

    return data


def pad_or_truncate(
        audio_reference,
        audio_estimates
):
    """Pad or truncate estimates by duration of references:
    - If reference > estimates: add zeros at the and of the estimated signal
    - If estimates > references: truncate estimates to duration of references

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    Returns
    -------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    """
    est_shape = audio_estimates.shape
    ref_shape = audio_reference.shape
    if est_shape[1] != ref_shape[1]:
        if est_shape[1] >= ref_shape[1]:
            audio_estimates = audio_estimates[:, :ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [
                    (0, 0),
                    (0, ref_shape[1] - est_shape[1]),
                    (0, 0)
                ],
                mode='constant'
            )

    return audio_reference, audio_estimates


def evaluate(
        references,
        estimates,
        win=1 * 44100,
        hop=1 * 44100,
        mode='v4',
        padding=True
):
    """BSS_EVAL images evaluation using metrics module

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    window : int, defaults to 44100
        window size in samples
    hop : int
        hop size in samples, defaults to 44100 (no overlap)
    mode : str
        BSSEval version, default to `v4`
    Returns
    -------
    SDR : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    ISR : np.ndarray, shape=(nsrc,)
        vector of Source to Spatial Distortion Image (ISR)
    SIR : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    SAR : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    """

    estimates = np.array(estimates)
    references = np.array(references)

    if padding:
        references, estimates = pad_or_truncate(references, estimates)

    SDR, ISR, SIR, SAR, _ = bss_eval(
        references,
        estimates,
        compute_permutation=False,
        window=win,
        hop=hop,
        framewise_filters=(mode == "v3"),
        bsseval_sources_version=False
    )

    return SDR, ISR, SIR, SAR


