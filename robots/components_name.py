# fmt: off
hand_joints_name = [
                'rh_WRJ2', 'rh_WRJ1', 
                'rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_LFJ5', 'rh_THJ5', 
                'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_LFJ4', 'rh_THJ4', 
                'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_LFJ3', 'rh_THJ3', 
                'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_LFJ2', 'rh_THJ2', 
                'rh_LFJ1', 'rh_THJ1',
]


useful_hand_link_names = dict(
    pre_palm = [
        'rh_forearm', 'rh_wrist'
    ],
    palm = [ 'rh_palm' ],
    th = [
        'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle', 'rh_thdistal',
    ],
    ff = [
        'rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal',
    ],
    mf = [
        'rh_mfknuckle', 'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal',
    ],
    rf = [
        'rh_rfknuckle', 'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal',
    ],
    lf = [
        'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal',
    ],
    tip = [
        'rh_thtip', 'rh_fftip', 'rh_mftip', 'rh_rftip', 'rh_lftip',
    ],  #tip don't have collision
    no_vol = [
        'rh_manipulator', 'rh_imu',
    ]
)


distal_link_names = [
            'rh_thdistal', 'rh_ffdistal', 'rh_mfdistal', 'rh_rfdistal', 'rh_lfdistal',
]
middle_link_names = [
            'rh_thmiddle', 'rh_ffmiddle', 'rh_mfmiddle', 'rh_rfmiddle', 'rh_lfmiddle',
]
proximal_link_names = [
            'rh_thproximal', 'rh_ffproximal', 'rh_mfproximal', 'rh_rfproximal', 'rh_lfproximal',
]

dmp_links_names = distal_link_names + middle_link_names + proximal_link_names



collision_pairs = [
    ('rh_lfproximal', 'rh_mfdistal'),
    ('rh_mfmiddle', 'rh_rfdistal'),
    ('rh_mfdistal', 'rh_lfmetacarpal'),
    ('rh_ffmiddle', 'rh_lfmiddle'),
    ('rh_rfmiddle', 'rh_lfmiddle'),
    ('rh_thdistal', 'rh_ffproximal'),
    ('rh_rfproximal', 'rh_ffdistal'),
    ('rh_palm', 'rh_lfdistal'),
    ('rh_mfproximal', 'rh_lfdistal'),
    ('rh_ffproximal', 'rh_lfmetacarpal'),
    ('rh_lfproximal', 'rh_ffmiddle'),
    ('rh_mfdistal', 'rh_lfdistal'),
    ('rh_ffproximal', 'rh_lfproximal'),
    ('rh_mfproximal', 'rh_rfdistal'),
    ('rh_ffmiddle', 'rh_lfdistal'),
    ('rh_rfmiddle', 'rh_lfdistal'),
    ('rh_thdistal', 'rh_ffdistal'),
    ('rh_mfproximal', 'rh_ffmiddle'),
    ('rh_thmiddle', 'rh_ffproximal'),
    ('rh_ffproximal', 'rh_mfdistal'),
    ('rh_ffmiddle', 'rh_rfdistal'),
    ('rh_thdistal', 'rh_mfdistal'),
    ('rh_mfmiddle', 'rh_rfmiddle'),
    ('rh_lfmiddle', 'rh_ffdistal'),
    ('rh_palm', 'rh_thdistal'),
    ('rh_rfproximal', 'rh_lfmetacarpal'),
    ('rh_rfproximal', 'rh_lfproximal'),
    ('rh_ffproximal', 'rh_mfproximal'),
    ('rh_lfmiddle', 'rh_mfdistal'),
    ('rh_thdistal', 'rh_mfproximal'),
    ('rh_ffdistal', 'rh_lfmetacarpal'),
    ('rh_lfproximal', 'rh_rfdistal'),
    ('rh_thmiddle', 'rh_ffdistal'),
    ('rh_rfproximal', 'rh_mfdistal'),
    ('rh_rfdistal', 'rh_lfmetacarpal'),
    ('rh_ffproximal', 'rh_lfmiddle'),
    ('rh_palm', 'rh_thmiddle'),
    ('rh_thdistal', 'rh_lfmiddle'),
    ('rh_mfproximal', 'rh_rfproximal'),
    ('rh_mfproximal', 'rh_rfmiddle'),
    ('rh_ffdistal', 'rh_mfdistal'),
    ('rh_mfmiddle', 'rh_lfmetacarpal'),
    ('rh_mfdistal', 'rh_rfdistal'),
    ('rh_lfproximal', 'rh_mfmiddle'),
    ('rh_ffmiddle', 'rh_rfmiddle'),
    ('rh_mfmiddle', 'rh_ffdistal'),
    ('rh_rfdistal', 'rh_lfdistal'),
    ('rh_ffproximal', 'rh_lfdistal'),
    ('rh_rfproximal', 'rh_ffmiddle'),
    ('rh_rfproximal', 'rh_lfmiddle'),
    ('rh_thdistal', 'rh_lfdistal'),
    ('rh_ffproximal', 'rh_rfproximal'),
    ('rh_ffproximal', 'rh_rfdistal'),
    ('rh_lfproximal', 'rh_ffdistal'),
    ('rh_lfproximal', 'rh_rfmiddle'),
    ('rh_thdistal', 'rh_mfknuckle'),
    ('rh_mfproximal', 'rh_lfmetacarpal'),
    ('rh_thdistal', 'rh_ffmiddle'),
    ('rh_palm', 'rh_lfproximal'),
    ('rh_mfproximal', 'rh_lfproximal'),
    ('rh_thdistal', 'rh_ffknuckle'),
    ('rh_mfproximal', 'rh_ffdistal'),
    ('rh_rfproximal', 'rh_lfdistal'),
    ('rh_lfmiddle', 'rh_rfdistal'),
    ('rh_ffmiddle', 'rh_lfmetacarpal'),
    ('rh_rfmiddle', 'rh_lfmetacarpal'),
    ('rh_mfmiddle', 'rh_lfmiddle'),
    ('rh_ffproximal', 'rh_mfmiddle'),
    ('rh_thdistal', 'rh_mfmiddle'),
    ('rh_rfmiddle', 'rh_ffdistal'),
    ('rh_ffdistal', 'rh_lfdistal'),
    ('rh_palm', 'rh_lfknuckle'),
    ('rh_ffmiddle', 'rh_mfdistal'),
    ('rh_rfmiddle', 'rh_mfdistal'),
    ('rh_ffdistal', 'rh_rfdistal'),
    ('rh_thmiddle', 'rh_ffmiddle'),
    ('rh_ffproximal', 'rh_rfmiddle'),
    ('rh_mfmiddle', 'rh_lfdistal'),
    ('rh_rfproximal', 'rh_mfmiddle'),
    ('rh_palm', 'rh_lfmiddle'),
    ('rh_mfproximal', 'rh_lfmiddle')
]

