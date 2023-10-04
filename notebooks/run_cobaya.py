import os
import time

from looti import cobaya_theory as cem

n_train_list = [500]

for n_train in n_train_list:

    emudir = "/work/bk935060/Looti/looti/emulators/justClsLensed_10sigma_%i_npca20/" %(n_train)
    savedir = "/work/bk935060/Looti/cobaya/looti_TTTEEE_10sigma_%i_npca20/chains/" %(n_train)

    print(emudir)
    print(savedir)

    looti_info = {
                "quantities": 
                            {'TT': { 'emuobj_directory': emudir},
                                'TE': { 'emuobj_directory': emudir},
                                'EE': { 'emuobj_directory': emudir}
                            },
    }

    my_looti = cem.Looti_Cobaya()
    my_looti.set_args(looti_info)
    my_looti.initialize()

    sigma = 10

    info = {
        "output":os.path.expanduser(savedir),
        "debug":False,
        "test":False,
        "stop_at_error":False,
        "timing":True,
        "resume":False,
        "force":True,
        "theory": {
            "looti": my_looti,
            #"classy": {
            #    "extra_args": {
            #        # default CLASS settings:
            #        "output": "tCl",
            #        #"non linear": "halofit",
            #        "lensing":"yes",
            #        #"compute damping scale":"yes",
            #        #'N_ncdm' : 1,
            #        #'m_ncdm' : 0.06,
            #        #'T_ncdm' : 0.71611,
            #        #"N_ur": 3.048,
            #    },
        },
        "likelihood": {
            "planck_2018_highl_plik.TTTEEE_lite":{},
            "planck_2018_lowl.TT_clik":{},
            "planck_2018_lowl.EE_clik":{},
            },
        'params': {
                    'A_s': {'latex': 'A_\\mathrm{s}',
                            'prior': {'max': 2.10100e-09 + sigma * 0.1016, 'min': 2.10100e-09 - sigma * 0.1016},
                            'proposal': 1e-11,   
                            'ref': {'dist': 'norm', 'loc': 2.10100e-09, 'scale':  1e-11}
                            },
                    'H0': {'latex': 'H_\\mathrm{0}',
                        'prior': {'max': 67.27 + sigma * 0.6, 'min': 67.27 - sigma * 0.6},
                        'proposal': 1.0,     
                        'ref': {'dist': 'norm', 'loc': 67.27, 'scale': 1.00}
                        },
                    'clamp': {'derived': 'lambda A_s, tau_reio: '
                                        '1e9*A_s*np.exp(-2*tau_reio)',
                            'latex': '10^9 A_\\mathrm{s} e^{-2\\tau}'},
                    'n_s': {'latex': 'n_\\mathrm{s}',
                            'prior': {'max': 0.9649 + sigma * 0.0044, 'min': 0.9649 - sigma * 0.0044},
                            'proposal': 0.004,    
                            'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}
                            },
                    'omega_b': {'latex': '\\Omega_\\mathrm{b} h^2',
                                'prior': {'max': 0.02236 + sigma * 0.00015, 'min': 0.02236 - sigma * 0.00015},
                                'proposal': 0.0002,
                                'ref': {'dist': 'norm','loc': 0.02236,'scale': 0.0002}
                                },
                    'omega_cdm': {'latex': '\\omega_\\mathrm{cdm} ',
                                'prior': {'max': 0.1202 + sigma * 0.0014, 'min': 0.1202 - sigma * 0.0014},
                                'proposal': 0.003,
                                'ref': {'dist': 'norm', 'loc': 0.1202, 'scale': 0.003}
                                },
                    'tau_reio': {'latex': '\\tau_\\mathrm{reio}',
                                'prior': {'max': 0.0544 + sigma * 0.0081, 'min': 0.0544 - sigma * 0.0070},
                                'proposal': 0.01,   #053075671
                                'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.01}
                                },
                    'A_planck': 1.0
        },  
        "sampler": { 
            "mcmc": {
                "drag":False,
                "learn_proposal": True,
                "oversample_power": 0.4,
                "proposal_scale":2.1,
                "Rminus1_stop": 0.01,
                "max_tries": 24000,
                #"covmat": os.path.expanduser("~/software/projects/looti/cobaya/lcdm.covmat"),
                },
            },
        }


    from cobaya.run import run

    updated_info, sampler = run(info)
