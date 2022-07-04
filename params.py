params = {'solo_weakUK': { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 4,
                          'GRU': 16,
                           'cs':True,
                           'no_weak' : False,
                           'pre_trained': '../pretrained_models/UKDALE_100weak_0strong.h5'},
          'strong_weakUK': { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 3,
                          'GRU': 64,
                          'cs': False,
                            'no_weak' : False,
                          'pre_trained': '../pretrained_models/UKDALE_100weak_100strong.h5'},
          'mixed' : { 'drop': 0.1,
                         'kernel': 5,
                         'layers': 3,
                          'GRU': 64,
                        'cs': False,
                        'no_weak' : False,
                        'pre_trained': '../pretrained_models/UKDALE_REFIT_RESAMPLED_60weak_20strong.h5'},
          'strong_weakREFIT': { 'drop': 0.1,
                                'kernel':5,
                                'layers':4,
                                'GRU':64,
                                'cs': False,
                                'no_weak' : False,
                                'pre_trained': '../pretrained_models/REFIT_RESAMPLED_100weak_100strong_OPTIMIZED0.001_Trial_fixed_seed.h5'
              }}



uk_params = {'mean': 414.43,
            'std': 689.27}
refit_params = {'mean': 547.01,
                'std': 809.99}