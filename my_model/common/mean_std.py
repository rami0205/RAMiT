def mean_std(scale, target_mode):
    if scale==2:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2397, 0.2290, 0.2389)
    elif scale==3:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2373, 0.2265, 0.2367)
    elif scale==4:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2352, 0.2244, 0.2349)
    elif target_mode=='light_dn': # image normalization with statistics from HQ sets
        mean = (0.4775, 0.4515, 0.4047)
        std = (0.2442, 0.2367, 0.2457)
    elif target_mode=='light_realdn': # image normalization with statistics from HQ sets
        mean = (0.0000, 0.0000, 0.0000)
        std = (1.0000, 1.0000, 1.0000)
    elif target_mode=='light_graydn': # image normalization with statistics from HQ sets
        mean = (0.4539,)
        std = (0.2326,)
    elif target_mode=='light_lle':
        mean = (0.1687, 0.1599, 0.1526)
        std = (0.1142, 0.1094, 0.1094)
    elif target_mode=='light_dr':
        mean = (0.5110, 0.5105, 0.4877)
        std = (0.2313, 0.2317, 0.2397)
    return mean, std