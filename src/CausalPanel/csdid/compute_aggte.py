import numpy as np
from .aggregators import aggte_simple, aggte_group, aggte_dynamic, aggte_calendar

def compute_aggte(MP,
                  agg_method="group",
                  balance_e=None,
                  min_e=float('-inf'),
                  max_e=float('inf'),
                  na_rm=False,
                  bstrap=None,
                  biters=None,
                  cband=None,
                  alp=None,
                  clustervars=None,
                  call=None):
    """
    Compute aggregated ATT estimates using aggregation methods:
    - 'simple': Weighted average over all post-treatment ATT(g,t)
    - 'group': Average over group-level ATTs
    - 'dynamic': Event-time specific ATT dynamics
    - 'calendar': Time-period specific ATT averages
    """

    # Unpack MP object and configuration
    group = np.array(MP['group'])
    time = np.array(MP['time'])
    att = np.array(MP['att'])
    inffunc = MP['inffunc']
    dp = MP['DIDparams']
    data = dp['data']
    tlist = np.array(dp['tlist'])
    glist = np.array(dp['glist'])
    gname = dp['gname']
    tname = dp['tname']
    idname = dp['idname']
    n = MP['n']

    # Override params
    clustervars = dp.get('clustervars') if clustervars is None else clustervars
    bstrap = dp.get('bstrap') if bstrap is None else bstrap
    biters = dp.get('biters') if biters is None else biters
    alp = dp.get('alp') if alp is None else alp
    cband = dp.get('cband') if cband is None else cband

    dp.update({
        'clustervars': clustervars,
        'bstrap': bstrap,
        'biters': biters,
        'alp': alp,
        'cband': cband
    })

    valid_methods = ['simple', 'group', 'dynamic', 'calendar']
    if agg_method not in valid_methods:
        raise ValueError(f"`agg_method` must be one of {valid_methods}")

    # Remove NAs if requested
    if na_rm:
        notna = ~np.isnan(att)
        group, time, att = group[notna], time[notna], att[notna]
        inffunc = inffunc[:, notna]
        glist = np.sort(np.unique(group))

        if agg_method == "group":
            gnotna = []
            for g in glist:
                mask = (group == g) & (g <= time)
                gnotna.append(np.any(~np.isnan(att[mask])))
            glist = glist[np.array(gnotna)]
            mask = np.isin(group, glist)
            group, time, att = group[mask], time[mask], att[mask]
            inffunc = inffunc[:, mask]

    if not na_rm and np.any(np.isnan(att)):
        raise ValueError("Missing values in ATT(g,t). Set `na_rm=True` to drop them.")

    dta = data[data[tname] == tlist[0]]

    # Recode group and time to contiguous indices
    original_time = time.copy()
    original_group = group.copy()
    original_glist = glist.copy()
    original_tlist = tlist.copy()
    all_keys = np.sort(np.unique(np.concatenate((original_tlist, original_glist))))
    uniq_map = {val: idx + 1 for idx, val in enumerate(all_keys)}
    rev_map = {v: k for k, v in uniq_map.items()}

    def orig2t(x): return uniq_map.get(x, None)
    def t2orig(x): return rev_map.get(x, None)

    time = np.array([orig2t(x) for x in original_time])
    group = np.array([orig2t(x) for x in original_group])
    glist = np.array([orig2t(x) for x in original_glist])
    tlist = np.array(list(set(time)))
    max_t = max(time)

    # Compute weights
    weights = dta['.w'].to_numpy()
    pg_map = {g: np.mean(weights * (dta[gname].to_numpy() == g)) for g in original_glist}
    pg = np.array([pg_map[gi] for gi in original_group])
    pgg = np.array([pg_map[g] for g in original_glist])
    keepers = [i for i in range(len(group)) if group[i] <= time[i] <= (group[i] + max_e)]
    G = np.array([orig2t(g) for g in dta[gname].to_numpy()])

    if agg_method == "simple":
        return aggte_simple(att, inffunc, pg, keepers, weights, G, group, dp)
    elif agg_method == "group":
        return aggte_group(att, inffunc, group, time, glist, pg, pgg, keepers, weights, G, dp, original_glist)
    elif agg_method == "dynamic":
        return aggte_dynamic(att, inffunc, original_time, original_group, pg, max_t, balance_e,
                             min_e, max_e, weights, G, group, dp, t2orig)
    elif agg_method == "calendar":
        return aggte_calendar(att, inffunc, group, time, pg, weights, G, tlist, glist, dp, t2orig)
    else:
        raise NotImplementedError(f"Aggregation method `{agg_method}` not implemented.")


'''
def compute_aggte(MP,
                      agg_method    = "group",
                      balance_e     = None,
                      min_e         = float('-inf'),
                      max_e         = float('inf'),
                      na_rm         = False,
                      bstrap        = None,
                      biters        = None,
                      cband         = None,
                      alp           = None,
                      clustervars   = None,
                      call          = None):
    # =============================================================================
    # unpack MP object
    # =============================================================================
        group       = np.array( MP['group'] )
        t           = np.array( MP['time']     )
        att         = np.array( MP['att'] )
        dp          = MP['DIDparams']
        tlist       = np.array( dp['tlist'] )
        glist       = np.array( dp['glist'] )
        data        = dp['data']
        inffunc     = MP['inffunc']
        n           = MP['n']
        gname       = dp['gname']
        tname       = dp['tname']
        idname      = dp['idname']  

        
        if clustervars is None:
            clustervars = dp['clustervars']
        if bstrap is None:
            bstrap = dp['bstrap']
        if biters is None:
            biters = dp['biters']
        if alp is None:
            alp = dp['alp']
        if cband is None:
            cband = dp['cband']



        # Overwrite MP objects (to compute bootstrap)
        MP['DIDparams']['clustervars'] = clustervars
        MP['DIDparams']['bstrap'] = bstrap
        MP['DIDparams']['biters'] = biters
        MP['DIDparams']['alp'] = alp
        MP['DIDparams']['cband'] = cband
        
    # =============================================================================
    #  Treat data
    # =============================================================================

        if agg_method not in ["simple", "dynamic", "group", "calendar"]:
            raise ValueError("`agg_method` must be one of ['simple', 'dynamic', 'group', 'calendar']")
        # Removing missing values    
        if na_rm:
            notna = ~np.isnan(att)
            group = group[notna]
            t = t[notna]
            att = att[notna]
            inffunc = inffunc[:, notna]
            glist = np.sort(np.unique(group))
        
            if agg_method == "group":
                gnotna = []
                for g in glist:
                    indices = np.where((group == g) & (g <= t))
                    is_not_na = np.any(~np.isnan(att[indices]))
                    gnotna.append(is_not_na)
                
                gnotna = np.array(gnotna)
                glist = glist[gnotna]
                not_all_na = np.isin(group, glist)
                group = group[not_all_na]
                t = t[not_all_na]
                att = att[not_all_na]
                inffunc = inffunc[:, not_all_na]
                glist = np.sort(np.unique(group))

        
        if (not na_rm) and np.any(np.isnan(att)):
            raise ValueError("Missing values at att_gt found. If you want to remove these, set `na_rm = True`.")

        dta = data[data[tname] == tlist[0]]

    # =============================================================================
    #  Treat data 2
    # =============================================================================
        originalt = t
        originalgroup = group
        originalglist = glist
        originaltlist = tlist
        # In case g's are not part of tlist
        originalgtlist = np.sort(np.unique(np.concatenate((originaltlist, originalglist))))
        uniquet = list(range(1, len(originalgtlist) + 1))
        
        # Function to switch from "new" t values to original t values
        def t2orig(t):
            return originalgtlist[uniquet.index(t) if t in uniquet else -1]
        
        # Function to switch between "original" t values and new t values
        def orig2t(orig):
            new_t = [uniquet[i] for i in range(len(originalgtlist)) if originalgtlist[i] == orig]
            out = new_t[0] if new_t else None
            return out
        
        t     = [orig2t(orig) for orig in originalt]
        group = [orig2t(orig) for orig in originalgroup]
        glist = [orig2t(orig) for orig in originalglist]
        tlist = np.asarray(list(set(t)))
        maxT  = max(t)
            
        # Set the weights
        # return data.columns
        weights_ind = dta['.w'].to_numpy()
        
        # We can work in overall probabilities because conditioning will cancel out
        # since it shows up in numerator and denominator
        pg = np.array([np.mean(weights_ind * (dta[gname].to_numpy() == g)) for g in originalglist])
        
        # Length of this is equal to the number of groups
        pgg = pg

        # Same but length is equal to the number of ATT(g,t)
        pg = [pg[glist.index(g)] for g in group]  
        
        # Which group time average treatment effects are post-treatment
        keepers = [i for i in range(len(group)) if group[i] <= t[i] <= (group[i] + max_e)] ### added second condition to allow for limit on longest period included in att
        
        # n x 1 vector of group variable
        G = [orig2t(g) for g in dta[gname].to_numpy()]

    # =============================================================================
    #  simple
    # =============================================================================

        if agg_method == "simple":
            # Simple ATT
            # Averages all post-treatment ATT(g,t) with weights given by group size
            pg = np.array(pg)
            simple_att = np.sum(att[keepers] * pg[keepers]) / np.sum(pg[keepers])
            if np.isnan(simple_att):
                simple_att = None
        
            # Get the part of the influence function coming from estimated weights
            simple_wif = wif(keepers, pg, weights_ind, G, group)
        
            # Get the overall influence function
            simple_if = get_agg_inf_func(att = att , 
                                        inffunc = inffunc , 
                                        whichones = keepers ,
                                        weights_agg = np.array(pg)[keepers]/np.sum(np.array(pg)[keepers]) , 
                                        wif = simple_wif )[:, None]
        
            # Get standard errors from the overall influence function
            simple_se = get_se(simple_if, dp)
            
            if simple_se is not None:
                if simple_se <= np.sqrt(np.finfo(float).eps) * 10:
                    simple_se = None
        
            return {
                    "agg_method": "simple",
                    "att": simple_att,
                    "se": simple_se,
                    "influence_function": {'simple_att': simple_if},
                    "DIDparams": dp
                }
            
    # =============================================================================
    #  GRoup
    # =============================================================================

        if agg_method == "group":
            group = np.array(group)
            t = np.array(t) 
            pg = np.array(pg) 
            selective_att_g = [np.mean(att[( group== g) & (t >= g) & (t <= (group + max_e))]) for g in glist]
            selective_att_g = np.asarray(selective_att_g)
            selective_att_g[np.isnan(selective_att_g)] = None
        
            selective_se_inner = [None] * len(glist)
            for i, g in enumerate(glist):
                whichg = np.where(np.logical_and.reduce((group == g, g <= t, t <= (group + max_e))))[0]
                weightsg =  pg[whichg] / np.sum(pg[whichg])
                inf_func_g = get_agg_inf_func(att = att , 
                                                inffunc = inffunc , 
                                                whichones = whichg ,
                                                weights_agg = weightsg , 
                                                wif = None)[:, None]
                se_g = get_se(inf_func_g, dp)
                selective_se_inner[i] = {'inf_func': inf_func_g, 'se': se_g}
                
            # recover standard errors separately by group   
            selective_se_g = np.asarray([item['se'] for item in selective_se_inner]).T
            
            selective_se_g[selective_se_g <= np.sqrt(np.finfo(float).eps) * 10] = None
            
            selective_inf_func_g = np.column_stack([elem["inf_func"] for elem in selective_se_inner])
    
            # use multiplier bootstrap (across groups) to get critical value
            # for constructing uniform confidence bands   
            selective_crit_val = stats.norm.ppf(1 - alp/2)
            
            if dp['cband']:
                if not dp['bstrap']:
                    print("Used bootstrap procedure to compute simultaneous confidence band")
            
                selective_crit_val = bootstrap_influence_function(selective_inf_func_g, dp)['crit_val']
            
                if np.isnan(selective_crit_val) or np.isinf(selective_crit_val):
                    print("Simultaneous critical value is NA. This probably happened because we cannot compute t-statistic (std errors are NA). We then report pointwise conf. intervals.")
                    selective_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if selective_crit_val < stats.norm.ppf(1 - alp/2):
                    print("Simultaneous conf. band is somehow smaller than pointwise one using normal approximation. Since this is unusual, we are reporting pointwise confidence intervals")
                    selective_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if selective_crit_val >= 7:
                    print("Simultaneous critical value is arguably 'too large' to be reliable. This usually happens when the number of observations per group is small and/or there is not much variation in outcomes.")
    
            # get overall att under selective treatment timing
            # (here use pgg instead of pg because we can just look at each group)            
            selective_att = np.sum(selective_att_g * pgg) / np.sum(pgg)
            
            # account for having to estimate pgg in the influence function    
            selective_wif = wif(keepers = np.arange(1, len(glist)+1)-1, 
                                pg  = pgg, 
                                weights_ind = weights_ind, 
                                G = G, 
                                group = group)
            
            # get overall influence function   
            selective_inf_func = get_agg_inf_func(att = selective_att_g, 
                                                inffunc = selective_inf_func_g,
                                                whichones = np.arange(1, len(glist)+1)-1, 
                                                weights_agg = pgg/np.sum(pgg),
                                                wif = selective_wif)[:, None]    
            
            # get overall standard error        
            selective_se = get_se(selective_inf_func, dp)
            print("selective_se: ", selective_se)
            if not np.isnan(selective_se):
                if selective_se <= np.sqrt(np.finfo(float).eps) * 10:
                    selective_se = None
        
            return {
                    "agg_method": "group",
                    "att": selective_att,
                    "se": selective_se,
                    "egt": originalglist,
                    "att_egt": selective_att_g,
                    "se_egt": selective_se_g,
                    "crit_val_egt": selective_crit_val,
                    "influence_function": {'selective_inf_func_g': selective_inf_func_g, 
                                           'selective_inf_func': selective_inf_func},
                    "DIDparams": dp
                }
    
    # =============================================================================
    #  Dynamic
    # =============================================================================

        if agg_method== "dynamic":
            # event times
            # this looks at all available event times
            # note: event times can be negative here.
            # note: event time = 0 corresponds to "on impact"
            eseq = np.unique(np.array(originalt) - np.array(originalgroup) ) # Subtract corresponding elements and convert to NumPy array
            eseq = np.sort(eseq)  # Sort the unique values in ascending order
        
            # if the user specifies balance_e, then we are going to
            # drop some event times and some groups; if not, we just
            # keep everything (that is what this variable is for)
            originalt = np.array(originalt)
            originalgroup = np.array(originalgroup)
            pg = np.array(pg)
            include_balanced_gt = np.repeat(True, len(originalgroup))

            if balance_e is not None:
                include_balanced_gt = (t2orig(maxT) - originalgroup >= balance_e)        
                eseq = np.unique(originalt[include_balanced_gt] - originalgroup[include_balanced_gt])
                eseq = np.sort(eseq)      
                eseq = eseq[(eseq <= balance_e) & (eseq >= balance_e - t2orig(maxT) + t2orig(1))]
            eseq = eseq[(eseq >= min_e) & (eseq <= max_e)]

            dynamic_att_e = []
            for e in eseq:
                whiche = np.where((originalt - originalgroup == e) & include_balanced_gt)
                atte = att[whiche]
                pge = pg[whiche] / np.sum(pg[whiche])
                dynamic_att_e.append(np.sum(atte * pge))
            
            dynamic_se_inner = []
            for e in eseq:
                whiche = np.where((originalt - originalgroup == e) & (include_balanced_gt) )[0]
                pge = pg[whiche] / sum(pg[whiche])
                wif_e = wif(whiche, 
                            pg, 
                            weights_ind, 
                            G, 
                            group)
                inf_func_e = get_agg_inf_func(att         = att, 
                                            inffunc     = inffunc, 
                                            whichones   = whiche, 
                                            weights_agg = pge, 
                                            wif         = wif_e)[:, None]
                se_e = get_se(inf_func_e, dp)
                dynamic_se_inner.append({'inf_func': inf_func_e, 'se': se_e})

            dynamic_se_e = np.array([item['se'] for item in dynamic_se_inner]).T
            
            dynamic_se_e[dynamic_se_e <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
            
            dynamic_inf_func_e = np.column_stack([item['inf_func'] for item in dynamic_se_inner])
                    
            dynamic_crit_val = stats.norm.ppf(1 - alp/2)
            if dp['cband']:
                if not dp['bstrap']:
                    print('Used bootstrap procedure to compute simultaneous confidence band')
                dynamic_crit_val = bootstrap_influence_function(dynamic_inf_func_e, dp)['crit_val']
            
                if np.isnan(dynamic_crit_val) or np.isinf(dynamic_crit_val):
                    print('Simultaneous critical value is NA. This probably happened because we cannot compute t-statistic (std errors are NA). We then report pointwise conf. intervals.')
                    dynamic_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if dynamic_crit_val < stats.norm.ppf(1 - alp/2):
                    print('Simultaneous conf. band is somehow smaller than pointwise one using normal approximation. Since this is unusual, we are reporting pointwise confidence intervals')
                    dynamic_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if dynamic_crit_val >= 7:
                    print("Simultaneous critical value is arguably 'too large' to be reliable. This usually happens when the number of observations per group is small and/or there is not much variation in outcomes.")

            epos = eseq >= 0
            dynamic_att = np.mean(np.array(dynamic_att_e)[epos])
            dynamic_inf_func = get_agg_inf_func(att         = np.array(dynamic_att_e)[epos],
                                                inffunc     = np.array(dynamic_inf_func_e[:, epos]),
                                                whichones   = np.arange(1, np.sum(epos)+1)-1,
                                                weights_agg = np.repeat(1 / np.sum(epos), np.sum(epos)),
                                                wif=None)[:, None]
            
            dynamic_se = get_se(dynamic_inf_func, dp)
            if not np.isnan(dynamic_se):
                if dynamic_se <= np.sqrt(np.finfo(float).eps) * 10:
                    dynamic_se = np.nan
        
            return {
                    "agg_method": "dynamic",
                    "att": dynamic_att,
                    "se": dynamic_se,
                    "event_times": eseq,
                    "att_by_event_time": dynamic_att_e,
                    "se_by_event_time": dynamic_se_e,
                    "influence_function": {'dynamic_inf_func_e': dynamic_inf_func_e,
                                           'dynamic_inf_func': dynamic_inf_func},
                    "critical_value": dynamic_crit_val,
                    "DIDparams": dp
                }
            
    # =============================================================================
    #  Calendar
    # =============================================================================

        if agg_method == "calendar":
            minG = min(group)
            calendar_tlist = tlist[tlist >= minG]
            pg = np.array(pg)
            calendar_att_t = []
            group = np.array(group)
            t = np.array(t)
            for t1 in calendar_tlist:
                whicht = np.where((t == t1) & (group <= t))[0]
                attt = att[whicht]
                pgt = pg[whicht] / np.sum(pg[whicht])
                calendar_att_t.append(np.sum(pgt * attt))
                
            # get standard errors and influence functions
            # for each time specific att
            calendar_se_inner = []
            for t1 in calendar_tlist:
                which_t = np.where((t == t1) & (group <= t))[0]
                pgt = pg[which_t] / np.sum(pg[which_t])
                wif_t = wif(keepers=which_t, 
                            pg=pg, 
                            weights_ind=weights_ind, 
                            G=G, 
                            group=group)
                inf_func_t = get_agg_inf_func(att=att, 
                                                inffunc=inffunc, 
                                                whichones=which_t, 
                                                weights_agg=pgt, 
                                                wif=wif_t)[:, None]
                se_t = get_se(inf_func_t, dp)
                calendar_se_inner.append({"inf_func": inf_func_t, "se": se_t})
        
        
        
            # recover standard errors separately by time
            calendar_se_t = np.array([se["se"] for se in calendar_se_inner]).T
            calendar_se_t[calendar_se_t <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
            
            # recover influence function separately by time
            calendar_inf_func_t = np.column_stack([se["inf_func"] for se in calendar_se_inner])
        
            # use multiplier boostrap (across groups) to get critical value
            # for constructing uniform confidence bands
            calendar_crit_val = stats.norm.ppf(1 - alp/2)
            
            if dp['cband']:
                if not dp['bstrap']:
                    warnings.warn('Used bootstrap procedure to compute simultaneous confidence band')
            
                # mboot function is not provided, please define it separately
                calendar_crit_val = bootstrap_influence_function(calendar_inf_func_t, dp)['crit_val']
            
                if np.isnan(calendar_crit_val) or np.isinf(calendar_crit_val):
                    warnings.warn('Simultaneous critical value is NA. This probably happened because we cannot compute t-statistic (std errors are NA). We then report pointwise conf. intervals.')
                    calendar_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if calendar_crit_val < stats.norm.ppf(1 - alp/2):
                    warnings.warn('Simultaneous conf. band is somehow smaller than pointwise one using normal approximation. Since this is unusual, we are reporting pointwise confidence intervals.')
                    calendar_crit_val = stats.norm.ppf(1 - alp/2)
                    dp['cband'] = False
            
                if calendar_crit_val >= 7:
                    warnings.warn("Simultaneous critical value is arguably 'too large' to be reliable. This usually happens when the number of observations per group is small and/or there is not much variation in outcomes.")
        
        
            # get overall att under calendar time effects
            # this is just average over all time periods
            calendar_att = np.mean(calendar_att_t)
            
            # get overall influence function
            calendar_inf_func = get_agg_inf_func(att=calendar_att_t,
                                                inffunc=calendar_inf_func_t,
                                                whichones=range(len(calendar_tlist)),
                                                weights_agg=np.repeat(1/len(calendar_tlist), len(calendar_tlist)),
                                                wif=None)[:, None]
            calendar_inf_func = np.array(calendar_inf_func)
            
            # get overall standard error
            calendar_se = get_se(calendar_inf_func, dp)
            if not np.isnan(calendar_se):
                if calendar_se <= np.sqrt(np.finfo(float).eps) * 10:
                    calendar_se = np.nan
        
            return {
                    "agg_method": "calendar",
                    "att": calendar_att,
                    "se": calendar_se,
                    "egt": list(map(t2orig, calendar_tlist)),
                    "att_egt": calendar_att_t,
                    "se_egt": calendar_se_t,
                    "critical_value": calendar_crit_val,
                    "influence_function": {'calendar_inf_func_t': calendar_inf_func_t, 
                                           'calendar_inf_func': calendar_inf_func},
                    "DIDparams": dp
                }
'''