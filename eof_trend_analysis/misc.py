import base_spatial; import importlib; importlib.reload(base_spatial)
from base_spatial import *

# Plot d and it's error using DFA2 and DFA3

def dfa_hist(temp, scale_lim=[5,10], s=[]):

    if s: plt.rcParams.update(s) #pylab.rcParams.update(s)

    sns.set_style("white")

    f, ax1 = plt.subplots(1, 1, figsize=(5, 3.5), sharey=False)  
    
    scales, fluct, coeff, error, kernel, positions, weights, slope = dfa_bradley(temp, 2, scale_lim=scale_lim, P=2, plot=True)

    ax1.plot(positions, kernel(positions), color='darkblue')
    n, bins, patches = ax1.hist(slope, weights=weights, color='royalblue', alpha=0.5, bins=55, density=True) 
    mean = positions[np.argmax(kernel(positions))]
    ax1.set_xlabel(r'$d$')    # fontsize=18
    ax1.set_ylabel(r'$p(d)$') # fontsize=18
    ax1.set_xlim([0.55,1])
    ax1.set_ylim([0,20])
    ax1.set_yticks([0,20])

    lower_x, upper_x = half_max_x(positions,kernel(positions))

    upp = upper_x-mean 
    low = mean-lower_x 

    ax1.axvline(x = mean, color='darkblue', label='$d_{DFA2}$')  # 'darkmagenta'
    ax1.axvline(upper_x, color='darkblue', ls='--', label='$\Delta d$')
    ax1.axvline(lower_x, color='darkblue', ls='--')

    print("mean = ", mean)
    print("up_x - mean = ", upp)
    print("mean - low_x = ", low)

    fwhm = positions[np.where(np.logical_and(positions <= upper_x, positions >= lower_x))]
    fwhm_up = positions[np.where(np.logical_and(positions <= upper_x, positions >= mean))]
    fwhm_low = positions[np.where(np.logical_and(positions <= mean, positions >= lower_x))]

    print("Area within FWHM = ", np.sum(kernel(fwhm)) / np.sum(kernel(positions)))
    print("Upper Area within FWHM_up = ", np.sum(kernel(fwhm_up)) / np.sum(kernel(positions[np.argmax(kernel(positions)):])))
    print("Lower Area within FWHM_low = ", np.sum(kernel(fwhm_low)) / np.sum(kernel(positions[:np.argmax(kernel(positions))])))

    scales, fluct, coeff, error, kernel, pos3, weights, slope = dfa_bradley(temp, 3, scale_lim=scale_lim, P=2, plot=True)
    lower_x, upper_x = half_max_x(pos3,kernel(pos3))

    mean3 = pos3[np.argmax(kernel(pos3))]
    ax1.axvline(x = mean3, color='limegreen', label='$d_{DFA3}$')

    ax1.plot(positions, kernel(positions), color='limegreen')
    n, bins, patches = ax1.hist(slope, weights=weights, color='limegreen', alpha=0.2, bins=55, density=True)

    ax1.axvline(upper_x, color='limegreen', ls='--') 
    ax1.axvline(lower_x, color='limegreen', ls='--')

    ax1.legend(framealpha=0.9) # fontsize=20

    upp, low = upper_x-mean3, mean3-lower_x 
    print("dfa3: mean = ", mean3)
    print("dfa3: up_x - mean = ", upp)
    print("dfa3: mean - low_x = ", low)

    ax1.set_xticks([0.6, 0.8, 1])
    ax1.set_xticklabels([0.1, 0.3, 0.5])
    
    return(f, ax1)


# Show that time series has a Gaussian distribution

def distr(x):
    
    rc('xtick', labelsize=20); rc('ytick', labelsize=20)
    rc('font', size=20); rc('axes', labelsize=20)
    sns.set_style("white")

    f, ax = plt.subplots(figsize=(5.5, 3.7))
    
    n, bins, patches = ax.hist(x, density=True, bins=30, color='b', alpha=0.5, label='Data')      
    ax.set_ylabel('Frequency')
    ax.set_xlabel(r'$PC_{1,t}$'); 
    ax.set_xlim([-12,12])

    (mu, sigma) = norm.fit(x)
    y = norm.pdf(bins, mu, sigma)
    ax.plot(bins, y, 'b-', linewidth=2, alpha=0.7, label='Fit')
    ax.legend(loc='upper left', fontsize=18)
    f.tight_layout()
    
    return(f, ax)


# Plot different parameters of grid data

def coord(dat, nam, coor, step, plim, trlim, dlim, con, a=1, two_tail=False):
    
    if two_tail == True:
        dat['p'] = dat['p'].apply(lambda x: 2*x)
        dat['p_ar'] = dat['p_ar'].apply(lambda x: 2*x)

    tr = [] 
    p  = [] 
    p_ar  = [] 
    d  = []

    lat1, lat2, lon1, lon2 = coor[2], coor[0], coor[1], coor[3]
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2)
    
    lats = np.arange(lat1, lat2+0.01,step)
    lons = np.arange(lon1, lon2+0.01,step)

    for j in range(len(lats)): 
        tr_lat = [] 
        p_lat = [] 
        par_lat = [] 
        d_lat  = []
    
        for i in range(len(lons)):
    
            lon, lat = lons[i], lats[j]  
    
            data = dat[dat['lon']==lon] 
            data = data[data['lat']==lat] 
        
            tr_lat.append(data['trend'].mean()) 
            p_lat.append(data['p'].mean())      
            par_lat.append(data['p_ar'].mean()) 
            d_lat.append(data['d'].mean())      
        
        tr.append(tr_lat)
        p.append(p_lat)
        p_ar.append(par_lat)
        d.append(d_lat)
    
    f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 4), sharey=False)

    levels = np.linspace(plim[0], plim[1], con)
    im = ax.contourf(lons, lats, p, levels, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    set_ax(ax, lat1, lat2, lon1, lon2)
    ax.set_title(r'arfima(0,d,0)', y=1.03) 
    cb = f.colorbar(im, fraction=a*0.024, pad=0.05, orientation='vertical')
    cb.set_label('p value', fontsize=18)   
    cb.set_ticks([plim[0],0.05,plim[1]])
    f.savefig('ecmwf/' + nam + '/coord_p_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")

    f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 4), sharey=False)

    im = ax.contourf(lons, lats, p_ar, levels, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
    set_ax(ax, lat1, lat2, lon1, lon2)
    ax.set_title(r'arfima(1,d,0)', y=1.03)
    cb = f.colorbar(im, fraction=a*0.024, pad=0.05, orientation='vertical')
    cb.set_label('p value', fontsize=18)   
    cb.set_ticks([plim[0],0.05,plim[1]])
    f.savefig('ecmwf/' + nam + '/coord_par_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")

    f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 4), sharey=False)

    levels = np.linspace(trlim[0],trlim[1],con)
    im = ax.contourf(lons, lats, tr, levels, cmap=plt.cm.autumn, transform=ccrs.PlateCarree())
    set_ax(ax, lat1, lat2, lon1, lon2)
    cb = f.colorbar(im, fraction=a*0.024, pad=0.05, orientation='vertical')
    cb.set_label(r'$m$', fontsize=18)   
    cb.set_ticks(trlim)
    f.savefig('ecmwf/' + nam + '/coord_trend_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")

    f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(6, 4), sharey=False)

    levels = np.linspace(dlim[0],dlim[1],con)
    im = ax.contourf(lons, lats, d, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    set_ax(ax, lat1, lat2, lon1, lon2)
    cb = f.colorbar(im, fraction=a*0.024, pad=0.05, orientation='vertical')
    cb.set_label(r'$d$', fontsize=18)   
    cb.set_ticks(dlim) 
    f.savefig('ecmwf/' + nam + '/coord_d_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")
    
    da_area = area_grid(lats, lons) 
    total_area = np.sum(~np.isnan(tr)*da_area) 
    
    tr_weighted = (np.array(tr)*da_area) 
    av_tr = np.nansum(tr_weighted) / total_area 
    d_weighted = (np.array(d)*da_area)  
    av_d = np.nansum(d_weighted) / total_area
    p_weighted = (np.array(p)*da_area) 
    av_p = np.nansum(p_weighted) / total_area
    par_weighted = (np.array(p_ar)*da_area) 
    av_par = np.nansum(par_weighted) / total_area
    
    return(av_tr, av_d, av_p, av_par) 




# from ttest

def coord_grid(dat, nam, coor, step, plim, trlim, dlim, con, a=1, h=1, arp=False):

    tr = [] 
    p  = [] 
    p_ar  = [] 
    d  = []
    ar = []

    lat1, lat2, lon1, lon2 = coor[2], coor[0], coor[1], coor[3]
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2)
    
    lats = np.arange(lat1, lat2+0.01,step)
    lons = np.arange(lon1, lon2+0.01,step)

    for j in range(len(lats)): 
        tr_lat = [] 
        p_lat = [] 
        par_lat = [] 
        d_lat  = []
        ar_lat = []
    
        for i in range(len(lons)):
    
            lon, lat = lons[i], lats[j]  
    
            data = dat[dat['lon']==lon] 
            data = data[data['lat']==lat] 
        
            tr_lat.append(data['trend'].mean()) 
            d_lat.append(data['d'].mean()) 
            ar_lat.append(data['ar'].mean()) 
            
            p_lat.append(2*data['tp'].mean())      
            par_lat.append(2*data['tarp'].mean()) 
        
        tr.append(tr_lat)
        p.append(p_lat)
        p_ar.append(par_lat)
        d.append(d_lat)
        ar.append(ar_lat)

    s = 0.87
    fig=plt.figure(figsize=[s*14,s*8])
    gs = gridspec.GridSpec(6,6, figure=fig)

    plt.subplots_adjust(wspace=0.4, hspace=.4)

    ax11=fig.add_subplot(gs[0:3,0:3], projection=ccrs.PlateCarree()) 
    ax12=fig.add_subplot(gs[0:3,3:6], projection=ccrs.PlateCarree())
    ax21=fig.add_subplot(gs[3:6,0:3], projection=ccrs.PlateCarree()) 
    ax22=fig.add_subplot(gs[3:6,3:6], projection=ccrs.PlateCarree())

    levels = np.linspace(plim[0], plim[1], con)
    im = ax11.contourf(lons, lats, p, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), extend="both")
    im.cmap.set_under(color="darkblue"); im.cmap.set_over(color="darkred")
    set_ax(ax11, lat1, lat2, lon1, lon2, xax=False, yax=True)
    ax11.set_title(r'arfima(0,d,0)', y=1.03) 

    im = ax12.contourf(lons, lats, p_ar, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), extend="both")
    im.cmap.set_under(color="darkblue"); im.cmap.set_over(color="darkred")
    set_ax(ax12, lat1, lat2, lon1, lon2, xax=False, yax=False)
    ax12.set_title(r'arfima(1,d,0)', y=1.03)
    
    cbar_ax = fig.add_axes([h*0.89, 0.525, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=[plim[0],0.05,plim[1]], orientation='vertical') 
    cb.set_label('p value', fontsize=18) 

    levels = np.linspace(trlim[0],trlim[1],con)
    if arp == False: im = ax21.contourf(lons, lats, tr, levels, cmap=plt.cm.autumn, transform=ccrs.PlateCarree())
    if arp == True:  im = ax21.contourf(lons, lats, ar, levels, cmap=plt.cm.Oranges.reversed(), transform=ccrs.PlateCarree())  #plt.cm.PuOr.reversed()
    
    set_ax(ax21, lat1, lat2, lon1, lon2)
    
    cbar_ax = fig.add_axes([h*0.49, 0.13, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=trlim, orientation='vertical')  
    if arp == True: cb.set_label('ar', fontsize=18) 

    levels = np.linspace(dlim[0],dlim[1],con)
    im = ax22.contourf(lons, lats, d, levels, cmap=plt.cm.YlOrBr.reversed(), transform=ccrs.PlateCarree()) #plt.cm.coolwarm
    set_ax(ax22, lat1, lat2, lon1, lon2, xax=True, yax=False)
    
    cbar_ax = fig.add_axes([h*0.89, 0.13, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=dlim, orientation='vertical') 
    cb.set_label(r'$d$', fontsize=18)
    
    if arp == False: fig.savefig('ecmwf/' + nam + '/coord_2side_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")
    if arp == True: fig.savefig('ecmwf/' + nam + '/coord_2side_ar_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")

def pol_coord_grid(dat, nam, coor, step, plim, trlim, dlim, con, a=1, h=1, arp=False, cen=-90):

    tr = [] 
    p  = [] 
    p_ar  = [] 
    d  = []
    ar = []

    lat1, lat2, lon1, lon2 = coor[2], coor[0], coor[1], coor[3]
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2)
    
    lats = np.arange(lat1, lat2+0.01,step)
    lons = np.arange(lon1, lon2+0.01,step)

    for j in range(len(lats)): 
        tr_lat = [] 
        p_lat = [] 
        par_lat = [] 
        d_lat  = []
        ar_lat = []
    
        for i in range(len(lons)):
    
            lon, lat = lons[i], lats[j]  
    
            data = dat[dat['lon']==lon] 
            data = data[data['lat']==lat] 
        
            tr_lat.append(365.25*10*data['trend'].mean()) 
            d_lat.append(data['d'].mean()) 
            ar_lat.append(data['ar'].mean()) 
            
            p_lat.append(2*data['tp'].mean())      
            par_lat.append(2*data['tarp'].mean()) 
        
        tr.append(tr_lat) #tr.append(tr_lat)
        p.append(p_lat)
        p_ar.append(par_lat)
        d.append(d_lat)
        ar.append(ar_lat)

    s = 0.87
    fig=plt.figure(figsize=[s*14,s*8])
    gs = gridspec.GridSpec(6,6, figure=fig)

    plt.subplots_adjust(wspace=0.4, hspace=.4)
    
    ax11=fig.add_subplot(gs[0:3,0:3], projection=ccrs.AzimuthalEquidistant(central_latitude=cen)) 
    ax12=fig.add_subplot(gs[0:3,3:6], projection=ccrs.AzimuthalEquidistant(central_latitude=cen))
    ax21=fig.add_subplot(gs[3:6,0:3], projection=ccrs.AzimuthalEquidistant(central_latitude=cen)) 
    ax22=fig.add_subplot(gs[3:6,3:6], projection=ccrs.AzimuthalEquidistant(central_latitude=cen))

    levels = np.linspace(plim[0], plim[1], con)
    im = ax11.contourf(lons, lats, p, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), extend="both")
    im.cmap.set_under(color="darkblue"); im.cmap.set_over(color="darkred")
    set_pax(ax11, lat1, lat2, lon1, lon2)
    ax11.set_title(r'arfima(0,d,0)', y=1.03) 

    im = ax12.contourf(lons, lats, p_ar, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), extend="both")
    im.cmap.set_under(color="darkblue"); im.cmap.set_over(color="darkred")
    set_pax(ax12, lat1, lat2, lon1, lon2)
    ax12.set_title(r'arfima(1,d,0)', y=1.03)
    
    cbar_ax = fig.add_axes([h*0.89, 0.525, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=[plim[0],0.05,plim[1]], orientation='vertical') 
    cb.set_label('p value', fontsize=18) 

    levels = np.linspace(trlim[0],trlim[1],con)
    #im = ax21.contourf(lons, lats, tr, cmap=plt.cm.autumn.reversed(), transform=ccrs.PlateCarree())
    if arp == False: im = ax21.contourf(lons, lats, tr, levels, cmap=plt.cm.autumn.reversed(), transform=ccrs.PlateCarree())
    if arp == True:  im = ax21.contourf(lons, lats, ar, levels, cmap=plt.cm.Oranges.reversed(), transform=ccrs.PlateCarree())  #plt.cm.PuOr.reversed()
    set_pax(ax21, lat1, lat2, lon1, lon2)
    
    cbar_ax = fig.add_axes([h*0.49, 0.13, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=trlim, orientation='vertical')  
    if arp == True: cb.set_label('ar', fontsize=18) 

    levels = np.linspace(dlim[0],dlim[1],con)
    im = ax22.contourf(lons, lats, d, levels, cmap=plt.cm.YlOrBr.reversed(), transform=ccrs.PlateCarree()) #plt.cm.coolwarm
    set_pax(ax22, lat1, lat2, lon1, lon2)
    
    cbar_ax = fig.add_axes([h*0.89, 0.13, 0.010, a*0.355]) 
    fig.canvas.mpl_connect('resize_event', resize_colobar)
    cb = fig.colorbar(im, cax=cbar_ax, ticks=dlim, orientation='vertical') 
    cb.set_label(r'$d$', fontsize=18)
    
    if arp == False: fig.savefig('ecmwf/coord_2side_' + nam + '_' + co + '.png', format='png', dpi=100, bbox_inches="tight")
    if arp == True: fig.savefig('ecmwf/coord_2side_ar_' + nam + '_' + co + '.png', format='png', dpi=100, bbox_inches="tight")
        
def coord(dat, nam, coor, step, plim, trlim, dlim, con, a=1, h=1):

    tr = [] 
    p  = [] 
    p_ar  = [] 
    d  = []

    lat1, lat2, lon1, lon2 = coor[2], coor[0], coor[1], coor[3]
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2)
    
    lats = np.arange(lat1, lat2+0.01,step)
    lons = np.arange(lon1, lon2+0.01,step)

    for j in range(len(lats)): 
        tr_lat = [] 
        p_lat = [] 
        par_lat = [] 
        d_lat  = []
    
        for i in range(len(lons)):
    
            lon, lat = lons[i], lats[j]  
    
            data = dat[dat['lon']==lon] 
            data = data[data['lat']==lat] 
        
            tr_lat.append(data['trend'].mean()) 
            d_lat.append(data['d'].mean()) 
            
            p_lat.append(2*data['tp'].mean())      
            par_lat.append(2*data['tarp'].mean()) 
        
        tr.append(tr_lat)
        p.append(p_lat)
        p_ar.append(par_lat)
        d.append(d_lat)
        
    f, ax = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 6), sharey=False)
    ax=ax.flatten()  

    levels = np.linspace(trlim[0],trlim[1],con)
    im = ax[0].contourf(lons, lats, tr, levels, cmap=plt.cm.autumn, transform=ccrs.PlateCarree())
    set_ax(ax[0], lat1, lat2, lon1, lon2, xax=True, yax=True)
    
    cbar_ax = f.add_axes([1.02, 0.29, a*0.011, a*0.41]) 
    f.canvas.mpl_connect('resize_event', resize_colobar)
    cb = f.colorbar(im, cax=cbar_ax, ticks=trlim, orientation='vertical')  

    levels = np.linspace(dlim[0],dlim[1],con)
    im = ax[1].contourf(lons, lats, d, levels, cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree())
    set_ax(ax[1], lat1, lat2, lon1, lon2, xax=True, yax=False)
    
    cbar_ax = f.add_axes([1.12, 0.29, a*0.011, a*0.41]) 
    f.canvas.mpl_connect('resize_event', resize_colobar)
    cb = f.colorbar(im, cax=cbar_ax, ticks=dlim, orientation='vertical') 
    cb.set_label(r'$d$', fontsize=18)
    
    levels = np.linspace(plim[0], plim[1], con)
    im = ax[2].contourf(lons, lats, p_ar, levels, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree(), extend="both")
    im.cmap.set_under(color="k")
    im.cmap.set_over(color="k")
    set_ax(ax[2], lat1, lat2, lon1, lon2, xax=True, yax=False)
    
    cbar_ax = f.add_axes([0.92, 0.29, a*0.011, a*0.41]) 
    f.canvas.mpl_connect('resize_event', resize_colobar)
    cb = f.colorbar(im, cax=cbar_ax, ticks=[plim[0],0.05,plim[1]], orientation='vertical') 
    cb.set_label('p value', fontsize=18) 
    
    f.savefig('ecmwf/' + nam + '/coord_3_' + nam + '.png', format='png', dpi=120, bbox_inches="tight")   
    
def new_var(dat):
    
    dat['zstat'] = dat['zstat'].apply(lambda x: x/np.sqrt(1000) )
    dat['tstat'] = dat['tstat'].apply(lambda x: x/np.sqrt(1000) )
    dat['zarstat'] = dat['zarstat'].apply(lambda x: x/np.sqrt(1000) )
    dat['tarstat'] = dat['tarstat'].apply(lambda x: x/np.sqrt(1000) )

    df = 1000 - 1
    dat['zp'] = dat['zstat'].apply(lambda x: stats.norm.cdf(x) )
    dat['tp'] = dat['tstat'].apply(lambda x: 1-stats.t.cdf(x, df=df) )
    dat['zarp'] = dat['zarstat'].apply(lambda x: stats.norm.cdf(x) )
    dat['tarp'] = dat['tarstat'].apply(lambda x: 1-stats.t.cdf(x, df=df) )
    
    return(dat)


# get parameters for grid

def big_temp(f, nam, coor, sep, order=2):
    
    T = xr.open_dataset(f) 
    T_detr = T.groupby("time.dayofyear") - T.groupby("time.dayofyear").mean('time')
    
    dT, derrT, dlerrT, arT, varT, betT   = [], [], [], [], [], []
    lonT, latT, tpl, tarpl = [], [], [], []
    d3T, dupT, dlowT, p3T, pupT, plowT = [], [], [], [], [], []
    
    lon_l = np.arange(coor[1],coor[3]+0.01,sep)
    lat_l = np.arange(coor[2],coor[0]+0.01,sep)
    
    for i in range(len(lon_l)): 
        for j in range(len(lat_l)): 
    
            lon, lat = lon_l[i], lat_l[j]
            print("lon = ", lon, "lat = ", lat)
    
            T_zoom = T_detr.sel(latitude=slice(lat, lat), longitude=slice(lon, lon)) # T_zoom before T_detr ? 
            Td = T_zoom['t2m'].values.flatten()
        
            if np.isnan(Td).any(): continue
            if not(Td.tolist()): continue
                
            lonT.append(lon)
            latT.append(lat)

            '''quick method'''

            x = np.arange(len(Td))
            scales, fluct, coeff, d_err = dfa_bradley(Td, 2)  # This line is slow

            d = float(coeff-0.5) 
            dup = d + d_err[1]
            dlow = d - d_err[0]
    
            diff = grlet_diff(Td, 1095, d) 
            x1 = diff[:-1]; x2 = diff[1:]
            popt, pcov = scipy.optimize.curve_fit(line, x1, x2) 
    
            ar, a = [popt[1]], popt[1]
    
            var = np.var(Td); vpc = var
            popt, pcov = scipy.optimize.curve_fit(line, x, Td) 
            alph, r_trend = popt 

            tarp_up, vtr = ttest_an(a, dup, var, r_trend, N=len(Td)) # approx
            tarp_low, vtr = ttest_an(a, dlow, var, r_trend, N=len(Td)) # approx
            tarp_3, d3 = np.nan, np.nan
            tp_, vtr = ttest_an(0, d, var, r_trend, N=len(Td)) 
            tarp_, vtr = ttest_an(a, d, var, r_trend, N=len(Td)) 

            ''' old method
            d, a, r_trend, vpc, tp_, vtr, d_err   = p_val(Td, ar=0, an=True)
            d, a, r_trend, vpc, tarp_, vtr, d_err = p_val(Td, ar=1, an=True)  
            dup, a, r_trend, vpc, tarp_up, vtr, d_err   = p_val(Td, ar=1, d_b='upp', an=True)
            dlow, a, r_trend, vpc, tarp_low, vtr, d_err = p_val(Td, ar=1, d_b='low', an=True)
            d3, a, r_trend, vpc, tarp_3, vtr, d_err = p_val(Td, ar=1, order=3, an=True)
            '''
            
            d3T.append(d3); dupT.append(dup); dlowT.append(dlow)
            p3T.append(tarp_3); pupT.append(tarp_up); plowT.append(tarp_low)
            dT.append(d); derrT.append(d_err[1]); dlerrT.append(d_err[0]); arT.append(a); 
            betT.append(r_trend); varT.append(vpc); tpl.append(tp_); tarpl.append(tarp_)
            
    data = pd.DataFrame({'lon': lonT, 'lat': latT, 'd': dT, 'du_err': derrT, 'dl_err': dlerrT,
                         'ar': arT, 'var': varT, 'trend': betT, 'tp': tpl, 
                         'tarp': tarpl, 'd3': d3T, 'dup': dupT, 'dlow': dlowT, 
                         'par3': p3T, 'par_up': pupT, 'par_low': plowT}) 

    try:
        data.to_csv('ecmwf/' + nam + '/coor_bdfa_unsym_1side_' + nam + '_ord' + str(order) + '_res' + str(sep) + '.csv', sep='\t')
    except:
        pass
            
    return(data)


# fit LRC and SRC parameters of time series

def dfa_fits(ax, temp, order, ref=False, leg_fs=22):

    region, scales, fluct, mean, error, lhs_arr, rhs_arr, slope, error, weights, offset = dfa_bradley(temp, order, scale_lim=[5,10], P=2, plot='fits')

    ax.plot(np.log2(scales)[1:-1], np.log2(fluct)[1:-1], 'bo', markersize=8, mec = 'k', label='data')

    fit_sparsity = 20
    n_col = int(len(slope)/fit_sparsity)
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1.*i/n_col) for i in range(n_col)])

    for i in range(0,len(slope),fit_sparsity):
    
        x = region[:,0][lhs_arr[i]:rhs_arr[i]]
        a, b = offset[i], slope[i]
        y = a + b*x
    
        ax.plot(x, y, linewidth=2) 
   
    i = len(slope)-1
    x = region[:,0][lhs_arr[i]:rhs_arr[i]]
    a, b = offset[i], slope[i]
    y = a + b*x
    ax.plot(x, y, linewidth=2, label='fit')

    if ref==True:
        x = region[:,0][2:]
        a, b = 0, 0.5
        y = a + b*x
        ax.plot(x, y, '--k', linewidth=1, alpha=0.5, label=r'$d = 0$')

    ax.set_xlabel(r'$\log_{10}(s)$')
    ax.set_ylabel(r'$\log_{10}F^2 (s)$') # r'$\log_{10}\langle F^2 (s) \rangle$'
    ax.legend(fontsize=leg_fs)

    d = mean - 0.5

    return(ax, d)


def ar_fits(ax, temp, d):

    #grlet_diff(data_, memory_, d_, delta_t_=1.0)
    #memory should not be 10! Memory should be M = 1095.
    diff = grlet_diff(temp, 1095, d) # temp, 10, d
    x1 = diff[:-1]
    x2 = diff[1:]
    
    ax.plot(x1, x2, '.', markersize='1') 
    
    ax.set_xlabel(r'$x_t$')
    ax.set_ylabel(r'$x_{t+1}$')
    
    popt, pcov = scipy.optimize.curve_fit(line, x1, x2) 
    ax.plot(x1, line(x1, *popt), 'orange', label=r'$\phi$=%3.3f' %popt[1]) 

    ax.legend() 

    return(ax, popt[1])


def trend_fits(ax, temp, show_var=True):

    x = np.arange(len(temp))
    X_l = sm.add_constant(x)
    #X_e = np.column_stack((np.exp(0.0003*x), np.ones(len(temp)))) 
 
    res = sm.OLS(temp, X_l).fit()
    #res_e = sm.OLS(temp, X_e).fit()

    pred_ols = res.get_prediction()
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]

    ax.set_xlabel('Year')
    ax.set_xticks([0, 30*365.25, 60*365.25]) #ax.set_xticks([7*365.25, 57*365.25, 107*365.25])
    ax.set_xticklabels([1950, 1980, 2010])           #ax.set_xticklabels([1900, 1950, 2000])
    ax.set_ylabel(r'$T~(^{\circ} C)$')

    ax.set_yticks([-10, 0, 10])

    ax.plot(x, temp, "-", linewidth=0.2, label="data") 
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    ax.plot(x, res.fittedvalues, "-", color='tab:red', linewidth=3, label="OLS")
    ax.legend(loc="lower right", ncol=2)
    #ax.set_ylim([-15, 15]) ###

    #ax.set_yticks([-10, 0, 10, 20])
    #ax.set_xlim([0, 30*365.25])

    if show_var==True:
        ax.plot(x, iv_u, "r--")
        ax.plot(x, iv_l, "r--")

    m = res.params[1]

    return(ax, m, res.summary())

def vfrac(T, coor, var_size, ret='pc1', sys='clim', fr=0.99):
    
    lat1, lat2, lon1, lon2 = int(coor[0]), int(coor[2]), int(coor[1]), int(coor[3])

    if sys=='clim': 
        T_zoom = T.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
        T_detr = T_zoom.groupby("time.dayofyear") - T_zoom.groupby("time.dayofyear").mean('time')
        temp_detr = T_detr['t2m'].values  
        lats = T_detr['latitude'].values 

        wgts = sqrt(abs(cos(deg2rad(lats))))*earth_radius(lats) 
        wgts *= len(wgts)/(sum(wgts))                                            
        weights_array = wgts[:, np.newaxis]                 
        solver = Eof(temp_detr, weights=weights_array) 
     
    elif sys=='spiral' or sys=='standard':  
        T = T.astype('float64')
        T = T - T.mean('t')

        T_zoom = T.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)) 
        temp_detr = T_zoom['u'].values[10:] 
        solver = Eof(temp_detr) 
    
    clear_output(wait=True) 
    varfrac = solver.varianceFraction() 
 
    '''Normalize the pc's and append pc1 variance 
       in this section.''' 

    if ret == 'pc1_var':  
        pc  = solver.pcs(npcs=1)                  
        eof = solver.eofs(neofs=1)    
       
        c = eof[0].flatten() 
        c = np.mean(c[~np.isnan(c)])
        pc1 = c*np.array(pc[:,0])

        var_size.append(np.var(pc1))
        print(np.var(pc1))

    if ret == 'pc1':
        var_size.append(varfrac[0])
        print(varfrac[0])
    
    elif ret == 'Dkld':
        frac = 0
        for i in range(3000): # 300
            frac += varfrac[i]
            if frac >= fr:
                var_size.append(i)
                print(i)
                break
                
    return(var_size)


def plot_pc_var(vfr, pcl, s=20, sv=None):
    params = {'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': s-1.5,
              'axes.labelsize': s, 'font.size': s, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params)
    sns.set_palette("pastel") # deep

    pcv = [np.var(x) for x in pcl] 
    x = [1,2,3,4,5,6,7,8]

    f, ax = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

    norm = np.sum(vfr[:8])/np.sum(pcv[:8])
    ax.plot(x, vfr[:8], 'k--', marker='h', markersize=13.4, mec = 'k', mfc = 'g', label=r'$\lambda_k$')
    ax.plot(x, np.array(pcv[:8])*norm, 'k-', marker='o', markersize=12, mec = 'k', mfc = 'b', label=r'$\lambda_k c_{k}^2$')

    ax.set_ylabel('PC var')
    ax.set_xlabel(r'PC$_k$')
    ax.legend(loc='upper right', fontsize=16, borderaxespad=0.54, framealpha=0.95)
    ax.grid(alpha=0.3)

    ins = ax.inset_axes([0.355,0.35,0.63,0.63])
    ins.plot(x, vfr[:8], 'k--', marker='h', markersize=10, mec = 'k', mfc = 'g', label=r' $\lambda$')
    ins.plot(x, pcv[:8], 'k-', marker='o', markersize=9, mec = 'k', mfc = 'b', label=' var.')
    ins.set_yscale('log')
    ins.set_xticks([])

    if sv: f.savefig('ecmwf/pc_var_'+sv+'.png', format='png', dpi=120, bbox_inches="tight")