import base_time_series; import importlib; importlib.reload(base_time_series)
from base_time_series import *

# Plot coastlines and borders on maps
def set_pax(ax, lat1=0, lat2=0, lon1=0, lon2=0):
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)   

# Add axes and borders to plots
def set_ax(ax, lat1, lat2, lon1, lon2, xax=True, yax=True, 
           coast=True, border=True):
    
    if coast: ax.coastlines()
    if border: ax.add_feature(cfeature.BORDERS) 

    x_step, y_step = int(abs(lon2-lon1)/4), int(abs(lat2-lat1)/4)
    
    if xax==True:
        #ax.set_xticks(np.arange(lon1, lon2+0.01,x_step), crs=ccrs.PlateCarree())
        #ax.set_xticks(np.arange(lon1, lon2+0.01,5), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(lon1, lon2+0.01,x_step), crs=ccrs.PlateCarree())
        lon_formatter = cticker.LongitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.set_xlabel(r'Longitude', rotation= 0)
        
    if yax==True:
        ax.set_yticks(np.arange(lat1, lat2+0.01,y_step), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)    
        ax.set_ylabel(r'Latitude')
    
def max_(eof):
    return(max([abs(np.nanmin(eof)), np.nanmax(eof)]))

def r_(x):
    return(max(round(max_(x)%1,2),floor(max_(x))))

def resize_colobar(event):
    plt.draw()

    posn = ax.get_position()
    cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0,
                          0.04, posn.height])  


# Calculate the Earth's radius
def earth_radius(lat):

    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)

    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r

# Calculate the area of grids (e.g. a 1º by 1º grid) for different longitude and latitude
def area_grid(lat, lon, wgt=True):

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    if wgt==True: 
        area = dy * dx
    if wgt==False:
        area = dlat * dlon 
    
    return area 

# Weights of grid sizes of Earth (considering spherical shape)
def weights(lons, lats, T):
    
    nanlist = []
    for i in range(len(lats)): 
        n, notnan = 0, 0
        for j in range(len(lons)): 
            lon, lat = lons[j], lats[i] 
            T_zoom = T.sel(latitude=slice(lat, lat), longitude=slice(lon, lon))  
            Tf = T_zoom['t2m'].values.flatten()
            n += 1
            if np.isnan(Tf).any(): continue
            if not (Tf.tolist()):  continue
            notnan += 1
        
        nanlist.append(notnan/n)
    
    wgts = abs(cos(deg2rad(lats)))*earth_radius(lats)
    
    wg = np.array(nanlist)*wgts
    wgts /= sum(wg)/sum(nanlist) 
    
    return(wgts) 

# Calculate the average signal of all the different grids    
def av_signal(T_, coor, step, per=[], season=None, weight=True, num=False, hem=None):
    
    if type(T_)==xr.core.dataset.Dataset or type(T_)==xr.core.dataarray.DataArray: T = T_
    elif type(T_)==str: T = xr.open_dataset(T_)
    if per: T = T.sel(time=slice(per[0], per[1]))

    if season:
        if season=='spring': season=[3,4,5]
        if season=='summer': season=[6,7,8]
        if season=='autumn' or season=='fall': season=[9,10,11]
        if season=='winter': season=[1,2,12]

        if coor[0]<0 and coor[2]<0: 
            print("Southern Hemisphere")
            if season=='spring': season=[9,10,11]
            if season=='summer': season=[1,2,12]
            if season=='autumn' or season=='fall': season=[3,4,5]
            if season=='winter': season=[6,7,8]

        T = T.sel(time=T.time.dt.month.isin(season))

    T_detr = T.groupby("time.dayofyear") - T.groupby("time.dayofyear").mean('time')
    
    Td = np.zeros(len(T_detr['time']))    
    total_area = 0

    lat1, lat2, lon1, lon2 = coor[2], coor[0], coor[1], coor[3]
    lats = np.arange(lat1, lat2+0.01,step)
    lons = np.arange(lon1, lon2+0.01,step)

    da_area = area_grid(lats, lons, wgt=weight) 

    if (lon1<0 and lon2 >0):
        lons1 = np.arange(0, lon2+0.01,step)
        lons2 = np.arange(360+lon1,360+0.01,step)
        lons = np.concatenate((lons1, lons2), axis=None)
    
    #da_area = area_grid(lats, lons, wgt=weight) 

    for i in range(len(lons)): 
        for j in range(len(lats)): 
        
            lon, lat = lons[i], lats[j] 
    
            T_zoom = T_detr.sel(latitude=slice(lat, lat), longitude=slice(lon, lon))  
            Tf = T_zoom['t2m'].values.flatten()
            
            if np.isnan(Tf).any(): continue
            if not (Tf.tolist()):  continue
                
            # for negative longitude the code needs some care   
            Td += np.array(Tf)*da_area[j, i]
            total_area += da_area[j, i] 
        
    Td /= total_area 

    if num:                                  
        _, _, _, var, tp, _, _    = p_val(Td, ar=0, an=False)
        d, a, tr, var, tpar, _, _ = p_val(Td, ar=1, an=False)
    else:
        tp, tpar = np.nan, np.nan

        _, _, _, _, tp_an, vtr_an, _        = p_val(Td, ar=0, an=True)
        d, a, tr, var, tpar_an, vtrar_an, _ = p_val(Td, ar=1, an=True)

    l = pd.DataFrame({'m': tr, 'd': d, 'a': a, 'var': var, 
                      'tp_mc': 2*tp, 'tpar_mc': 2*tpar, 
                      'tp': 2*tp_an, 'tpar': 2*tpar_an,
                      'vtr': vtr_an, 'vtrar': vtrar_an}, index=[0])
        
    return(Td, l)

def smod(x):
    if x > 180: x -= 360
    return x

# EOF analysis (and preparing the data by selecting seasons, time range, weighting, deseasoning etc.)
def eof_run(T_, npcs=15, coor=None, t_slice=[], w=True, season=None, sv=None, step=1, ant=False): 
    
    if type(T_)==xr.core.dataset.Dataset or type(T_)==xr.core.dataarray.DataArray: T = T_
    elif type(T_)==str: T = xr.open_dataset(T_)
   
    if type(coor)==xr.core.dataarray.DataArray:
        mask = coor; print("Using a mask")
        mask = mask.rename({'lon': 'longitude','lat': 'latitude'})

        lon = np.arange(-180, 180, step); lat = np.arange(-90, 90, step)
        id_lon = lon[np.where(~np.all(np.isnan(mask), axis=0))]
        id_lat = lat[np.where(~np.all(np.isnan(mask), axis=1))]

        lat1, lat2, lon1, lon2 = id_lat[-1], id_lat[0], id_lon[0], id_lon[-1]
        T = T.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
        T = T.where(mask.notnull())

        if sv: T.to_netcdf("ecmwf/"+sv+"_day_ecmwf.nc")

    elif type(coor)==list:
        lat1, lat2, lon1, lon2 = int(coor[0]), int(coor[2]), int(coor[1]), int(coor[3])
        
        if (lon1 >= 0 and lon2 >= 0) or (lon1 <= 0 and lon2 <= 0) or (ant==True): #or (lon1==-180 and lon2==180):
            T = T.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
        else:
            print("complicated case due to longitudinal sign change")
            T1 = T.sel(latitude=slice(lat1, lat2), longitude=slice(0,lon2))
            T2 = T.sel(latitude=slice(lat1, lat2), longitude=slice(360+lon1, 360))
            T = T1.combine_first(T2)

    if t_slice:  
        T = T.sel(time=slice(t_slice[0], t_slice[1])) 

    if season:
        if season=='spring': season=[3,4,5]
        if season=='summer': season=[6,7,8]
        if season=='autumn' or season=='fall': season=[9,10,11]
        if season=='winter': season=[1,2,12]

        if coor[0]<0 and coor[2]<0: 
            #print("Southern Hemisphere")
            if season=='spring': season=[9,10,11]
            if season=='summer': season=[1,2,12]
            if season=='autumn' or season=='fall': season=[3,4,5]
            if season=='winter': season=[6,7,8]

        T = T.sel(time=T.time.dt.month.isin(season))

        # change season for southern hemisphere
        
    T_detr = T.groupby("time.dayofyear") - T.groupby("time.dayofyear").mean('time')
    temp_detr = T_detr['t2m'].values
    lons = T_detr['longitude'].values
    lats = T_detr['latitude'].values

    #if lon1 < 0: lons = np.array([smod(x) for x in lons])
    
    if w==True:
        wgts = weights(lons, lats, T_detr)                     
        weights_array = wgts[:, np.newaxis]                
        solver = Eof(temp_detr, weights=weights_array) 
    if w==False:
        weights_array = 1                
        solver = Eof(temp_detr) 

    pc  = solver.pcs(npcs=npcs)                  
    eof = solver.eofs(neofs=npcs)
    varfrac = solver.varianceFraction()
    lambdas = solver.eigenvalues()
    
    return(weights_array, lons, lats, pc, eof, varfrac, lat1, lat2, lon1, lon2)

# Plot the first three PCs and EOFs
def eof_plot(T_, nam, coor, a=1, b=1.01, sh='23', an=True, per=[], pcn=0, 
             proj='cartesian', cen=-90, shift=False, season=None, st='50', step=1):
   
    weights_array, lons, lats, pc, eof, varfrac, lat1, lat2, lon1, lon2 = eof_run(T_, coor=coor, t_slice=per, season=season, step=step)
        
    f, ax = plt.subplots(1, 3, figsize=(12, 2), sharey=False)
        
    tp, tarp, pc_, eof_ = [], [], [], []

    for i in range(3):
        k = i + pcn
        
        c = eof[k].flatten() 
        c = np.mean(c[~np.isnan(c)]) 
        pci = c*np.array(pc[:,k]) 

        # the pc's should be ordered here
        # pc_ = order_pcs(pc_, varfrac, eof, seq=seq)
            
        ax[i].plot(pci, color='b', linewidth=0.5)  
        ax[i].axhline(0, color='k', ls='--', alpha=0.5)
        frac = str(np.array(varfrac[k]*100).round(2))
        ax[i].set_title("PC "+str(k+1)+" ("+frac+"%)", fontsize=18)
        
        if st=='50':
            year_ = np.array([0, 20*365.25, 40*365.25, 60*365.25])
            years = [1950, 1970, 1990, 2010]
        elif st=='79':
            year_ = np.array([1, 21*365.25, 41*365.25])
            years = [1980, 2000, 2020]
        if season: year_ /= 4

        ax[i].set_xlabel('Year')
        ax[i].set_xticks(year_)
        ax[i].set_xticklabels(years)
            
        yabs_max = abs(max(ax[i].get_ylim(), key=abs))
        ax[i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
        
        d, a, r_trend, vpc, tp_, vtr, d_err   = p_val(pci, ar=0, an=an)
        d, a, r_trend, vpc, tarp_, vtr, d_err = p_val(pci, ar=1, an=an)
            
        tp.append(tp_); tarp.append(tarp_); pc_.append(pci)

    #f.tight_layout() 
    f.savefig('/Users/tphillips/Atmospheric time series/ecmwf/pc_norm_' + nam + '_pc_' + str(pcn) + '_' + st + '.png', format='png', dpi=120, bbox_inches="tight")
    
    if proj=='polar': proj_ = ccrs.NorthPolarStereo() # ccrs.AzimuthalEquidistant(central_latitude=cen) # ccrs.NorthPolarStereo()
    if proj=='cartesian': proj_ = ccrs.PlateCarree()
    f, ax = plt.subplots(1, 3, subplot_kw={'projection': proj_}, figsize=(12, 6), sharey=False)

    kwtrans = dict(central_latitude=90, central_longitude=0.)
    trans = ccrs.Stereographic(**kwtrans)

    ax=ax.flatten()  

    eofmx = 0
    
    for i in range(3):  
        c = eof[i+pcn].flatten()  
        c = np.mean(c[~np.isnan(c)]) 
        eofi = (1/c)*eof[i+pcn].squeeze() 
        eofi /= weights_array 
        
        if max_(eofi) > max_(eofmx):
            eofmx = eofi
            
        if sh=='23':
            if i == 0: 
                eofsg = eofi 
                clevs = np.linspace(b*(-max_(eofmx)), b*max_(eofmx), 40)
                fill_1 = ax[i].contourf(lons, lats, eofi, clevs,                          
                        cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())   
            if i == 1: 
                eof2 = eofi
            if i == 2:
                clevs = np.linspace(1.01*(-max_(eofmx)), 1.01*max_(eofmx), 40)
                
                fill = ax[1].contourf(lons, lats, eof2, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
                fill = ax[i].contourf(lons, lats, eofi, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) 
                
        if sh=='13':
            if i == 0: 
                eof1 = eofi  
            if i == 1: 
                eofsg = eofi; eofmx = eof1
                
                clevs = np.linspace(1.01*(-max_(eofsg)), 1.01*max_(eofsg), 40) 
                fill_1 = ax[1].contourf(lons, lats, eofi, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())  
                
            if i == 2:
                clevs = np.linspace(b*(-max_(eofmx)), b*max_(eofmx), 40)
                
                fill = ax[0].contourf(lons, lats, eof1, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())  
                fill = ax[i].contourf(lons, lats, eofi, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) 
        if sh=='12':
            if i == 0: 
                eof1 = eofi 
            if i == 1: 
                eofm = eofmx
                clevs = np.linspace(1.01*(-max_(eofmx)), 1.01*max_(eofmx), 40)
                fill = ax[0].contourf(lons, lats, eof1, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
                fill = ax[i].contourf(lons, lats, eofi, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) 
                
            if i == 2:
                eofsg = eofi; eofmx = eofm
                clevs = np.linspace(b*(-max_(eofsg)), b*max_(eofsg), 40)
                fill_1 = ax[i].contourf(lons, lats, eofi, clevs,                          
                    cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) # ccrs.PlateCarree()
        if sh=='ind':
            #fill = ax[i].contourf(lons, lats, eofi, clevs=40,                         
            #             cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())   

            clevs = np.linspace(b*(-max_(eofi)), b*max_(eofi), 30) 

            if i==0: 
                eofm1 = eofi
                fill_1 = ax[i].contourf(lons, lats, eofi, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) 
            if i==1: 
                eofm2 = eofi
                fill_2 = ax[i].contourf(lons, lats, eofi, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
            if i==2: 
                eofm3 = eofi
                fill_3 = ax[i].contourf(lons, lats, eofi, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())  
         
            
        eof_.append(eofi)

    for i in range(3):

        ax[i].coastlines()
        ax[i].add_feature(cfeature.BORDERS)
        ax[i].set_title('EOF '+ str(i+pcn+1), fontsize=18)

        if proj=='cartesian':
            ax[i].set_xlabel('Longitude')
            step = int(abs(lon2-lon1)/3)
            if shift==False:
                ax[i].set_xticks(np.arange(lon1,lon2,step), crs=ccrs.PlateCarree())
                lon_formatter = cticker.LongitudeFormatter()
                ax[i].xaxis.set_major_formatter(lon_formatter)
           
    if proj=='cartesian':
        step = int(abs(lat2-lat1)/3)
        ax[0].set_yticks(np.arange(lat2,lat1,step), crs=ccrs.PlateCarree())
        lat_formatter = cticker.LatitudeFormatter()
        ax[0].yaxis.set_major_formatter(lat_formatter)  
        
    if sh != 'ind':
        cbar_ax_1 = f.add_axes([1.02, 0.29, a*0.011, a*0.41]) 
        cbar_ax = f.add_axes([0.92, 0.29, a*0.011, a*0.41])                                    
        f.canvas.mpl_connect('resize_event', resize_colobar) 
        cb_1 = f.colorbar(fill_1, cax=cbar_ax_1, ticks=[-r_(eofsg), 0, r_(eofsg)], orientation='vertical') 
        cb = f.colorbar(fill, cax=cbar_ax, ticks=[-r_(eofmx), 0, r_(eofmx)], orientation='vertical')  
    else:
        for i in range(3):
            cbar_ax = f.add_axes([0.92 + i*0.1, 0.29, a*0.011, a*0.41]) 
            if i==0: f.colorbar(fill_1, cax=cbar_ax, ticks=[-r_(eofm1), 0, r_(eofm1)], orientation='vertical') 
            if i==1: f.colorbar(fill_2, cax=cbar_ax, ticks=[-r_(eofm2), 0, r_(eofm2)], orientation='vertical') 
            if i==2: f.colorbar(fill_3, cax=cbar_ax, ticks=[-r_(eofm3), 0, r_(eofm3)], orientation='vertical') 
        
    f.savefig('/Users/tphillips/Atmospheric time series/ecmwf/eof_norm_' + nam + '_pc_' + str(pcn) + '_' + st + '_' + sh + '.png', format='png', dpi=120, bbox_inches="tight")
    
    return(pc_, eof_, tp, tarp)

# Normalize EOF's with spatial average weight
def eof_rescale(f, nam, coor, clev, a=1, no=1, an=None, proj='cartesian', cen=-90, per=[], season=None):
    
    weights_array, lons, lats, pc, eof, varfrac = eof_run(f, coor=coor, t_slice=per, season=season)
    lat1, lat2, lon1, lon2 = int(coor[0]), int(coor[2]), int(coor[1]), int(coor[3])
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2)
   
    clevs = np.linspace(clev[0], clev[1], 20)
    f, ax = plt.subplots(1, 1, figsize=(4, 2), sharey=True)

    c = eof[no-1].flatten()                         
    c = np.mean(c[~np.isnan(c)])      # Weight              
    pc1 = c*np.array(pc[:,no-1])                      
        
    eof1 = (1/c)*eof[no-1].squeeze()                  
    eof1 /= weights_array                          
    
    ax.plot(pc1, color='b', linewidth=0.5) 
            
    ax.axhline(0, color='k', ls='--', alpha=0.5)
    frac = str(np.array(varfrac[no-1]*100).round(2))
    ax.set_title(frac+"%", fontsize=18)
       
    ax.set_xlabel('Year')
    ax.set_xticks([0, 20*365.25, 40*365.25, 60*365.25])
    ax.set_xticklabels([1950, 1970, 1990, 2010])
        
    ax.set_ylabel("PC 1 (°C)", fontsize=18)
    ax.set_yticks([-10,0,10])

    f.savefig('/Users/tphillips/Atmospheric time series/ecmwf/pc_norm_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")
    
    if proj=='polar':
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.AzimuthalEquidistant(central_latitude=cen)}, figsize=(4, 6), sharey=False) 
        fill = ax.contourf(lons, lats, eof1, clevs, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree()) 
        set_pax(ax, lat2, lat1, lon1, lon2)
    if proj=='cartesian':
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(4, 6), sharey=False) 
        fill = ax.contourf(lons, lats, eof1, clevs, cmap=plt.cm.RdBu_r, transform = ccrs.PlateCarree())  
        set_ax(ax, lat2, lat1, lon1, lon2) 

    ax.set_title('EOF '+str(no), fontsize=18)
    
    cbar_ax = f.add_axes([0.92, 0.29, a*0.011, a*0.41]) 
    f.canvas.mpl_connect('resize_event', resize_colobar)
    cb = f.colorbar(fill, cax=cbar_ax, ticks=clev, orientation='vertical') #, cax=cax
           
    f.savefig('/Users/tphillips/Atmospheric time series/ecmwf/eof_norm_full_' + nam + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")
        
    d, a, r_trend, vpc, tp_, vtr, d_err   = p_val(pc1, ar=0, an=an)
    d, a, r_trend, vpc, tarp_, vtr, d_err = p_val(pc1, ar=1, an=an)
    
    return(pc1, eof1, 2*tp_, 2*tarp_)

# Calculate and weights PCs
def eof_pcs(f, nam, coor, pcs=[1,2,3], npcs=40, per=[], weight=True, season=None, step=1):
    
    _, _, _, pc, eof, varfrac, _, _, _, _ = eof_run(f, npcs=npcs, coor=coor, t_slice=per, w=weight, season=season, step=step)
    
    pcl, cl, vl = [], [], []
    pcs = [x-1 for x in pcs]
    
    for i in pcs:  
        c = eof[i].flatten() 
        c = np.mean(c[~np.isnan(c)])
        
        pcl.append(c*np.array(pc[:,i]))
        cl.append(c)
        vl.append(varfrac[i])

    return(cl, vl, pcl)

# Calculate statistics for PCs
def eof_pc(f, coor, per=[], weight=True, season=None):
    
    _, _, _, pc, eof, _, _, _, _, _ = eof_run(f, npcs=2, coor=coor, t_slice=per, w=weight, season=season)
        
    c = eof[0].flatten() 
    c = np.mean(c[~np.isnan(c)])
    pc1 = c*np.array(pc[:,0])
        
    _, _, _, _, tp_an, vtr_an, _        = p_val(pc1, ar=0, an=True)
    d, a, tr, vpc, tpar_an, vtrar_an, _ = p_val(pc1, ar=1, an=True)
        
    l = [tr, d, a, vpc, 2*tp_an, 2*tpar_an, vtr_an, vtrar_an]
        
    return(pc1, l)

# Single EOF plot
def recons(f, nam, coor, n=3, proj='cartesian', cen=-90, per=[], season=None, ant=False, clev=None):
    #T = xr.open_dataset(f)
   
    lat1, lat2, lon1, lon2 = int(coor[0]), int(coor[2]), int(coor[1]), int(coor[3]) 
    co = str(lat1) + '_' + str(lon1) + '_' + str(lat2) + '_' + str(lon2) 
    weights_array, lons, lats, pc, eof, _, _, _, _, _ = eof_run(f, coor=coor, t_slice=per, season=season, ant=ant)

    for i in range(n): 
        pc1 = np.array(pc[:,i]) 

        x = np.arange(len(pc1)) 
        popt, pcov = scipy.optimize.curve_fit(line, x, pc1) 
        alph, r_trend = popt 
        r_trend *= 10*365.25 
        
        #c = eof[i].flatten()  
        #c = np.mean(c[~np.isnan(c)]) 
        #eof1 = (1/c)*eof[i].squeeze()    
        eof1 = eof[i].squeeze()                 
        eof1 /= weights_array 
    
        if i == 0: 
            eofn = r_trend*eof1
            #eofn = eof1
        else:
            eofn += r_trend*eof1
            #eofn += eof1

    if clev: 
        clevs = np.linspace(clev[0], clev[1], 15)
    else:
        clevs = 15
            
    if proj=='polar':
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.AzimuthalEquidistant(central_latitude=cen)}, figsize=(4, 6), sharey=False) 
        fill = ax.contourf(lons, lats, eofn, clevs, #levels=15, 
                            cmap=plt.cm.autumn.reversed(), transform=ccrs.PlateCarree())  # plt.cm.autumn, plt.cm.RdBu_r
        set_pax(ax, lat2, lat1, lon1, lon2)
    else:
        f, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(4, 6), sharey=False) 
        fill = ax.contourf(lons, lats, eofn, clevs, #levels=15, 
                            cmap=plt.cm.autumn.reversed(), transform=ccrs.PlateCarree())  # plt.cm.autumn, plt.cm.RdBu_r
        set_ax(ax, lat2, lat1, lon1, lon2)
    
    ax.set_title('Reconstructed Trend', fontsize=18)    
    #ax.set_title('Reconstructed EOF', fontsize=18) 
    
    cbar_ax = f.add_axes([1.05, 0.29, 0.025, 0.45]) 
    f.canvas.mpl_connect('resize_event', resize_colobar)   

    if clev:
        cb = f.colorbar(fill, cax=cbar_ax, ticks=[clev[0], clev[1]], orientation='vertical')
    else:
        cb = f.colorbar(fill, cax=cbar_ax, ticks=[np.nanmin(eofn), np.nanmax(eofn)], orientation='vertical')

    f.savefig('ecmwf/recon_eof_n' + str(n) + '_' + co + '.png', format='png', dpi=120, bbox_inches="tight")


# Adding PC's 
def add_pc_plot(pcs, av_sig, par='m', name=None, sv=None): 

    par_l = np.array(pcs[par]) 
    y = np.append(par_l, av_sig[par][0])      
    x = np.append(list(range(1,len(y))), 1000)

    sns.set_style("white") # , rc=params
    fig = plt.figure(figsize=(5, 3)) 
    bax = brokenaxes( 
        xlims=((1, len(y)), (999, 1001)),
        hspace=.15,
    )

    bax.plot(x, y, 'k-', marker='o', mec='k', mfc='b', label='$y=x=10^{0}$ to $10^{4}$')

    bax.axs[1].set_xticks([1000])
    bax.axs[1].set_xticklabels([r'$\infty$'])

    bax.grid(axis='both', which='major', ls='-', alpha=0.3)
    bax.grid(axis='both', which='minor', ls='--', alpha=0.3)

    bax.axs[0].set_xlabel('no PCs')
    bax.axs[0].xaxis.set_label_coords(0.65, -0.16)
    if name != None:
        bax.axs[0].set_ylabel(name) 
    else: 
        bax.axs[0].set_ylabel(par) 
    bax.axs[0].yaxis.set_label_coords(-0.25, 0.5) 
 
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    bax.axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-2,2)) 
 
    if sv: 
        fig.savefig('ecmwf/add_pc/add_pcs_' + par + '_' + sv + '.png', format='png', dpi=120, bbox_inches="tight")
 
    plt.show() 

# Adding PC's 
def add_pc_plot_ax(ax, pcs, av_sig, par='m', name=None, xax=True): 
                                                                                                            
    par_l = np.array(pcs[par]) 
    y = np.append(par_l, av_sig[par][0])      
    x = np.append(list(range(1,len(y))), 1000) 

    #sns.set_style("white") # , rc=params
    #fig = plt.figure(figsize=(5, 3)) 
    bax = brokenaxes( 
        xlims=((1, len(y)), (999, 1001)),
        subplot_spec=ax,
    )

    bax.plot(x, y, 'k-', marker='o', mec='k', mfc='b', label='$y=x=10^{0}$ to $10^{4}$')

    bax.axs[1].set_xticks([1000])
    bax.axs[1].set_xticklabels([r'$\infty$'])

    bax.grid(axis='both', which='major', ls='-', alpha=0.3)
    bax.grid(axis='both', which='minor', ls='--', alpha=0.3)

    if xax==True:
        bax.axs[0].set_xlabel('no PCs')
        bax.axs[0].xaxis.set_label_coords(0.65, -0.16)
    if name != None: bax.axs[0].set_ylabel(name)
    else: bax.axs[0].set_ylabel(par)
    bax.axs[0].yaxis.set_label_coords(-0.25, 0.5)
    bax.axs[0].ticklabel_format(style='sci', axis='y', scilimits=(-2,2))


def tr(y):
    x = np.arange(len(y))
    popt, pcov = scipy.optimize.curve_fit(line, x, y) 
    alph, r_trend = popt 
    return(np.abs(r_trend))

# Ordering the PC's in different ways
def order_pcs(pc_, eig, eofm, seq='eig'):
    if seq=='eig':
        index = list(range(len(pc_)))
    
    if seq=='norm': # |<EOF_k>|
        sl = [np.abs(eofm[i]) for i in range(len(eofm))]
        index = np.argsort(sl)[::-1]
        sl, pc_ = (list(t) for t in zip(*sorted(zip(sl, pc_), reverse=True)))
        #index = np.argsort(sl)[::-1]

    if seq=='norm eig': # mu_k |<EOF_k>|^2
        sl = [eig[i]*(eofm[i])**2 for i in range(len(eig))]
        index = np.argsort(sl)[::-1]
        sl, pc_ = (list(t) for t in zip(*sorted(zip(sl, pc_), reverse=True)))
        #index = np.argsort(sl)[::-1]

    if seq=='var':
        sorter = lambda x: np.var(x)
        pc_.sort(reverse=True, key=sorter)
        index = []

    if seq=='trend':
        pc_.sort(reverse=True, key=tr)
        index = []

    return(index, pc_)

# Calculating statistics for added PC's
def add_pc(pc_, eig, eofm, seq='eig', save=None):

    dl, al, ml, varl, tpl, tarpl, vtrl = [], [], [], [], [], [], []
    pc_add = np.zeros(len(pc_[0]))
    index, pc_ = order_pcs(pc_, eig, eofm, seq=seq)

    print("index:", index)

    for i in range(len(pc_)):
        pc_add += np.array(pc_[i]) 

        d, a, m, var, tp, vtr, d_err = p_val(pc_add, ar=0, an=True)
        d, a, m, var, tarp, vtr, d_err = p_val(pc_add, ar=1, an=True)
        dl.append(d); al.append(a); ml.append(m); tpl.append(tp)
        varl.append(var); tarpl.append(tarp); vtrl.append(vtr)
    
    data = pd.DataFrame({'d': dl, 'a': al, 'm': ml,
                     'var': varl, 'tpar': 2*np.array(tarpl), 'tp': 2*np.array(tpl),
                     'vtrar': vtrl, 'eig': eig, 'eofm': eofm})

    dat_pc = pd.DataFrame({'pc': pc_add})

    if save != None:
        data.to_csv('pc_add_data_'+save+'.csv', sep='\t')
        dat_pc.to_csv('pc_add_series_'+save+'.csv', sep='\t')
    
    return(data, pc_add)

# Print statistics for PCs
def print_4pc(pc_, index=[0,1,2,3,4,5]):
    #_, _, _, _, tp_1, _, _             = p_val(pc_[index[0]], ar=0, an=True)
    tp_1, tp_2, tp_3, tp_4, tp_12, tp_34  = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    d1, a1, r1, _, tp_ar1, vtr1, _     = p_val(pc_[index[0]], ar=1, an=True)
    #print("d =", round(d1, 3), " a =", round(a1, 3), " m =", r1, " p =", round(2*tp_1, 3), "p ar =", 2*tp_ar1)

    #_, _, _, _, tp_2, _, _             = p_val(pc_[index[1]], ar=0, an=True)
    d2, a2, r2, _, tp_ar2, vtr2, _     = p_val(pc_[index[1]], ar=1, an=True)
    #print("d =", round(d2, 3), " a =", round(a2, 3), " m =", r2, " p =", round(2*tp_2, 3), "p ar =", 2*tp_ar2)

    #_, _, _, _, tp_3, _, _             = p_val(pc_[index[2]], ar=0, an=True)
    d3, a3, r3, _, tp_ar3, vtr3, _     = p_val(pc_[index[2]], ar=1, an=True)
    #print("d =", round(d3, 3), " a =", round(a3, 3), " m =", r3, " p =", round(2*tp_3, 3), "p ar =", 2*tp_ar3)

    #_, _, _, _, tp_4, _, _             = p_val(pc_[index[3]], ar=0, an=True)
    d4, a4, r4, _, tp_ar4, vtr4, _     = p_val(pc_[index[3]], ar=1, an=True)
    #print("d =", round(d4, 3), " a =", round(a4, 3), " m =", r4, " p =", round(2*tp_4, 3), "p ar =", 2*tp_ar4)

    #_, _, _, _, tp_12, _, _             = p_val(pc_[index[0]]+pc_[index[1]], ar=0, an=True)
    d12, a12, r12, _, tp_ar12, vtr12, _ = p_val(pc_[index[0]]+pc_[index[1]], ar=1, an=True)
    #print('1+2', "d =", round(d12, 3), " a =", round(a12, 3), " m =", r12, " p =", round(2*tp_12, 3), "p ar =", 2*tp_ar12)

    #_, _, _, _, tp_34, _, _             = p_val(pc_[index[0]]+pc_[index[1]]+pc_[index[2]], ar=0, an=True)
    d34, a34, r34, _, tp_ar34, vtr34, _ = p_val(pc_[index[0]]+pc_[index[1]]+pc_[index[2]], ar=1, an=True)
    #print('1+2+3', "d =", round(d34, 3), " a =", round(a34, 3), " m =", r34, " p =", round(2*tp_34, 3), "p ar =", 2*tp_ar34)

    #_, _, _, _, tp_34, _, _            = p_val(pc_[index[2]]+pc_[index[3]], ar=0, an=True)
    #d34, a34, r34, _, tp_ar34, _, _ = p_val(pc_[index[2]]+pc_[index[3]], ar=1, an=True)
    #print('3+4', "d =", round(d34, 3), " a =", round(a34, 3), " m =", r34, " p =", round(tp_34, 3), "p ar =", tp_ar34)

    return([d1, d2, d3, d4, d12, d34], [a1, a2, a3, a4, a12, a34],
           [r1, r2, r3, r4, r12, r34], [2*tp_1, 2*tp_2, 2*tp_3, 2*tp_4, 2*tp_12, 2*tp_34],
           [2*tp_ar1, 2*tp_ar2, 2*tp_ar3, 2*tp_ar4, 2*tp_ar12, 2*tp_ar34],
           [vtr1, vtr2, vtr3, vtr4, vtr12, vtr34])

# Seasonal PCs
def pc_seas(f, nam, coor, pcl, per, seq='eig', season=None, 
            step=1, df=pd.DataFrame({}), npcs=51):

    if season: print(season)

    eof_, eig_, pc_ = eof_pcs(f, nam, coor, pcs=list(range(1,npcs)), npcs=npcs, 
                                 weight=True, per=per, season=season) 
    _, l_a = av_signal(f, coor, step, weight=True, per=per, season=season) 
    index, pc_ = order_pcs(pc_, eig_, eof_, seq=seq); print('index:', index[:6])

    dl, al, rl, tl, atl, vtrl = print_4pc(pc_, index=index)
    dl.append(l_a['d'][0]); al.append(l_a['a'][0]); rl.append(l_a['m'][0])
    tl.append(l_a['tp'][0]); atl.append(l_a['tpar'][0]); vtrl.append(l_a['vtrar'][0])

    a_ = 365.25*10
    if df.empty==True: 
        df = pd.DataFrame({'s':7*['whole'], 'l': pcl, 'd': dl, 'a': al, 'm': a_*np.array(rl), 'p': tl, 'par': atl, 'vtr': a_*a_*np.array(vtrl), 
                           'sdtr': a_*np.sqrt(np.array(vtrl))})
    else:
        df2 = pd.DataFrame({'s':7*[season], 'l': pcl, 'd': dl, 'a': al, 'm': a_*np.array(rl), 'p': tl, 'par': atl, 'vtr': a_*a_*np.array(vtrl),
                            'sdtr': a_*np.sqrt(np.array(vtrl))}) 
        df = df.append(df2)
    return(df)

# Create table of statistics for seasonal PCs
def print_pc_seas(f, nam, coor, seq='eig', step=1, 
                  npcs=51, per=['1979-01-01', '2022-12-31']): 
    pcl = ['1', '2', '3', '4', '12', '34', 'av']

    df = pc_seas(f, nam, coor, pcl, per, seq=seq, season=None, step=step, npcs=npcs)
    df = pc_seas(f, nam, coor, pcl, per, seq=seq, season='spring', step=step, df=df, npcs=npcs)
    df = pc_seas(f, nam, coor, pcl, per, seq=seq, season='summer', step=step, df=df, npcs=npcs)
    df = pc_seas(f, nam, coor, pcl, per, seq=seq, season='autumn', step=step, df=df, npcs=npcs)
    df = pc_seas(f, nam, coor, pcl, per, seq=seq, season='winter', step=step, df=df, npcs=npcs)

    return(df)
                      
# Bar plot of statistics
def bar_plot(df, s=21, sv=None, bar=False, ax1l=None, ax2l=None, lloc='upper right', 
             alpha=0.8, npc=2, dpi=120, size=(14, 5), scale='lin'):

    params = {'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': s-1.5,
              'axes.labelsize': s, 'font.size': s, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params) # "ticks"
    sns.mpl.rc("figure", figsize=(6, 6))

    species = ("All", "Spr.", "Sum.", "Aut.", "Win.")
    sns.set_palette("gist_earth")
    sns.set_palette("deep")

    penguin_means = {
        'PC1': df[df['l']=='1']['m2'].tolist(),
        'PC2': df[df['l']=='2']['m2'].tolist(),
        'PC1+PC2': df[df['l']=='12']['m2'].tolist(),
        'Av.': df[df['l']=='av']['m2'].tolist(),
    }

    x = np.arange(len(species)) 
    width = 0.18 
    multiplier = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=size) 

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, edgecolor='black', label=attribute)
        multiplier += 1

    penguin_err = {
        'PC1': [df[df['l']=='1']['m2'].tolist()[0],df[df['l']=='1']['sdtr'].tolist()[0]],
        'PC2': [df[df['l']=='2']['m2'].tolist()[0],df[df['l']=='2']['sdtr'].tolist()[0]],
        'PC1+PC2': [df[df['l']=='12']['m2'].tolist()[0], df[df['l']=='12']['sdtr'].tolist()[0]],
        'Av.': [df[df['l']=='av']['m2'].tolist()[0], df[df['l']=='av']['sdtr'].tolist()[0]],
    }
    multiplier = 0
    for attribute, measurement in penguin_err.items():
        offset = width * multiplier
        (_, caps, _) = ax1.errorbar(0 + offset, measurement[0], yerr=measurement[1], fmt='ro', markersize=6, mec ='k',
                               ecolor='k', elinewidth=2, capsize=3)
        for cap in caps: cap.set_markeredgewidth(1)
        multiplier += 1    

    ax1.set_ylabel(r'$m$')
    ax1.set_xticks(x + width, species)
    ax1.legend(loc=lloc, ncol=2, framealpha=alpha)
    ax1.grid(alpha=0.3)
    if ax1l: ax1.set_ylim(ax1l)

    if npc == 2:
        penguin_means = {
            'PC1': df[df['l']=='1']['par'].tolist(),
            'PC2': df[df['l']=='2']['par'].tolist(),
            'PC1+PC2': df[df['l']=='12']['par'].tolist(),
            'Av': df[df['l']=='av']['par'].tolist(),
        }
    else:
        penguin_means = {
            'PC1': [df[df['l']=='1']['par'].tolist(),
                    5*['b']],
            'Av': [df[df['l']=='av']['par'].tolist(),
                    5*['r']],
        }

    multiplier = 0
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        if npc == 2: rects = ax2.bar(x + offset + 0, measurement, width, edgecolor='black', label=attribute)
        else: rects = ax2.bar(x + offset + 0, measurement[0], width, color=measurement[1], edgecolor='black', label=attribute)
        multiplier += 1

    ax2.set_ylabel(r'$p$ value')
    ax2.set_xticks(x + width + 0.8, species)
    #ax2.legend(loc=lloc, ncol=2, framealpha=alpha)
    if scale=='log': ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    if ax2l: ax2.set_ylim([0, ax2l])

    fig.tight_layout()

    if sv:
        fig.savefig('/Users/tphillips/Atmospheric time series/ecmwf/pval_m_'+sv+'_compare_p1d0_2p.png', 
                    format='png', dpi=dpi, bbox_inches="tight")

    plt.show()

# Bar plot of p values
def p_bar_plot(df, s=21, sv=None, lloc='upper right'):

    params = {'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': s-1.5,
              'axes.labelsize': s, 'font.size': s, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params)
    sns.set_palette("deep")

    species = ("All", "Spr.", "Sum.", "Aut.", "Win.")

    x = np.arange(len(species)) 
    width = 0.18 
    multiplier = 0

    fig, ax = plt.subplots(figsize=(6, 5))

    penguin_means = {
        'PC1': df[df['l']=='1']['par'].tolist(),
        'Av': df[df['l']=='av']['par'].tolist(),
    }

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, edgecolor='black', label=attribute)
        multiplier += 1

    ax.set_ylabel('p value')
    ax.set_xticks(x + width, species)
    ax.legend(loc=lloc, framealpha=0.95)
    ax.grid(alpha=0.3)

    if sv:
        fig.savefig('ecmwf/pval_p1d0_' + sv + 'compare_2p.png', 
                    format='png', dpi=120, bbox_inches="tight")

    plt.show()

# Bar plot of trends
def m_bar_plot(df, s=21, sv=None, bar=False, up=0.35, lloc='upper right', q=False):

    params = {'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': s-1.5,
              'axes.labelsize': s, 'font.size': s, 'legend.handlelength': 2}

    sns.set_theme(style="white", rc=params)
    sns.set_palette("deep")

    species = ("All", "Spr.", "Sum.", "Aut.", "Win.")

    x = np.arange(len(species)) 
    width = 0.18 
    multiplier = 0

    fig, ax = plt.subplots(figsize=(6, 5))

    sd = 'sdtr'
    if q==True: sd = 'sdtr2'

    if bar==False:
        penguin_means = {
            'PC1': df[df['l']=='1']['m2'].tolist(),
            'Av': df[df['l']=='av']['m2'].tolist(),
        }

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            multiplier += 1

    else: 
        penguin_means = {
            'PC1': [df[df['l']=='1']['m2'].tolist(),
                    df[df['l']=='1'][sd].tolist()],
            'PC2': [df[df['l']=='2']['m2'].tolist(),
                    df[df['l']=='2'][sd].tolist()],
            'PC1+PC2': [df[df['l']=='12']['m2'].tolist(),
            df[df['l']=='12'][sd].tolist()],
            'Av':  [df[df['l']=='av']['m2'].tolist(),
                    df[df['l']=='av'][sd].tolist()],
        }

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement[0], width, edgecolor='black', label=attribute)
            (_, caps, _) = ax.errorbar(x + offset, measurement[0], yerr=measurement[1], fmt='ro', markersize=6, mec ='k',
            ecolor='k', elinewidth=2, capsize=3)
            for cap in caps: cap.set_markeredgewidth(1)
            multiplier += 1

    ax.set_ylabel('m')
    ax.set_xticks(x + width, species)
    ax.legend(loc=lloc, framealpha=0.95)
    ax.grid(alpha=0.3)
    ax.set_ylim([0,up])

    if sv:
        fig.savefig('ecmwf/m_p1d0_' + sv + 'compare_2p.png', 
                    format='png', dpi=120, bbox_inches="tight")

    plt.show()
