'''
@author: mjiang,jgirard
'''
from util import *
# from sparse import *
import sys

def test_ind(ind,N):
    res = ind
    if ind < 0 : 
        res = -ind
        if res >= N: 
            res = 2*N - 2 - ind
    if ind >= N : 
        res = 2*N - 2 - ind
        if res < 0:
            res = -ind
    return res
    

def b3splineTrans(im_in,step):
    (nx,ny) = np.shape(im_in)
    im_out = np.zeros((nx,ny))
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    
    buff = np.zeros((nx,ny))
    
    for i in np.arange(nx):
        for j in np.arange(ny):
            jl = test_ind(j-step,ny)
            jr = test_ind(j+step,ny)
            jl2 = test_ind(j-2*step,ny)
            jr2 = test_ind(j+2*step,ny)
            buff[i,j] = c3 * im_in[i,j] + c2 * (im_in[i,jl] + im_in[i,jr]) + c1 * (im_in[i,jl2] + im_in[i,jr2])
    
    for j in np.arange(ny):
        for i in np.arange(nx):
            il = test_ind(i-step,nx)
            ir = test_ind(i+step,nx)
            il2 = test_ind(i-2*step,nx)
            ir2 = test_ind(i+2*step,nx)
            im_out[i,j] = c3 * buff[i,j] + c2 * (buff[il,j] + buff[ir,j]) + c1 * (buff[il2,j] + buff[ir2,j])
    
    return im_out

def b3spline_fast(step_hole):
    c1 = 1./16
    c2 = 1./4
    c3 = 3./8
    length = np.int(4*step_hole+1) #JGMOD
    kernel1d = np.zeros((1,length))
    kernel1d[0,0] = c1
    kernel1d[0,-1] = c1
    kernel1d[0,step_hole] = c2
    kernel1d[0,-1-step_hole] = c2
    kernel1d[0,2*step_hole] = c3
    kernel2d = np.dot(kernel1d.T,kernel1d)
    return kernel2d

def star2d(im,scale,fast = True,gen2=True,normalization=False):
    (nx,ny) = np.shape(im)
    nz = scale
    # Normalized transfromation
    head = 'star2d_gen2' if gen2 else 'star2d_gen1'
    trans = 1 if gen2 else 2
    if normalization and (pm.trHead != head):
        pm.trHead = head
        pm.trTab = nsNorm(nx,ny,nz,trans)
    wt = np.zeros((nz,nx,ny))
    step_hole = 1
    im_in = np.copy(im)
    
    for i in np.arange(nz-1):
        if fast:
            kernel2d = b3spline_fast(step_hole)
            im_out = psg.convolve2d(im_in, kernel2d, boundary='symm',mode='same')
        else:
            im_out = b3splineTrans(im_in,step_hole)
            
        if gen2:
            if fast:
                im_aux = psg.convolve2d(im_out, kernel2d, boundary='symm',mode='same')
            else:
                im_aux = b3splineTrans(im_out,step_hole)
            wt[i,:,:] = im_in - im_aux
        else:        
            wt[i,:,:] = im_in - im_out
            
        if normalization:
            wt[i,:,:] /= pm.trTab[i]
        im_in = np.copy(im_out)
        step_hole *= 2
	step_hole=np.int(step_hole) # JG MOD
        
    wt[nz-1,:,:] = np.copy(im_out)
    if normalization:
        wt[nz-1,:,:] /= pm.trTab[nz-1]
    
    return wt


# def star2d_boost(im,mr2d,gen2=True,normalization=False):
#     (nx,ny) = np.shape(im)
#     nz = mr2d.nbr_scale()        
#     if gen2:
#         mr2d.ModifiedATWT = Bool.True
#     else:
#         mr2d.ModifiedATWT = Bool.False
#     # Normalized transfromation
#     head = 'star2d_gen2' if gen2 else 'star2d_gen1'
#     if normalization and (pm.trHead != head):
#         pm.trHead = head
#         pm.trTab = nsNorm_boost(mr2d,gen2)
#         
#     imArr = Iflt(nx,ny)
#     imArr.data[:] = im
#     mr2d.transform(imArr)
#         
#     wt = np.zeros((nz,nx,ny))
#     for i in np.arange(nz):
#         wt[i] = np.copy(mr2d.band(i).data)
#         if normalization:
#             wt[i] /= pm.trTab[i]
# #             tmpArr = Iflt(nx,ny)
# #             tmpArr.data[:] = wt[i]
#             mr2d.band(i).data[:] = wt[i]
# #             mr2d.insert_band(tmpArr,i)
# #             tmpArr.free()
#     imArr.free()    
#     return wt

   
def istar2d(wtOri,fast=True,gen2=True,normalization=False):
    (nz,nx,ny) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    head = 'star2d_gen2' if gen2 else 'star2d_gen1'
    trans = 1 if gen2 else 2    
    if normalization:
        if pm.trHead != head:
            pm.trHead = head
            pm.trTab = nsNorm(nx,ny,nz,trans)
        for i in np.arange(nz):
            wt[i,:,:] *= pm.trTab[i]
    
    if gen2:
        '''
        h' = h, g' = Dirac
        '''
        step_hole = pow(2,nz-2)
        imRec = np.copy(wt[nz-1,:,:])
        for k in np.arange(nz-2,-1,-1):            
            if fast:
                kernel2d = b3spline_fast(step_hole)
                im_out = psg.convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            else:
                im_out = b3splineTrans(imRec,step_hole)
            imRec = im_out + wt[k,:,:]
            step_hole /= 2
	    step_hole=np.int(step_hole) # JG MOD          
    else:
        '''
        h' = Dirac, g' = Dirac
        '''
#         imRec = np.sum(wt,axis=0)
        '''
        h' = h, g' = Dirac + h
        '''
        imRec = np.copy(wt[nz-1,:,:])
        step_hole = pow(2,nz-2)
        for k in np.arange(nz-2,-1,-1):
            if fast:
                kernel2d = b3spline_fast(step_hole)
                imRec = psg.convolve2d(imRec, kernel2d, boundary='symm',mode='same')
                im_out = psg.convolve2d(wt[k,:,:], kernel2d, boundary='symm',mode='same')
            else:
                imRec = b3splineTrans(imRec,step_hole)
                im_out = b3splineTrans(wt[k,:,:],step_hole)
            imRec += wt[k,:,:]+im_out
            step_hole /= 2
	    step_hole=np.int(step_hole) # JG MOD
    return imRec

# def istar2d_boost(wt,mr2d,gen2=True,adjoint=False,normalization=False):
#     nx = mr2d.size_ima_nl()
#     ny = mr2d.size_ima_nc()
#     nz = mr2d.nbr_scale()
#     if gen2:
#         mr2d.ModifiedATWT = Bool.True
#     else:
#         mr2d.ModifiedATWT = Bool.False
#     for sc in np.arange(nz):
#         mr2d.band(sc).data[:] = wt[sc]
#     # Unnormalization step
#     head = 'star2d_gen2' if gen2 else 'star2d_gen1' 
#     if normalization:
#         if pm.trHead != head:
#             pm.trHead = head
#             pm.trTab = nsNorm_boost(mr2d,gen2)
#         for i in np.arange(nz):
#             tmp = np.copy(mr2d.band(i).data)
#             tmp *= pm.trTab[i]
#             tmpArr = Iflt(nx,ny)
#             tmpArr.data[:] = tmp
# #             mr2d.band(i).data[:] = tmp
#             mr2d.insert_band(tmpArr,i)
#             tmpArr.free()
#             
#     rec = Iflt(nx,ny)    
#     if adjoint:
#         mr2d.rec_adjoint(rec)
#     else:
#         mr2d.recons(rec)
#     imRec = np.copy(rec.data)
#     rec.free()
#     return imRec
        

def adstar2d(wtOri,fast=True,gen2=True,normalization=False):
    (nz,nx,ny) = np.shape(wtOri)
    wt = np.copy(wtOri)
    # Unnormalization step
    # !Attention: wt is not the original wt after unnormalization
    head = 'star2d_gen2' if gen2 else 'star2d_gen1'
    trans = 1 if gen2 else 2    
    if normalization:
        if pm.trHead != head:
            pm.trHead = head
            pm.trTab = nsNorm(nx,ny,nz,trans)
        for i in np.arange(nz):
            wt[i,:,:] *= pm.trTab[i]
    
    imRec = np.copy(wt[nz-1,:,:])
    step_hole = np.int(pow(2,nz-2)) #JG MOD
    for k in np.arange(nz-2,-1,-1):
        if fast:
            kernel2d = b3spline_fast(step_hole)
            imRec = psg.convolve2d(imRec, kernel2d, boundary='symm',mode='same')
            im_out = psg.convolve2d(wt[k,:,:], kernel2d, boundary='symm',mode='same')
            if gen2:
                im_out2 = psg.convolve2d(im_out, kernel2d, boundary='symm',mode='same')
                imRec += wt[k,:,:] -im_out2
            else: imRec += wt[k,:,:] -im_out
        else:
            imRec = b3splineTrans(imRec,step_hole)
            im_out = b3splineTrans(wt[k,:,:],step_hole)
            if gen2:
                im_out2 = b3splineTrans(im_out,step_hole)
                imRec += wt[k,:,:] -im_out2
            else: imRec += wt[k,:,:]-im_out
        step_hole /= 2
    return imRec

def nsNorm(nx,ny,nz,trans=1):
    im = np.zeros((nx,ny))
    im[nx/2,ny/2] = 1
    if trans == 1:                       # starlet transform 2nd generation
        wt = star2d(im,nz,fast=True,gen2=True,normalization=False)
        tmp = wt**2
    elif trans == 2:                      # starlet transform 1st generation
        wt = star2d(im,nz,fast=True,gen2=False,normalization=False)
        tmp = wt**2
    tabNs = np.sqrt(np.sum(np.sum(tmp,1),1))       
    return tabNs
    
# def nsNorm_boost(mr2d,gen2=False):
#     nx = mr2d.size_ima_nl()
#     ny = mr2d.size_ima_nc()
#     im = np.zeros((nx,ny))
#     im[nx/2,ny/2] = 1
#     wt = star2d_boost(im,mr2d,gen2=gen2,normalization=False)
#     tmp = wt**2
#     tabNs = np.sqrt(np.sum(np.sum(tmp,1),1))       
#     return tabNs 
 
def pstar2d(im,nz,Niter,fast=True,gen2=True,hard=False,den=False):
    (nx,ny) = np.shape(im)
    rsd = np.copy(im)
    wt = star2d(rsd,nz,fast,gen2,True)
    mwt = wt.max()
    wt = np.zeros((nz,nx,ny))
    
    for it in np.arange(Niter):
        ld = mwt * (1. - (it+1.)/Niter)
        if ld < 0:
            ld = 0
        print 'lamda='+str(ld)
        tmp = star2d(rsd,nz,fast,gen2,True)
        wt += tmp
        if den:
            noise = mad(wt[0])
            print noise
            hardTh(wt,3*noise)
        if hard:
            hardTh(wt,ld)
        else:
            softTh(wt,ld)
        wt[wt<0] = 0
        rec = istar2d(wt,fast,gen2,True)
        print (rec>=0).all()
        rsd = im - rec
        fits.writeto('pstar2d'+str(it)+'.fits',rsd,clobber=True)
        print (np.abs(rsd)).sum()
    return wt

def fdr(pvalTab,alpha,sigma):
    '''
    Calculate number of significant discoveries for a zero-mean sigma-deviation Gaussian distribution
    
    @param pvalTab: p value table
    
    @param alpha: p value threshold
    
    @param sigma: Noise deviation
    
    @return: Number of discoveries after fdr correction,and ajusted p value
    '''
    N = len(pvalTab)
    # Ascending ordered p-val table
    pvalSort = np.sort(pvalTab)
    # Bonferroni correction
    alphaTab = alpha*(np.arange(N)+1.)/(N)
    nb = (pvalSort<=alphaTab).sum()
    return (nb,pvalSort[nb-1])

def fdrGaussTh(wt,alpha,sig,weights=None):
    '''
    Thresholding using False Discovery Rate (FDR)
    
    @param wt: wavelet coefficient
    
    @param alpha: p value threshold
    
    @param sigma: Noise deviation
    
    @param weights: Weight matrix
    '''
    dim = np.size(np.shape(wt))        
    if dim == 2:
        wt = wt[np.newaxis,:,:]
    (nz,nx,ny) = np.shape(wt)
    if np.size(sig) == 1:
        sigTab = np.ones(nz) * sig
    else:
        sigTab = sig
    for k in np.arange(nz):
        wtScale = wt[k]
        wtScaleTab = wtScale.flatten()
        # Normalization of Gaussian distribution
        pvalTab = special.erfc(np.abs(wtScaleTab)/(np.sqrt(2)*sigTab[k]))
        # The corresponding index of ascending ordered p-val table
        pvalIndSort = np.argsort(pvalTab)
        (nb,qval) = fdr(pvalTab,alpha,sigTab[k])
        if nb>=1:
            th = np.abs(wtScaleTab[pvalIndSort[nb-1]])
        else:
            th = 10.*np.abs(wtScaleTab[pvalIndSort[-1]])
        softTh(wt[k],th,weights)
#         hardTh(wt[k],th,weights)


def fdrValue(wtScale,alpha,sigma):
    (nx,ny) = np.shape(wtScale)
    wtVal = wtScale.reshape(nx*ny)
    # Normalization of Gaussian distribution
    pval = special.erfc(np.abs(wtVal)/(np.sqrt(2)*sigma))
    # Ascending ordered p-val and its corresponding ordered index
    pvalIndSort = np.argsort(pval)
    pvalSort = np.sort(pval)
    # Bonferroni correction
    alphaTab = alpha*(np.arange(nx*ny)+1.)/(nx*ny)
    # Find the index corresponding to q-value (ajusted p-value)
    ind = (pvalSort<=alphaTab).sum()-1
    if ind>=0:
        valRet = np.abs(wtVal[pvalIndSort[ind]])
#         print pval[pvalIndSort[ind]]
    else:
        valRet = 10.*np.abs(wtVal[pvalIndSort[nx*ny-1]])
    return valRet
  
def univTh(alpha,sigTab,weights=None,reweighted=False): 
    dim = np.size(np.shape(alpha))        
    if dim == 2:
        alpha = alpha[np.newaxis,:,:]
    (nz,nx,ny) = np.shape(alpha)
    if np.size(sigTab) == 1:
        thTab = np.ones(nz) * sigTab * np.sqrt(2*np.log2(nx*ny))
    else:
        thTab = sigTab * np.sqrt(2*np.log2(nx*ny))
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i,:,:])[abs(alpha[i,:,:])<=thTab[i]] = 0
            (alpha[i,:,:])[alpha[i,:,:]>0] -= thTab[i]
            (alpha[i,:,:])[alpha[i,:,:]<0] += thTab[i]
        else:
            (alpha[i,:,:])[abs(alpha[i,:,:])<=thTab[i]*weights[i,:,:]] = 0
            (alpha[i,:,:])[alpha[i,:,:]>0] -= (thTab[i]*weights[i,:,:])[alpha[i,:,:]>0]
            (alpha[i,:,:])[alpha[i,:,:]<0] += (thTab[i]*weights[i,:,:])[alpha[i,:,:]<0]
    alpha = np.squeeze(alpha)
    
def wienerTh(alpha,win=9):
    gen2 = True
    dim = np.size(np.shape(alpha))
    if dim == 2:
        alpha = alpha[np.newaxis,:,:]
    (nz,nx,ny) = np.shape(alpha)
    
    head = 'star2d_gen2' if gen2 else 'star2d_gen1'
    trans = 1 if gen2 else 2
    if (pm.trHead != head):
        pm.trHead = head
        pm.trTab = nsNorm(nx,ny,nz,trans)
    stdNFst = mad(alpha[0])
    nsTab = stdNFst*pm.trTab
    for i in np.arange(nz):
        varSN = ndimage.generic_filter(alpha[i,:,:],np.var,size=win)
#         stdN = ndimage.generic_filter(alpha[i,:,:],mad,size=win)
        stdN = nsTab[i]
        varN = stdN**2
        alpha[i,:,:] *= (varSN-varN)/varSN
    alpha = np.squeeze(alpha)
    
def updateWeights(alpha,term,ksigma,wav2d1dNorm=False,scale2d=1,gen2=True):
    '''
    Update weignts for reweighted l1-norm optimization
    
    @param alpha: Coefficients, can be 1d,2d,3d
    
    @param term: Parameter to control the weight function, which is monodimensional
    
    @wav2d1dNorm: Flag for the 2d-1d coefficient, default is False
    
    @param scale2d: Number of 2d decomposition scales, used for 2d-1d decomposition, default is 1
    
    @param gen2: Starlet generation 2, used for 2d-1d decomposition, default is True
    
    @return: The updated weight
    '''
    dimAlpha = np.ndim(alpha)
    dimTerm = np.ndim(term)
    if dimTerm > 1:
        print "Error dimension for updating weights"
        sys.exit()
    # If coefficient is 3 dimensional
    if dimAlpha == 3:
        (nz,nx,ny) = np.shape(alpha)
        if np.size(term) == nz:
            weights = np.ones_like(alpha)               
            if wav2d1dNorm:
                frame = nz/scale2d
                head = 'star2d_gen2' if gen2 else 'star2d_gen1'     # Store the transformation head and calculate impulse response of transformation
                trans = 1 if gen2 else 2
                if pm.trHead != head:
                    pm.trTab = nsNorm(nx,ny,scale2d,trans)
                    pm.trHead = head
            for i in np.arange(nz):
    #             index = (np.abs(alpha[i]) < 3 * term[i])
    #             (weights[i])[index] = 3*term[i]/(np.abs(alpha[i])[index])
    #             eps = term[i]
#                 eps = 0.01
#                 if wav2d1dNorm:
#                     sc = (i/int(frame))%scale2d
#                     eps *= pm.trTab[-1-sc]
#                 noiseMat = np.ones_like(alpha[i])*eps
#                 index = (np.abs(alpha[i]) >= eps)
#                 (noiseMat)[index] = (np.abs(alpha[i]))[index]
#                 weights[i] = 0.2*term[i]/noiseMat
                
#                 eps = 0.01
                weightsScale = np.ones_like(alpha[i])
                index = (np.abs(alpha[i]) >= ksigma*term[i])
                weightsScale[index] = ksigma*term[i]/(np.abs(alpha[i])[index])
                weights[i] = np.copy(weightsScale)
                
        elif np.size(term) == 1:
#             eps = 0.001
#             noiseMat = np.ones_like(alpha)*eps
#             index = (np.abs(alpha) >= eps)
#             (noiseMat)[index] = (np.abs(alpha))[index]
#             weights = 3*term/noiseMat
            weights = np.ones_like(alpha)
            index = (np.abs(alpha) >= ksigma*term)
            weights[index] = ksigma*term/(np.abs(alpha)[index])  
        else:
            print "Error dimension for updating weights"
            sys.exit()                      
                
                
    # If coefficient is 2 dimensional            
    elif dimAlpha == 2:
        (nx,ny) = np.shape(alpha)
        if np.size(term) == 1:
#             eps = 0.001
#             noiseMat = np.ones_like(alpha)*eps
#             index = (np.abs(alpha) >= eps)
#             (noiseMat)[index] = (np.abs(alpha))[index]
#             weights = 3*term/noiseMat
            weights = np.ones_like(alpha)
            index = (np.abs(alpha) >= ksigma*term)
            weights[index] = ksigma*term/(np.abs(alpha)[index])
        elif np.size(term) == nx:
            weights = np.ones_like(alpha)                              
            for i in np.arange(nx):
    #             index = (np.abs(alpha[i]) < 3 * term[i])
    #             (weights[i])[index] = 3*term[i]/(np.abs(alpha[i])[index])
    #             eps = term[i]
#                 eps = 0.001
#                 noiseMat = np.ones_like(alpha[i])*eps
#                 index = (np.abs(alpha[i]) >= eps)
#                 (noiseMat)[index] = (np.abs(alpha[i]))[index]
#                 weights[i] = 3*term[i]/noiseMat                
                weightsScale = np.ones_like(alpha[i])
                index = (np.abs(alpha[i]) >= ksigma*term[i])
                weightsScale[index] = ksigma*term[i]/(np.abs(alpha[i])[index])
                weights[i] = np.copy(weightsScale)
        elif np.size(term) == ny:
            weights = np.ones_like(alpha)                              
            for i in np.arange(ny):
    #             index = (np.abs(alpha[i]) < 3 * term[i])
    #             (weights[i])[index] = 3*term[i]/(np.abs(alpha[i])[index])
    #             eps = term[i]
#                 eps = 0.001
#                 noiseMat = np.ones_like(alpha[:,i])*eps
#                 index = (np.abs(alpha[:,i]) >= eps)
#                 (noiseMat)[index] = (np.abs(alpha[:,i]))[index]
#                 weights[:,i] = 3*term[i]/noiseMat
                
                weightsScale = np.ones_like(alpha[:,i])
                index = (np.abs(alpha[:,i]) >= ksigma*term[i])
                weightsScale[index] = ksigma*term[i]/(np.abs(alpha[:,i])[index])
                weights[:,i] = np.copy(weightsScale)
        else:
                print "Error dimension for updating weights"
                sys.exit()
                  
            
    # If coefficient is 1 dimensional
    elif dimAlpha == 1:
        nx = np.size(alpha)
        if np.size(term) == 1 or np.size(term) == nx:
#             eps = 0.001
#             noiseMat = np.ones_like(alpha)*eps
#             index = (np.abs(alpha) >= eps)
#             (noiseMat)[index] = (np.abs(alpha))[index]
#             weights = 3*term/noiseMat
            
            weights = np.ones_like(alpha)
            index = (np.abs(alpha) >= ksigma*term)
            weights[index] = ksigma*term/(np.abs(alpha)[index])
        else:
            print "Error dimension for updating weights"
            sys.exit()
            
#     if dim == 1 and np.size(term) == nz:
#         weights = np.ones_like(alpha)
# #         weights = np.zeros((nz,nx,ny))
#         if wav2d1dNorm:
#             frame = nz/scale2d
#             head = 'star2d_gen2' if gen2 else 'star2d_gen1'     # Store the transformation head and calculate impulse response of transformation
#             trans = 1 if gen2 else 2
#             if pm.trHead != head:
#                 pm.trTab = nsNorm(nx,ny,scale2d,trans)
#                 pm.trHead = head
#         for i in np.arange(nz):
# #             index = (np.abs(alpha[i]) < 3 * term[i])
# #             (weights[i])[index] = 3*term[i]/(np.abs(alpha[i])[index])
# #             eps = term[i]
#             eps = 0.001
#             if wav2d1dNorm:
#                 sc = (i/int(frame))%scale2d
#                 eps *= pm.trTab[-1-sc]
#             noiseMat = np.ones_like(alpha[i])*eps
#             index = (np.abs(alpha[i]) >= eps)
#             (noiseMat)[index] = (np.abs(alpha[i]))[index]
#             weights[i] = 3*term[i]/noiseMat
#             weights[i] = eps/noiseMat
#             print i
#             (weights[i])[np.abs(alpha[i]) <= 3 * term[i]] = 1
#             index = np.abs(alpha[i]) > 3 * term[i]
#             (weights[i])[index] = 3 * term[i]/np.abs(alpha[index])
#             noiseMat[i] = 3*term[i]
#             noiseMat[i] = np.ones((nx,ny)) * 5*0.01
#         weights = 1./(1.+np.abs(alpha)/noiseMat)
#         weights = 1./(abs(alpha)+0.4)
#     elif np.shape(term) == (nz,nx,ny):
#         weights = 1./(1.+np.abs(alpha)/term) 
    return weights
