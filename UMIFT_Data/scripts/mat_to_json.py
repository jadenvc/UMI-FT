import scipy.io as sio, json, numpy as np

sensor_name = 'NFT5'
mat = sio.loadmat(sensor_name+'_norm_constants.mat')['norm_const'].ravel()[0]
def vec(key): return np.asarray(mat[key]).ravel().tolist()

norm = {
    'mu_x': vec('mu_x'),
    'sd_x': vec('sd_x'),
    'mu_y': vec('mu_y'),
    'sd_y': vec('sd_y')
}
with open(sensor_name+'_norm.json','w') as f: json.dump(norm,f,indent=2)