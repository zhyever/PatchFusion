import torch
import copy

if __name__ == '__main__':
    ckp = torch.load('nfs/monodepth3_checkpoints/paper/tune_double_ct_mix_0.1w_0.01_dblen_HQDepthhqdepth_20-Oct_03-58-9ad738a52f6f_latest.pt')
    weight = copy.deepcopy(ckp['model'])
    #  Unexpected key(s) in state_dict: "blur_mask", "mlp_feat.conv0.weight", "mlp_feat.conv0.bias", "mlp_feat.conv1.weight", "mlp_feat.conv1.bias", "mlp_feat.conv2.weight", "mlp_feat.conv2.bias", "mlp_bin.conv0.weight", "mlp_bin.conv0.bias", "mlp_bin.conv1.weight", "mlp_bin.conv1.bias"
    del weight['blur_mask']
    del weight['mlp_feat.conv0.weight']
    del weight['mlp_feat.conv0.bias']
    del weight['mlp_feat.conv1.weight']
    del weight['mlp_feat.conv1.bias']
    del weight['mlp_feat.conv2.weight']
    del weight['mlp_feat.conv2.bias']
    del weight['mlp_bin.conv0.weight']
    del weight['mlp_bin.conv0.bias']
    del weight['mlp_bin.conv1.weight']
    del weight['mlp_bin.conv1.bias']
    
    ckp['model'] = weight
    torch.save(ckp, 'nfs/monodepth3_checkpoints/paper/patchfusion_u4k.pt')