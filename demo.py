from cgi import test
from doctest import testfile
from nis import cat
import torch
from models.dior_model import DIORModel
import os, json
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


dataroot = '/home/user/YH/dressing-in-order/all_data/DeepFashion'
resultroot = '/home/user/YH/dressing-in-order/result'
exp_name = 'DIOR_64' # DIORv1_64
epoch = 'latest'
netG = 'dior' # diorv1
ngf = 64
#cuda5 = torch.device('cuda:5')
## this is a dummy "argparse" 
class Opt:
    def __init__(self):
        pass
if True:
    opt = Opt()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 8; opt.n_kpts = 18; opt.style_nc = 64
    opt.n_style_blocks = 4; opt.netG = netG; opt.netE = 'adgan'
    opt.ngf = ngf
    opt.norm_type = 'instance'; opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'; opt.init_gain = 0.02; opt.gpu_ids = [0]
    opt.frozen_flownet = True; opt.random_rate = 1; opt.perturb = False; opt.warmup=False
    opt.name = exp_name
    opt.vgg_path = ''; 
    opt.flownet_path = '/home/user/YH/dressing-in-order/checkpoints/DIOR_64/latest_net_Flow.pth'
    opt.checkpoints_dir = 'checkpoints'
    opt.frozen_enc = True
    opt.load_iter = 0
    opt.epoch = epoch
    opt.verbose = False

#device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu") 
# create model
model = DIORModel(opt)
model.setup(opt)

# load data
from datasets.deepfashion_datasets import DFVisualDataset
Dataset = DFVisualDataset
ds = Dataset(dataroot=dataroot, dim=(256,176), n_human_part=8)


# preload a set of pre-selected models defined in "standard_test_anns.txt" for quick visualizations 
# inputs = dict()
# for attr in ds.attr_keys:
#     inputs[attr] = ds.get_attr_visual_input(attr)
    
# define some tool functions for I/O
def load_img(pid, ds):
    if isinstance(pid,str): # load pose from scratch
        return None, None, load_pose_from_json(pid)
    if len(pid[0]) < 10: # load pre-selected models
        # person = inputs[pid[0]]
        # # for i in person:
        # #     i = i.cuda(device='cuda:5') 
        # #     print(i)
        # person = (i.cuda() for i in person)
        
        # pimg, parse, to_pose = person
        # pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
        pass
    else: # load model from scratch
        person = ds.get_inputs_by_key(pid[0])
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()

def load_pose_from_json(ani_pose_dir):
    with open(ani_pose_dir, 'r') as f:
        anno = json.load(f)
    len(anno['people'][0]['pose_keypoints_2d'])
    anno = list(anno['people'][0]['pose_keypoints_2d'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])

    coord = np.concatenate([x[:,None], y[:,None]], -1)
    #import pdb; pdb.set_trace()
    #coord = (coord * 1.1) - np.array([10,30])[None, :]
    pose  = pose_utils.cords_to_map(coord, (256,176), (256, 256))
    pose = np.transpose(pose,(2, 0, 1))
    pose = torch.Tensor(pose)
    return pose

def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None, result='./',task='tuckin',gender="WOMEN"):
    if pose != None:
        import utils.pose_utils as pose_utils
        print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1,2,0),radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2 # denormalize
        out = np.transpose(out, [1,2,0])

        if pose != None:
            out = np.concatenate((kpt, out),1)
    else:
        out = kpt

    genderdir=os.path.join(resultroot,gender)
    taskdir=os.path.join(genderdir,task)
    if not os.path.exists(taskdir):
        os.makedirs(taskdir)
    resultdir=os.path.join(taskdir,result+'.png')

    #fig = plt.figure(figsize=(6,4), dpi= 100, facecolor='w', edgecolor='k')
    plt.axis('off')
    plt.imsave(resultdir,out)
    #plt.imshow(out)

# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5,1,3,2], perturb=False):
    PID = [0,4,6,7]
    GID = [2,5,1,3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    if perturb:
        # pimg = perturb_images(pimg[None])[0]
        pass
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
    
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])
   
    
    # swap base garment if any
    gimgs = []
    for gid in gids:
        _,_,k = gid
        gimg, gparse, pose =  load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2])]

    # encode garment (overlay)
    garments = []
    over_gsegs = []
    oimgs = []
    for gid in ogids:
        oimg, oparse, pose = load_img(gid, ds)
        oimgs += [oimg * (oparse == gid[2])]
        seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
        over_gsegs += [seg]
    
    gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)
    
    return pimg, gimgs, oimgs, gen_img[0], to_pose



if __name__=='__main__':
    testfilepath=os.path.join(dataroot, 'fasion-annotation-%s.csv' % 'test')
    test_file=pd.read_csv(testfilepath, sep=':',header=0)
    name_list = test_file['name'].to_numpy()#how to index in panda
    #print(name_list[0])
    category_list =['Tees_Tanks', 'Blouses_Shirts', 'Dresses', 'Shorts', 'Sweaters', 'Pants', 'Skirts', 'Jackets_Coats', 'Rompers_Jumpsuits', 'Sweatshirts_Hoodies', 
                    'Cardigans', 'Graphic_Tees', 'Denim', 'Shirts_Polos', 'Jackets_Vests', 'Leggings', 'Suiting']
    gender_list = ['MEN','WOMEN']
    dire_list = ['1front','2side','3back','4full','6flat','7additional']

    
    import yaml
    if not os.path.exists('all_test.yaml'):
        all_test={gender_list[0]:{}, gender_list[1]:{}}
        for dfcategory in category_list:
            all_test[gender_list[0]][dfcategory]=[]
            all_test[gender_list[1]][dfcategory]=[]
            for i in range(len(name_list)):
                if name_list[i][7]==gender_list[1][0]:#woman
                    if name_list[i].split('WOMEN')[1].split('id0')[0] == dfcategory:
                            all_test['WOMEN'][dfcategory].append(name_list[i])
                else:#Man
                    if name_list[i].split('MEN')[1].split('id0')[0] == dfcategory:
                            all_test['MEN'][dfcategory].append(name_list[i])
        with open('all_test.yaml','w') as f:
            yaml.dump(all_test,f)
    else:
        with open('all_test.yaml','r') as f:
            all_test=yaml.load(f,Loader=yaml.FullLoader)
    
    ############input########
    task = 'layermultiple'#'tuckin','layersingle' or 'layermultiple'
    count = 1

    ########################process################################
    if task == 'tuckin':
        for gender in gender_list:
            for category_p in all_test[gender].keys():
                for i in range(3):
                    if len(all_test[gender][category_p]) !=0:
                        np.random.seed(i)
                        index_p = np.random.choice(len(all_test[gender][category_p]))
                        pid = (all_test[gender][category_p][index_p], None, None)
                        for category_g in all_test[gender].keys():
                            if category_p == category_g:
                                pass
                            else:
                                if len(all_test[gender][category_g]) != 0:
                                    index_g = np.random.choice(len(all_test[gender][category_g]))    
                                    gids = [
                                            (all_test[gender][category_g][index_g], None, 5),
                                            (all_test[gender][category_p][index_p], None, 1)
                                            ]
                                    result='{}-{}_2_{}-{}'.format(category_p,index_p,category_g,index_g)
                                    order =[2, 1, 5]#[2,5,1] or [2,1,5]
                                    task_name='tuckin{}{}'.format(order[1],order[2])
                                    pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model=model, pid=pid, gids=gids, order=order)
                                    print(count)
                                    plot_img(pimg, gimgs, gen_img=gen_img, result=result, task=task_name,gender=gender)
                                    count+=1
    elif task =='layersingle':
        for gender in gender_list:
            for category_p in all_test[gender].keys():
                for i in range(3):
                    if len(all_test[gender][category_p]) !=0:
                        np.random.seed(i)
                        index_p = np.random.choice(len(all_test[gender][category_p]))
                        pid = (all_test[gender][category_p][index_p],None,5)
                        for category_g in all_test[gender].keys():
                            if category_p == category_g:
                                pass
                            else:
                                if len(all_test[gender][category_g]) != 0:
                                    index_g = np.random.choice(len(all_test[gender][category_g]))    
                                    ogids = [
                                            (all_test[gender][category_g][index_g], None, 5)
                                            ]
                                    result='{}-{}_with_{}-{}'.format(category_p,index_p,category_g,index_g)
                                    task_name=task
                                    pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, ogids=ogids)
                                    plot_img(pimg=pimg, gimgs=gimgs, oimgs=oimgs, gen_img=gen_img, result=result, task=task_name,gender=gender)
    else:
            for gender in gender_list:
                for category_p in all_test[gender].keys():
                    for i in range(3):
                        if len(all_test[gender][category_p]) !=0:
                            np.random.seed(i)
                            index_p = np.random.choice(len(all_test[gender][category_p]))
                            pid = (all_test[gender][category_p][index_p],None,None)
                            for category_g1 in all_test[gender].keys():
                                if category_p == category_g1:
                                    pass
                                else:
                                    if len(all_test[gender][category_g1]) != 0:
                                        index_g1 = np.random.choice(len(all_test[gender][category_g1]))  
                                        for category_g2 in all_test[gender].keys(): 
                                            if category_g2 != category_g1 and category_g2 != category_p:
                                                if len(all_test[gender][category_g2]) != 0:
                                                    index_g2 = np.random.choice(len(all_test[gender][category_g2]))  
                                                    gids = [
                                                            (all_test[gender][category_g1][index_g1], None, 5),
                                                            (all_test[gender][category_g2][index_g2], None, 1)
                                                            ]
                                                    for category_og1 in all_test[gender].keys():
                                                        if category_p != category_og1 and category_og1 != category_g1 and category_og1!= category_g2:
                                                            if len(all_test[gender][category_og1]) != 0:
                                                                index_og1 = np.random.choice(len(all_test[gender][category_og1]))  
                                                                for category_og2 in all_test[gender].keys(): 
                                                                    if category_og2 != category_og1 and category_og2 != category_p and category_og2 != category_g1 and category_og2 != category_g2:
                                                                        if len(all_test[gender][category_og2]) != 0:
                                                                            index_og2 = np.random.choice(len(all_test[gender][category_og2]))  
                                                                            ogids = [
                                                                                    (all_test[gender][category_og1][index_og1], None, 5),
                                                                                    (all_test[gender][category_og2][index_og2], None, 3)
                                                                                    ]
                                                                            result='{}-{}_with_{}-{}_{}-{}_{}-{}_{}-{}'.format(category_p,index_p,category_g1,index_g1,category_g2,index_g2,category_og1,index_og1,category_og2,index_og2)
                                                                            task_name=task
                                                                            pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid=pid, gids=gids, ogids=ogids)
                                                                            plot_img(pimg=pimg, gimgs=gimgs, oimgs=oimgs, gen_img=gen_img, result=result, task=task_name, gender=gender)





    