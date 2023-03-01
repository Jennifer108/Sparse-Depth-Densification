import argparse
import torch
import torch.nn as nn
import torch.nn.parallel

from models import modulesour, netour, resnet, densenet, senet
# import loaddataourmake3d
import loaddataour
import util
import numpy as np
import sobel
import matplotlib.image
import matplotlib.pyplot as plt
import cv2


def main():
    model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
    # model = torch.nn.DataParallel(model).cuda()
    # model.load_state_dict(torch.load('checkpoint10.pth'))

    model =model.cuda()
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load("checkpoint.pth").items()})

    test_loader = loaddataour.getTestingData(1)
    test(test_loader, model, 0.25)

def compute_errors_make3d(gt, pred):
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    ee=np.abs(gt - pred) / gt
    ee[np.where(ee==np.inf)]=0
    abs_rel = np.mean(ee)
    sq_rel = np.mean(((gt - pred)**2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log

def test(test_loader, model, thre):
    model.eval()

    totalNumber = 0

    Ae = 0
    Pe = 0
    Re = 0
    Fe = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    for i, sample_batched in enumerate(test_loader):
        image, smcdepth,errormap,edge,depth,name = sample_batched['image'],sample_batched['smcdepth'],sample_batched['errormap'],sample_batched['edge'], sample_batched['depth'], sample_batched['name']

        # depth1=depth.squeeze(0).squeeze(0)
        # sparse_depth = cv2.applyColorMap((depth1.cpu().numpy()*255*1000).astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imshow("dede",sparse_depth)
        # cv2.waitKey()

        depth = depth.cuda()
        image = image.cuda()
        smcdepth=smcdepth.cuda()
        edge=edge.cuda()
        errormap=errormap.cuda()
        with torch.no_grad():
            image = torch.autograd.Variable(image)
            depth = torch.autograd.Variable(depth)
            smcdepth=torch.autograd.Variable(smcdepth)
            edge=torch.autograd.Variable(edge)
            errormap=torch.autograd.Variable(errormap)

            output = model(image,smcdepth,errormap,edge)
            # matplotlib.image.imsave('E:\\semanticSegmentation\\DepthEstimation\\ourmethod\\net\\data\\results\\our1000\\'+name[0], output.view(output.size(2),output.size(3)).data.cpu().numpy(),cmap = 'jet')
            output = torch.nn.functional.interpolate(output, size=[depth.size(2),depth.size(3)], mode='bilinear',align_corners=False)

            # compute_errors_make3d(depth.cpu().numpy().squeeze(0).squeeze(0)*1000,output.cpu().numpy().squeeze(0).squeeze(0))
            # save_pred_depth = predicted_depth * 65535/80.0
            # save_pred_depth[save_pred_depth<1e-3] = 1e-3
            # save_pred_img = Image.fromarray(save_pred_depth.astype(np.int32), 'I')
            # save_pred_img.save('%s/%05d_pred.png'%(self.output_images_dir, t_id_global))

            depth_edge = edge_detection(depth)
        output_edge = edge_detection(output)

        batchSize = depth.size(0)
        totalNumber = totalNumber + batchSize
        
        #output=output* 65535/80.0

        errors = util.evaluateError(output, depth)
        errorSum = util.addErrors(errorSum, errors, batchSize)
        averageError = util.averageErrors(errorSum, totalNumber)

        # plt.figure("Image")
        # plt.imshow(depth_edge.cpu().numpy().squeeze())
        # plt.figure("Image1")
        # plt.imshow(output_edge.cpu().numpy().squeeze())
        # plt.show()


    #     edge1_valid = (depth_edge > thre)
    #     edge2_valid = (output_edge > thre)

    #     nvalid = np.sum(torch.eq(edge1_valid, edge2_valid).float().data.cpu().numpy())
    #     A = nvalid / (depth.size(2)*depth.size(3))

    #     nvalid2 = np.sum(((edge1_valid + edge2_valid) ==2).float().data.cpu().numpy())
    #     P = nvalid2 / (np.sum(edge2_valid.data.cpu().numpy()))
    #     R = nvalid2 / (np.sum(edge1_valid.data.cpu().numpy()))

    #     F = (2 * P * R) / (P + R)

    #     Ae += A
    #     Pe += P
    #     Re += R
    #     Fe += F

    # Av = Ae / totalNumber
    # Pv = Pe / totalNumber
    # Rv = Re / totalNumber
    # Fv = Fe / totalNumber
    # print('PV', Pv)
    # print('RV', Rv)
    # print('FV', Fv)

    averageError['RMSE'] = np.sqrt(averageError['MSE'])
    print(averageError)

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modulesour.E_resnet(original_model) 
        model = netour.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modulesour.E_densenet(original_model)
        model = netour.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modulesour.E_senet(original_model)
        model = netour.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    return model
   

def edge_detection(depth):
    get_edge = sobel.Sobel().cuda()

    edge_xy = get_edge(depth)
    edge_sobel = torch.pow(edge_xy[:, 0, :, :], 2) + \
        torch.pow(edge_xy[:, 1, :, :], 2)
    edge_sobel = torch.sqrt(edge_sobel)

    return edge_sobel


if __name__ == '__main__':
    main()
