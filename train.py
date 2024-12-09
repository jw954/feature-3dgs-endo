#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, tv_loss 
from gaussian_renderer import gsplat_render as gsplat_render, network_gui
from gaussian_renderer import render as render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder
from models.semantic_dataloader import VariableSizeDataset
from torch.utils.data import DataLoader
from utils.timer import Timer

import lpips
import cv2

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset, dataset.sh_degree, hyper)
    print('initalized gaussian model!')
    scene = Scene(dataset, gaussians)
    timer = Timer()
    timer.start()
    print('done initializing scene\n')
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, 3000, timer)

# #TODO: continue rest of training implementation of EndoGaussian, it is different from feature3dgs. Below is the old training routine for feature3dgs
#     # 2D semantic feature map CNN decoder
#     viewpoint_stack = scene.getTrainCameras().copy()
#     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
#     gt_feature_map = viewpoint_cam.semantic_feature.cuda()
#     feature_out_dim = gt_feature_map.shape[0]

    
#     # speed up for SAM
#     if dataset.speedup:
#         feature_in_dim = int(feature_out_dim/2)
#         cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
#         cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)


#     gaussians.training_setup(opt)
#     if checkpoint:
#         (model_params, first_iter) = torch.load(checkpoint)
#         gaussians.restore(model_params, opt)

#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#     iter_start = torch.cuda.Event(enable_timing = True)
#     iter_end = torch.cuda.Event(enable_timing = True)

#     viewpoint_stack = None
#     ema_loss_for_log = 0.0
#     progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
#     first_iter += 1


#     for iteration in range(first_iter, opt.iterations + 1):

#         iter_start.record()

#         gaussians.update_learning_rate(iteration)

#         # Every 1000 its we increase the levels of SH up to a maximum degree
#         if iteration % 1000 == 0:
#             gaussians.oneupSHdegree()

#         # Pick a random Camera
#         if not viewpoint_stack:
#             viewpoint_stack = scene.getTrainCameras().copy()
#         viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

#         # Render
#         if (iteration - 1) == debug_from:
#             pipe.debug = True
#         render_pkg = render(viewpoint_cam, gaussians, pipe, background)        

#         feature_map, image, viewspace_point_tensor, visibility_filter, radii = render_pkg["feature_map"], render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        

#         # Loss
#         gt_image = viewpoint_cam.original_image.cuda()
#         Ll1 = l1_loss(image, gt_image)
#         gt_feature_map = viewpoint_cam.semantic_feature.cuda()
#         feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) 
#         if dataset.speedup:
#             feature_map = cnn_decoder(feature_map)
#         Ll1_feature = l1_loss(feature_map, gt_feature_map) 
#         loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 1.0 * Ll1_feature 

#         loss.backward()
#         iter_end.record()

#         with torch.no_grad():
#             # Progress bar
#             ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
#             if iteration % 10 == 0:
#                 progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
#                 progress_bar.update(10)
#             if iteration == opt.iterations:
#                 progress_bar.close()

#             # Log and save
#             training_report(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background)) 
#             if (iteration in saving_iterations):
#                 print("\n[ITER {}] Saving Gaussians".format(iteration))
#                 scene.save(iteration)
#                 print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
#                 if dataset.speedup:
#                     torch.save(cnn_decoder.state_dict(), scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
  

#             # Densification
#             if iteration < opt.densify_until_iter:
#                 # Keep track of max radii in image-space for pruning
#                 gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
#                 gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

#                 if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
#                     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
#                     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
#                 if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
#                     gaussians.reset_opacity()
            

#             # Optimizer step
#             if iteration < opt.iterations:
#                 gaussians.optimizer.step()
#                 gaussians.optimizer.zero_grad(set_to_none = True)
#                 if dataset.speedup:
#                     cnn_decoder_optimizer.step()
#                     cnn_decoder_optimizer.zero_grad(set_to_none = True)

#             if (iteration in checkpoint_iterations):
#                 print("\n[ITER {}] Saving Checkpoint".format(iteration))
#                 torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

#         with torch.no_grad():        
#             if network_gui.conn == None:
#                 network_gui.try_connect(dataset.render_items)
#             while network_gui.conn != None:
#                 try:
#                     net_image_bytes = None
#                     custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
#                     if custom_cam != None:
#                         render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
#                         net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
#                         net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
#                     metrics_dict = {
#                         "#": gaussians.get_opacity.shape[0],
#                         "loss": ema_loss_for_log
#                         # Add more metrics as needed
#                     }
#                     # Send the data
#                     network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
#                     if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
#                         break
#                 except Exception as e:
#                     # raise e
#                     network_gui.conn = None
            # TODO: change device in scene reconstruction
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer, device = "cuda:0"):
    print('reconstruction initiating')
    print('using device:',  device)
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # print('creating background')
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    #vgg net used to compute some perceptual loss function, not trained during training routine
    lpips_model = lpips.LPIPS(net="vgg").cuda()
    video_cams = scene.getVideoCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            batch_size = 16
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        
        # if opt.dataloader: set dataloader to false by default
        #     try:
        #         viewpoint_cams = next(loader)
        #     except StopIteration:
        #         print("reset dataloader")
        #         loader = iter(viewpoint_stack_loader)
        # else:
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            # depth_refine_iteration = 5
            # depth_refine_bounds = 4 * depth_refine_iteration
            # if iteration % depth_refine_iteration==0 and iteration <= depth_refine_bounds:
            #     depth_diff = torch.pow(gt_depth - depth, 2) * mask
            #     depth_diff = depth_diff.reshape(depth_diff.shape[0], -1)
            #     quantile = torch.quantile(depth_diff, 1.0 - 0.1, dim=1, keepdim=True)
            #     depth_to_refine = (depth_diff > quantile).reshape(*gt_depth.shape)
            #     gt_depth[depth_to_refine] = depth[depth_to_refine]
            #     viewpoint_cam.original_depth = gt_depth.detach().cpu()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images,0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)
                
        # mask_tensor = None
        if iteration < 1000:
            color_diff = torch.pow(image_tensor-gt_image_tensor, 2).sum(dim=1, keepdim=True)
            color_diff = color_diff.reshape(color_diff.shape[0], -1)
            quantile = torch.quantile(color_diff, 0.98, dim=1)
            color_to_refine = (color_diff > quantile).reshape(*mask_tensor.shape)
            mask_tensor[color_to_refine] = torch.ones(color_to_refine.sum()).bool().cuda()
                
        if iteration % 500 == 0:
            tmp = (color_to_refine.squeeze().detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('color_to_refine.png', tmp)
            tmp = (image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('image.png', tmp)
            tmp = (gt_image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite('gtimage.png', tmp)
            tmp = (mask_tensor.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite('mask.png', tmp)
        
        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)
        Ll1_depth = l1_loss(depth_tensor, gt_depth_tensor)
        # depth_tvloss = TV_loss(depth_tensor, mask_tensor)
        
        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()        

        loss = Ll1 + Ll1_depth
        
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

                #no render process defined in dataset in feature3dgs...
            # if dataset.render_process:
            #     if (iteration < 1000 and iteration % 10 == 1) \
            #         or (iteration < 3000 and iteration % 50 == 1) \
            #             or (iteration < 10000 and iteration %  100 == 1) \
            #                 or (iteration < 60000 and iteration % 100 ==1):
            #         render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            # timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                #add densification stats is different for endogaussin
                # gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    #hard code these
                    opacity_threshold = 0.005
                    # opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = 0.0002 
                    # densify_threshold = opt.densify_grad_threshold_coarse
                else:  
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #     gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > 500 and iteration % 100 == 0 :
                    size_threshold = 20 if iteration > 3000 else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                #     size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                #     gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > 500 and iteration % 100 == 0:
                    size_threshold = 40 if iteration > 3000 else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % 3000 == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

#for feature3dgs
def training_report_old(tb_writer, iteration, Ll1, Ll1_feature, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    dataset = lp.extract(args)
    print("dataset device:", dataset.data_device)
    dataset.data_device = "cuda:0"
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset, hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
