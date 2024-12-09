# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# from scene import Scene
# import os
# from tqdm import tqdm
# from os import makedirs
# from gaussian_renderer import gsplat_render as render, render_edit
# import torchvision
# from utils.general_utils import safe_state
# from argparse import ArgumentParser
# from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
# import cv2
# import matplotlib.pyplot as plt
# from utils.graphics_utils import getWorld2View2
# from utils.pose_utils import render_path_spiral
# import sklearn
# import sklearn.decomposition
# import numpy as np
# from PIL import Image
# import torch.nn as nn
# import torch.nn.functional as F
# from utils.clip_utils import CLIPEditor
# import yaml
# from models.networks import CNN_decoder, MLP_encoder


# def feature_visualize_saving(feature):
#     fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
#     fmap = nn.functional.normalize(fmap, dim=1)
#     pca = sklearn.decomposition.PCA(3, random_state=42)
#     f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy()
#     transformed = pca.fit_transform(f_samples)
#     feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
#     feature_pca_components = torch.tensor(pca.components_).float().cuda()
#     q1, q99 = np.percentile(transformed, [1, 99])
#     feature_pca_postprocess_sub = q1
#     feature_pca_postprocess_div = (q99 - q1)
#     del f_samples
#     vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
#     vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
#     vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu()
#     return vis_feature


# def parse_edit_config_and_text_encoding(edit_config):
#     edit_dict = {}
#     if edit_config is not None:
#         with open(edit_config, 'r') as f:
#             edit_config = yaml.safe_load(f)
#             print(edit_config)
#         objects = edit_config["edit"]["objects"]
#         targets = edit_config["edit"]["targets"].split(",")
#         edit_dict["positive_ids"] = [objects.index(t) for t in targets if t in objects]
#         edit_dict["score_threshold"] = edit_config["edit"]["threshold"]
        
#         # text encoding
#         clip_editor = CLIPEditor()
#         text_feature = clip_editor.encode_text([obj.replace("_", " ") for obj in objects])

#         # setup editing
#         op_dict = {}
#         for operation in edit_config["edit"]["operations"].split(","):
#             if operation == "extraction":
#                 op_dict["extraction"] = True
#             elif operation == "deletion":
#                 op_dict["deletion"] = True
#             elif operation == "color_func":
#                 op_dict["color_func"] = eval(edit_config["edit"]["colorFunc"])
#             else:
#                 raise NotImplementedError
#         edit_dict["operations"] = op_dict

#         idx = edit_dict["positive_ids"][0]

#     return edit_dict, text_feature, targets[idx]
        


# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, edit_config, speedup):
#     if edit_config != "no editing":
#         edit_dict, text_feature, target = parse_edit_config_and_text_encoding(edit_config)

#         edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
#         edit_gts_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt")
#         edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")
#         edit_gt_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "gt_feature_map")

#         makedirs(edit_render_path, exist_ok=True)
#         makedirs(edit_gts_path, exist_ok=True)
#         makedirs(edit_feature_map_path, exist_ok=True)
#         makedirs(edit_gt_feature_map_path, exist_ok=True)
    
#     else:
#         render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#         gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#         feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
#         gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
#         saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
#         #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
#         decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
#         depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth") ###
        
#         if speedup:
#             gt_feature_map = views[0].semantic_feature.cuda()
#             feature_out_dim = gt_feature_map.shape[0]
#             feature_in_dim = int(feature_out_dim/2)
#             cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
#             cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
#         makedirs(render_path, exist_ok=True)
#         makedirs(gts_path, exist_ok=True)
#         makedirs(feature_map_path, exist_ok=True)
#         makedirs(gt_feature_map_path, exist_ok=True)
#         makedirs(saved_feature_path, exist_ok=True)
#         makedirs(depth_path, exist_ok=True) ###

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         if edit_config != "no editing":
#             render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict) 
#             gt = view.original_image[0:3, :, :]
#             gt_feature_map = view.semantic_feature.cuda() 
#             torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) 
#             torchvision.utils.save_image(gt, os.path.join(edit_gts_path, '{0:05d}'.format(idx) + ".png"))
#             # visualize feature map
#             feature_map = render_pkg["feature_map"]
#             feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
#             if speedup:
#                 feature_map = cnn_decoder(feature_map)

#             feature_map_vis = feature_visualize_saving(feature_map)
#             Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
#             gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
#             Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

#         else:
#             render_pkg = render(view, gaussians, pipeline, background) 

#             gt = view.original_image[0:3, :, :]
#             gt_feature_map = view.semantic_feature.cuda() 
#             torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
#             torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
#             ### depth ###
#             depth = render_pkg["depth"]
#             scale_nor = depth.max().item()
#             depth_nor = depth / scale_nor
#             depth_tensor_squeezed = depth_nor.squeeze()  # Remove the channel dimension
#             colormap = plt.get_cmap('jet')
#             depth_colored = colormap(depth_tensor_squeezed.cpu().numpy())
#             depth_colored_rgb = depth_colored[:, :, :3]
#             depth_image = Image.fromarray((depth_colored_rgb * 255).astype(np.uint8))
#             output_path = os.path.join(depth_path, '{0:05d}'.format(idx) + ".png")
#             depth_image.save(output_path)
#             ##############

#             # visualize feature map
#             feature_map = render_pkg["feature_map"] 
#             feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
#             if speedup:
#                 feature_map = cnn_decoder(feature_map)

#             feature_map_vis = feature_visualize_saving(feature_map)
#             Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
#             gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
#             Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

#             # save feature map
#             feature_map = feature_map.cpu().numpy().astype(np.float16)
#             torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))


# def render_video(model_path, iteration, views, gaussians, pipeline, background, edit_config): ###
#     render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
#     makedirs(render_path, exist_ok=True)
#     view = views[0]
#     render_poses = render_path_spiral(views)

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     size = (view.original_image.shape[2], view.original_image.shape[1])
#     final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

#     if edit_config != "no editing":
#         edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)

#     for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
#         view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
#         view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
#         view.camera_center = view.world_view_transform.inverse()[3, :3]

#         if edit_config != "no editing":
#             rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) ###
#         else:
#             rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)

#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
#     final_video.release()



# def interpolate_matrices(start_matrix, end_matrix, steps):
#         # Generate interpolation factors
#         interpolation_factors = np.linspace(0, 1, steps)
#         # Interpolate between the matrices
#         interpolated_matrices = []
#         for factor in interpolation_factors:
#             interpolated_matrix = (1 - factor) * start_matrix + factor * end_matrix
#             interpolated_matrices.append(interpolated_matrix)
#         return np.array(interpolated_matrices)


# def multi_interpolate_matrices(matrix, num_interpolations):
#     interpolated_matrices = []
#     for i in range(matrix.shape[0] - 1):
#         start_matrix = matrix[i]
#         end_matrix = matrix[i + 1]
#         for j in range(num_interpolations):
#             t = (j + 1) / (num_interpolations + 1)
#             interpolated_matrix = (1 - t) * start_matrix + t * end_matrix
#             interpolated_matrices.append(interpolated_matrix)
#     return np.array(interpolated_matrices)


# ###
# def render_novel_views(model_path, name, iteration, views, gaussians, pipeline, background, 
#                        edit_config, speedup, multi_interpolate, num_views):
#     if multi_interpolate:
#         name = name + "_multi_interpolate"
#     # make dirs
#     if edit_config != "no editing":
#         edit_dict, text_feature, target = parse_edit_config_and_text_encoding(edit_config)
        
#         # edit
#         edit_render_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "renders")
#         edit_feature_map_path = os.path.join(model_path, name, "ours_{}_{}_{}".format(iteration, next(iter(edit_dict["operations"])), target), "feature_map")

#         makedirs(edit_render_path, exist_ok=True)
#         makedirs(edit_feature_map_path, exist_ok=True)
#     else:
#         # non-edit
#         render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#         feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
#         saved_feature_path = os.path.join(model_path, name, "ours_{}".format(iteration), "saved_feature")
#         #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
#         decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))

#         if speedup:
#             gt_feature_map = views[0].semantic_feature.cuda()
#             feature_out_dim = gt_feature_map.shape[0]
#             feature_in_dim = int(feature_out_dim/2)
#             cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
#             cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path))
        
#         makedirs(render_path, exist_ok=True)
#         makedirs(feature_map_path, exist_ok=True)
#         makedirs(saved_feature_path, exist_ok=True)

#     view = views[0]
    
#     # create novel poses
#     render_poses = []
#     for cam in views:
#         pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
#         render_poses.append(pose) 
#     if not multi_interpolate:
#         poses = interpolate_matrices(render_poses[0], render_poses[-1], num_views)
#     else:
#         poses = multi_interpolate_matrices(np.array(render_poses), 2)

#     # rendering process
#     for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
#         view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
#         view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
#         view.camera_center = view.world_view_transform.inverse()[3, :3]

#         if edit_config != "no editing":
#             render_pkg = render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)
#             gt = view.original_image[0:3, :, :]
#             gt_feature_map = view.semantic_feature.cuda()
#             torchvision.utils.save_image(render_pkg["render"], os.path.join(edit_render_path, '{0:05d}'.format(idx) + ".png")) 
#             # visualize feature map
#             feature_map = render_pkg["feature_map"] 
#             feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
#             if speedup:
#                 feature_map = cnn_decoder(feature_map)

#             feature_map_vis = feature_visualize_saving(feature_map)
#             Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(edit_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
#         else:
#             # mlp encoder
#             render_pkg = render(view, gaussians, pipeline, background) 

#             gt_feature_map = view.semantic_feature.cuda() 
#             torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
#             # visualize feature map
#             feature_map = render_pkg["feature_map"]
#             feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0) ###
#             if speedup:
#                 feature_map = cnn_decoder(feature_map)

#             feature_map_vis = feature_visualize_saving(feature_map)
#             Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

#             # save feature map
#             feature_map = feature_map.cpu().numpy().astype(np.float16)
#             torch.save(torch.tensor(feature_map).half(), os.path.join(saved_feature_path, '{0:05d}'.format(idx) + "_fmap_CxHxW.pt"))



# def render_novel_video(model_path, name, iteration, views, gaussians, pipeline, background, edit_config): 
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
#     makedirs(render_path, exist_ok=True)
#     view = views[0]
    
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     size = (view.original_image.shape[2], view.original_image.shape[1])
#     final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, 10, size)

#     if edit_config != "no editing":
#         edit_dict, text_feature = parse_edit_config_and_text_encoding(edit_config)
    
#     render_poses = [(cam.R, cam.T) for cam in views]
#     render_poses = []
#     for cam in views:
#         pose = np.concatenate([cam.R, cam.T.reshape(3, 1)], 1)
#         render_poses.append(pose)
    
#     # create novel poses
#     poses = interpolate_matrices(render_poses[0], render_poses[-1], 200) 

#     # rendering process
#     for idx, pose in enumerate(tqdm(poses, desc="Rendering progress")):
#         view.world_view_transform = torch.tensor(getWorld2View2(pose[:, :3], pose[:, 3], view.trans, view.scale)).transpose(0, 1).cuda()
#         view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
#         view.camera_center = view.world_view_transform.inverse()[3, :3]

#         if edit_config != "no editing":
#             rendering = torch.clamp(render_edit(view, gaussians, pipeline, background, text_feature, edit_dict)["render"], min=0., max=1.) 
#         else:
#             rendering = torch.clamp(render(view, gaussians, pipeline, background)["render"], min=0., max=1.)
        
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         final_video.write((rendering.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1])
#     final_video.release()


# def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, novel_view : bool, 
#                 video : bool , edit_config: str, novel_video : bool, multi_interpolate : bool, num_views : int): 
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree)
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

#         bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         if not skip_train:
#              render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)

#         if not skip_test:
#              render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)

#         if novel_view:
#              render_novel_views(dataset.model_path, "novel_views", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, 
#                                 edit_config, dataset.speedup, multi_interpolate, num_views)

#         if video:
#              render_video(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config)

#         if novel_video:
#              render_novel_video(dataset.model_path, "novel_views_video", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, edit_config, dataset.speedup)

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--novel_view", action="store_true") ###
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--video", action="store_true") ###
#     parser.add_argument("--novel_video", action="store_true") ###
#     parser.add_argument('--edit_config', default="no editing", type=str)
#     parser.add_argument("--multi_interpolate", action="store_true") ###
#     parser.add_argument("--num_views", default=200, type=int)
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.novel_view, 
#                 args.video, args.edit_config, args.novel_video, args.multi_interpolate, args.num_views) ###




# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2

import os

cpu_list = [0]
os.sched_setaffinity(0, cpu_list)



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, no_fine, render_test=False, reconstruct=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    gtdepth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    masks_path = os.path.join(model_path, name, "ours_{}".format(iteration), "masks")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(gtdepth_path, exist_ok=True)
    makedirs(masks_path, exist_ok=True)
    
    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        rendering = render(view, gaussians, pipeline, background, stage=stage)
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            mask = view.mask
            mask_list.append(mask)
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)
    
    if render_test:
        test_times = 50
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering = render(view, gaussians, pipeline, background, stage=stage)
        time2=time()
        print("FPS:",(len(views)-1)*test_times/(time2-time1))
    
    count = 0
    print("writing training images.")
    if len(gt_list) != 0:
        for image in tqdm(gt_list):
            torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
            count+=1
            
    count = 0
    print("writing rendering images.")
    if len(render_images) != 0:
        for image in tqdm(render_images):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing mask images.")
    if len(mask_list) != 0:
        for image in tqdm(mask_list):
            image = image.float()
            torchvision.utils.save_image(image, os.path.join(masks_path, '{0:05d}'.format(count) + ".png"))
            count +=1
    
    count = 0
    print("writing rendered depth images.")
    if len(render_depths) != 0:
        for image in tqdm(render_depths):
            image = np.clip(image.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
    
    count = 0
    print("writing gt depth images.")
    if len(gt_depths) != 0:
        for image in tqdm(gt_depths):
            image = image.cpu().squeeze().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(gtdepth_path, '{0:05d}'.format(count) + ".png"), image)
            count += 1
            
    render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
    render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'ours_video.mp4'), render_array, fps=30, quality=8)
    
    gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
    gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=30, quality=8)
                    
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    
    if reconstruct:
        reconstruct_point_cloud(render_images, mask_list, render_depths, camera_parameters, name)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, reconstruct: bool):
    
    cpu_list = list(range(cpu_count))[1:2]
    psutil.Process().cpu_affinity(cpu_list)
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset, dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_coarse=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.no_fine, reconstruct=False)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.no_fine, reconstruct=reconstruct)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),gaussians,pipeline,background, dataset.no_fine, render_test=True, reconstruct=False)

def reconstruct_point_cloud(images, masks, depths, camera_parameters, name):
    import cv2
    import copy
    output_frame_folder = os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    # frames = [0]
    focal_x, focal_y, width, height = camera_parameters
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()
        depth_np = depth_np.squeeze(0)
        mask = masks[i_frame]
        mask = mask.squeeze(0).cpu().numpy()
        
        rgb_new = copy.deepcopy(rgb_np)

        depth_smoother = (128, 64, 64)
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
        
        close_depth = np.percentile(depth_np[depth_np!=0], 5)
        inf_depth = np.percentile(depth_np, 95)
        depth_np = np.clip(depth_np, close_depth, inf_depth)

        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, focal_x, focal_y, width / 2, width / 2),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)

if __name__ == "__main__":
    import psutil

    cpu_count = psutil.cpu_count()
    print(cpu_count)

    cpu_list = list(range(cpu_count))[1:2]

    psutil.Process().cpu_affinity(cpu_list)

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--reconstruct", action="store_true")
    args = get_combined_args(parser)
    print("Rendering ", args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.reconstruct)