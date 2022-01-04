# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/29/21 11:07 AM
"""

import open3d as o3
import numpy as np
from scipy.spatial.kdtree import KDTree
import copy


def main():
    vis = o3.visualization.Visualizer()
    # vis.create_window()

    pc_file_path = '/home/cheng/proj/3d/TEASER-plusplus/data/human_models/head_models/model_man/3D_model_face_from_mr.pcd'
    # pc_gt_file_path = '/home/cheng/proj/3d/TEASER-plusplus/data/human_models/head_models/model_man/3D_model_from_mr.pcd'
    pc_original = o3.io.read_point_cloud(pc_file_path)
    # pc_gt = o3.io.read_point_cloud(pc_gt_file_path)

    '''init pc'''
    pc = pc_original.voxel_down_sample(1)
    # pc_show = pc_original.voxel_down_sample(20)
    # pc_gt = pc_gt.voxel_down_sample(2)

    '''compute'''
    center = pc.get_center()

    pc_down = get_rid_of_layer_loop_through_layers(pc)

    axis_pcd = o3.geometry.TriangleMesh()
    axis_pcd = axis_pcd.create_coordinate_frame(size=50, origin=center)

    o3.visualization.draw_geometries([pc, axis_pcd])
    o3.visualization.draw_geometries([pc_down, axis_pcd])

    return 0


def sphere_test():
    mesh = o3.geometry.TriangleMesh()
    pc_1 = o3.geometry.PointCloud()
    pc_2 = o3.geometry.PointCloud()

    sphere1 = mesh.create_sphere(radius=30, resolution=90)
    sphere2 = mesh.create_sphere(radius=10, resolution=30)
    axe = mesh.create_coordinate_frame(size=5)

    pc_1.points = sphere1.vertices
    # pc_1.points = pc_1.points[int(len(pc_1.points) * 2 / 3):]

    pc_2.points = sphere2.vertices
    center = pc_2.get_center()

    # pc_2.points = pc_2.points[0: int(len(pc_2.points))]

    pc = pc_1 + pc_2
    # pc_down = get_rid_of_inner_loop_through_dot_product(pc, center)
    pc_down = get_rid_of_layer_loop_through_layers(pc)

    # o3.visualization.draw_geometries([pc_1, pc_2])
    o3.visualization.draw_geometries([axe, pc])
    o3.visualization.draw_geometries([axe, pc_down])


def get_rid_of_layer_loop_through_layers(pc):
    layer_axis = 2
    layer_step = 20

    center = pc.get_center()
    '''cut off part below neck'''
    max_bound, min_bound = o3.geometry.PointCloud.get_max_bound(pc), o3.geometry.PointCloud.get_min_bound(pc)
    max_bound[-1] -= 50
    bounding_box_layer = o3.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pc = o3.geometry.PointCloud.crop(pc, bounding_box_layer)

    '''normalized and scale head toward sphere shape'''
    sphere_radius = 100.0
    points = np.array(pc.points)
    points -= center

    ratio = sphere_radius / (max_bound - center)
    scale_matrix = np.eye(3)
    scale_matrix[0, 0] = ratio[0]
    scale_matrix[1, 1] = ratio[1]
    scale_matrix[2, 2] = ratio[2]
    points = np.matmul(points, scale_matrix)

    # points += center
    pc.points = o3.utility.Vector3dVector(points)

    # new range
    max_bound, min_bound = o3.geometry.PointCloud.get_max_bound(pc), o3.geometry.PointCloud.get_min_bound(pc)

    # vis
    o3.visualization.draw_geometries([pc])

    pc_return = o3.geometry.PointCloud()

    '''filter points inside layer-wise'''
    for slice_start_in_layer_axis in np.arange(min_bound[layer_axis], max_bound[layer_axis], layer_step):
        '''crop layer point cloud'''
        max_bound_layer, min_bound_layer = copy.deepcopy(max_bound), copy.deepcopy(min_bound)
        max_bound_layer[layer_axis] = slice_start_in_layer_axis + layer_step
        min_bound_layer[layer_axis] = slice_start_in_layer_axis
        bounding_box_layer = o3.geometry.AxisAlignedBoundingBox(min_bound_layer, max_bound_layer)
        layer = o3.geometry.PointCloud.crop(pc, bounding_box_layer)

        # o3.visualization.draw_geometries([layer])

        '''layer points '''
        pc_return += get_rid_of_inner_all_radius(layer, center=(0, 0, 0))

    print('total ', len(np.array(pc.points)), ' in input')
    print('total ', len(np.array(pc_return.points)), ' in output')
    return pc_return


# def get_rid_of_inner_loop_through_dot_product(pc, center=None):
#     num_points_in_loop = 1000
#     max_distance_to_from_outter_layer = 0.02
#     max_dis_of_dot_product_vector_inline = 0.00008
#
#     points = np.asarray(pc.points)  # (N, 3)
#     np.random.shuffle(points)
#     points_porj_xyplane = copy.deepcopy(points)
#     points_porj_xyplane[:, -1] = 0.0
#     mask_testing = np.ones(points_porj_xyplane.shape[0]).astype(bool)  # (N)
#     # print(len(points_porj_xyplane), 'points in count')
#
#     pc_return = o3.geometry.PointCloud()
#     mask_parallel_and_inside = np.zeros(points_porj_xyplane.shape[0]).astype(bool)
#     # mask_out = np.zeros(points.shape[0]).astype(bool)                               # (N)
#
#     if points_porj_xyplane.shape[0] <= 5:
#         return pc_return
#
#     '''normalize all points: move to center, and rescale to unit length'''
#     if center is None:
#         center = points_porj_xyplane.mean(axis=0)
#     norms = np.expand_dims(np.linalg.norm(points_porj_xyplane - center, axis=-1), axis=-1)
#     points_normalized = (points_porj_xyplane - center) / norms  # (N, 3)
#     # index = dot < 0.1
#
#     '''dot product of all points, points in similar direction has dot product close to one,
#     variant direction has dot product close to zero '''
#
#
#     num_iter = int(np.ceil(points_normalized.shape[0] / num_points_in_loop))
#
#     for i in range(num_iter):
#         # loop start from points not tested yet, ignore points that already tested parallel and inside
#         index_batch_start = i * num_points_in_loop
#         # mask_parallel_and_inside_current = mask_parallel_and_inside[index_start:]
#         points_testing = points_normalized #[np.logical_not(mask_parallel_and_inside_current)]
#         norms_testing = norms #[np.logical_not(mask_parallel_and_inside_current)]
#
#         # index_batch_end = max((i + 1) * num_points_in_loop, points_testing.shape[0] - 1)
#         # update loop: if loop overed all testing points, break
#         if i >= len(points_testing):
#             break
#
#         #
#         mask_parallel_and_inside_batch = mask_parallel_and_inside[index_batch_start: min(index_batch_start+num_points_in_loop, len(points_testing))]
#         points_batch = points_testing[index_batch_start: min(index_batch_start+num_points_in_loop, len(points_testing))]
#         norms_batch = norms_testing[index_batch_start: min(index_batch_start+num_points_in_loop, len(points_testing))]#[np.logical_not(mask_parallel_and_inside_current)]
#
#         dot = np.dot(points_batch, points_testing.T)
#         mask_parallel = np.logical_and(1.0 - max_dis_of_dot_product_vector_inline < dot, dot < 1.0 + max_dis_of_dot_product_vector_inline)  # (N_male, N_female)
#
#         '''mark points in same direction but most distant from center(having biggest norm or close to)'''
#         for k in range(mask_parallel.shape[0]):
#             mask_parallel_ele = mask_parallel[k]  # (N)
#             norms_parallel_ele = norms_testing[mask_parallel_ele]  # (n_parallel)
#             # no maximum filtering, ignore point far away from points in outer surface
#             if norms_parallel_ele.shape[0] > 0:
#                 if norms_batch[k] < norms_parallel_ele.max() - max_distance_to_from_outter_layer or \
#                         norms_batch[k] > norms_parallel_ele.max() + max_distance_to_from_outter_layer:
#                     mask_parallel_and_inside_batch[k] = True
#
#     # dot = np.dot(points_normalized, points_normalized.T)                            # (N, N)
#     # dot.diagonal -= 1                                                               # (N, N), ignore point itself
#     # # print(index.shape)
#     #
#     # '''mark points that has similar direction'''
#     # mask = np.logical_and(dot < 1.0 + max_dis_of_dot_product_vector_inline, dot > 1.0 - max_dis_of_dot_product_vector_inline)                                   # (
#     #
#     # '''mark points in same direction but most distant from center(having biggest norm or close to)'''
#     # for i in range(points.shape[0]):
#     #     neighbors_mask = mask[i]
#     #     neighbors_norm = norms[neighbors_mask]
#     #     # no maximum filtering, ignore point far away from points in outer surface
#     #     if neighbors_norm.shape[0] > 0:
#     #         if neighbors_norm.max() - max_distance_to_from_outter_layer < norms[i] < neighbors_norm.max() + max_distance_to_from_outter_layer:
#     #             mask_out[i] = True
#
#     points_out = o3.utility.Vector3dVector(points[np.logical_not(mask_parallel_and_inside)])
#     # print(len(points_out), 'points in count')
#
#
#     pc_return.points = points_out
#
#     # print('total comb', points.shape[0] * points.shape[0])
#     # print('comb 0', len(np.where(index)[0]))
#     # print('idnex 0', np.where(index))
#
#     # print(p_1.shape, p_2.shape)
#     # c = np.cross(p_1, p_2)
#     # d = np.dot(p_1, p_2.T)
#     return pc_return


# def get_rid_of_inner_all_dot_product(pc, center=None):
#     max_distance_to_from_outter_layer = 2
#     max_dis_of_dot_product_vector_inline = 0.000001
#
#     pc_return = o3.geometry.PointCloud()
#     points = np.asarray(pc.points)                                                  # (N, 3)
#
#     points_porj_xyplane = copy.deepcopy(points)
#     points_porj_xyplane[:, -1] = 0.0
#     # np.random.shuffle(points_porj_xyplane)
#
#     mask_out = np.zeros(points_porj_xyplane.shape[0]).astype(bool)                               # (N)
#
#     if points_porj_xyplane.shape[0] < 5:
#         return pc_return
#
#     '''normalize all points: move to center, and rescale to unit length'''
#     if center is None:
#         center = points_porj_xyplane.mean(axis=0)
#     norms = np.expand_dims(np.linalg.norm(points_porj_xyplane - center, axis=-1), axis=-1)
#     points_normalized = (points_porj_xyplane - center) / norms                                   # (N, 3)
#     # index = dot < 0.1
#
#     '''dot product of all points, points in similar direction has dot product close to one,
#     variant direction has dot product close to zero '''
#     dot = np.dot(points_normalized, points_normalized.T)                            # (N, N)
#     dot -= np.eye(dot.shape[0])                                                     # (N, N), ignore point itself
#     # print(index.shape)
#
#     '''mark points that has similar direction'''
#     mask = dot > 1.0 - max_dis_of_dot_product_vector_inline                                   # (
#
#     '''mark points in same direction but most distant from center(having biggest norm or close to)'''
#     for i in range(points_porj_xyplane.shape[0]):
#         neighbors_mask = mask[i]
#         neighbors_norm = norms[neighbors_mask]
#         # no maximum filtering, ignore point far away from points in outer surface
#         if neighbors_norm.shape[0] > 0:
#             if neighbors_norm.max() - max_distance_to_from_outter_layer < norms[i]:
#                 mask_out[i] = True
#
#     points_out = o3.utility.Vector3dVector(points[mask_out])
#
#     # for i in range(points.shape[0]):
#     #     if i > 10:
#     #         break
#     #     dot_i = dot[i, :]
#     #     id_ones = np.logical_and(dot_i < 1.2, dot_i > 0.8)
#     #
#     #     '''add points'''
#     #     points_out = points[i, id_ones]
#     #     print(dot_i)
#     #     print(np.where(id_ones)[0].shape)
#
#     pc_return.points = points_out
#
#     # axis_pcd = o3.geometry.TriangleMesh()
#     # axis_pcd = axis_pcd.create_coordinate_frame(size=5, origin=center)
#     # o3.visualization.draw_geometries([pc, axis_pcd])
#     # o3.visualization.draw_geometries([pc_return])
#
#     return pc_return

def get_rid_of_inner_all_radius(pc, center=None):
    radius_shrink_factor = 0.9
    max_distance_to_from_outter_layer = 5.0

    pc_return = o3.geometry.PointCloud()
    points = np.asarray(pc.points)  # (N, 3)

    points_porj_xyplane = copy.deepcopy(points)
    # points_porj_xyplane[:, -1] = 0.0
    # np.random.shuffle(points_porj_xyplane)

    mask_out = np.zeros(points_porj_xyplane.shape[0]).astype(bool)  # (N)

    if points_porj_xyplane.shape[0] < 5:
        return pc_return

    '''normalize all points: move to center, and rescale to unit length'''
    if center is None:
        center = points_porj_xyplane.mean(axis=0)
    norms = np.expand_dims(np.linalg.norm(points_porj_xyplane - center, axis=-1), axis=-1)

    '''dot product of all points, points in similar direction has dot product close to one,
    variant direction has dot product close to zero '''
    radius = norms.mean()  # (N, N)
    # print(index.shape)

    '''mark points that has similar direction'''
    mask_out = norms > radius_shrink_factor * radius - max_distance_to_from_outter_layer  # (
    mask_out = mask_out.squeeze()
    '''mark points in same direction but most distant from center(having biggest norm or close to)'''

    points_out = o3.utility.Vector3dVector(points[mask_out])

    pc_return.points = points_out

    print(len(points), 'points in ')
    print(len(points_out), ' points out ')

    # debug visual
    axis_pcd = o3.geometry.TriangleMesh()
    axis = axis_pcd.create_coordinate_frame(size=10, origin=center)
    sphere_filter = axis_pcd.create_sphere(radius=radius, resolution=1000)
    sphere_filter.translate(center)
    sphere_filter = sphere_filter.sample_points_uniformly(number_of_points=1000)

    # o3.visualization.draw_geometries([pc, sphere_filter, axis])
    # o3.visualization.draw_geometries([pc_return])

    return pc_return


if __name__ == '__main__':
    main()
    # sphere_test()
    # a = np.arange(0, 12).reshape((3, -1))
    # b = np.array([True, False, True])
    # c = a[b]
    # print(c)
