import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image
import os

def get_from_camera(demo, seq):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()


    depth_sensor.set_option(rs.option.exposure, 4000) #12000
    depth_sensor.set_option(rs.option.depth_units, 0.0001)
    print("Depth Scale is: " , depth_scale)

    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()

    # # depth_sensor.set_option(rs.option.exposure, 4000) #12000
    # depth_sensor.set_option(rs.option.depth_units, 0.0001)
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
    spatial.set_option(rs.option.filter_smooth_delta, 1)
    spatial.set_option(rs.option.holes_fill, 1)
    temporal = rs.temporal_filter()
    # temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
    # temporal.set_option(rs.option.filter_smooth_delta, 1)
    # print("Depth Scale is: " , depth_scale)

    # Adjust exposure of rgb/color sensor
    color_sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # color_sensor.set_option(rs.option.exposure, 200) # for block
    color_sensor.set_option(rs.option.exposure, 50) # 200
    #color_sensor.set_option(rs.option.exposure, 150)
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #2 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    for i in range(30):
        pipeline.wait_for_frames() # wait for autoexposure to settle

    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get the point cloud
    cloud_path = 'kitchen/real' + str(demo) + '/pcd'+str(seq)+'.ply'
    for i in range(1):
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame:
            continue
        depth_frame = depth_to_disparity.process(depth_frame)
        filtered_depth = spatial.process(depth_frame)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = disparity_to_depth.process(filtered_depth)
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        cloud = pc.calculate(filtered_depth)
        if i == 0 and os.path.exists(cloud_path):
            os.remove(cloud_path)
        cloud.export_to_ply(cloud_path, color_frame)
        print(cloud)

    # # Get aligned frames
    # aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    # depth_frame = depth_to_disparity.process(aligned_depth_frame)
    # filtered_depth = spatial.process(depth_frame)
    # filtered_depth = temporal.process(filtered_depth)
    # filtered_depth = disparity_to_depth.process(filtered_depth)
    # color_frame = aligned_frames.get_color_frame()

    # depth_image = np.asanyarray(filtered_depth.get_data())
    # color_image = np.asanyarray(color_frame.get_data())

    # import ipdb; ipdb.set_trace()

    #np.save('/media/HDD/yuying/cliport-local/real_data/depth_'+str(seq)+'.npz', depth_image)
    #np.save('/media/HDD/yuying/cliport-local/real_data/color_'+str(seq)+'.npz', color_image)

    pipeline.stop()

    return depth_image, color_image, cloud


def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.

    Args:
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    # import ipdb; ipdb.set_trace()
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points


# def get_pointcloud(depth_image: np.ndarray, intrinsic: np.ndarray, offset=0):
#     """Unproject a depth image to a 3D poin cloud.
#     Args:
#         depth_image: [H, W]
#         intrinsic: [3, 3]
#         offset: offset of x and y indices.
#     Returns:
#         points: [H, W, 3]
#     """
#     v, u = np.indices(depth_image.shape)  # [H, W], [H, W]
#     z = depth_image  # [H, W]
#     uv1 = np.stack([u + offset, v + offset, np.ones_like(z)], axis=-1)
#     points = uv1 @ np.linalg.inv(intrinsic).T * z[..., None]
#     return points
#
#
# def get_intrinsic_matrix(width, height, fov, degree=False):
#     """Get the camera intrinsic matrix according to image size and fov."""
#     if degree:
#         fov = np.deg2rad(fov)
#     f = (width / 2.0) / np.tan(fov / 2.0)
#     xc = (width - 1.0) / 2.0
#     yc = (height - 1.0) / 2.0
#     K = np.array([[f, 0, xc], [0, f, yc], [0, 0, 1.0]])
#     return K


def get_heightmap(points, colors, bounds, width, height):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
      points: HxWx3 float array of 3D points in world coordinates.
      colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
      bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
        region in 3D space to generate heightmap in world coordinates.
      pixel_size: float defining size of each pixel in meters.

    Returns:
      heightmap: HxW float array of height (from lower z-bound) in meters.
      colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    pixel_size = (bounds[0, 1] - bounds[0, 0]) / width
    #pixel_size2 = (bounds[1, 1] - bounds[1, 0]) / height
    #print(pixel_size, pixel_size2)
    #height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # print(points.reshape(-1, 3).max(0) - points.reshape(-1, 3).min(0))
    # import trimesh
    # pcd = trimesh.PointCloud(points.reshape(-1, 3))
    # pcd.show()
    # pcd.export("test.obj")
    # exit(0)


    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]
    # import ipdb; ipdb.set_trace()



    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[py, px, c] = colors[:, c]
    return heightmap, colormap, pixel_size


def get_input(demo, seq):
    # import open3d as o3d
    # import pyrealsense2 as rs
    # import numpy as np
    #
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # profile = pipeline.start(config)
    # for x in range(5):
    #     pipeline.wait_for_frames()
    # frames = pipeline.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    # pc = rs.pointcloud()
    # pc.map_to(color_frame)
    # xyz = pc.calculate(depth_frame)
    depth, color, pointcloud = get_from_camera(demo, seq)

    #color = np.load('real_data/color_0.npz.npy')
    #color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #depth = np.load('real_data/depth_0.npz.npy') / 10000
    depth = depth / 10000

    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite('kitchen/real' + str(demo) + '/color'+str(seq)+'.jpg', color)
    np.save('kitchen/real' + str(demo) + '/color'+str(seq)+'.npz', depth)
    #cv2.imwrite('real_data_kitchen_new2/depth'+str(seq)+'.jpg', depth)
    # print(color.shape, depth.shape)
    # # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    plt.imshow(color)
    #plt.savefig('real_data_kitchen/color'+str(seq)+'.jpg')
    plt.show()
    plt.imshow(depth)
    #plt.savefig('real_data_kitchen/depth'+str(seq)+'.jpg')
    plt.show()




    #print(color.max(), color.min(), depth.max(), depth.min())
    #intrinsics = [380.7734680175781, 0, 324.0291748046875, 0, 380.7734680175781, 240.8050994873047, 0, 0, 1]
    intrinsics = [612.7603149414062, 0, 322.2873229980469, 0, 613.31005859375, 234.67587280273438, 0, 0, 1]
    intrinsics = np.array(intrinsics).reshape(3, 3)
    print('intrinsics:', intrinsics)

    xyz = get_pointcloud(depth, intrinsics)
    print(xyz.shape)
    print(xyz[:,:,0].max(), xyz[:,:,0].min(), xyz[:,:,1].max(), xyz[:,:,1].min(),xyz[:,:,2].max(), xyz[:,:,2].min(),)

    coord = xyz[330, 338].tolist()
    coord.append(1)
    coord = np.array(coord).reshape(-1, 1)
    print(coord)

    # desk2camera = [[0.999272567712139, 0.003400535682543303, -0.037983835707577705, -0.056222465507518395], [0.03790715819775544, 0.020321188107669558, 0.9990746201717181, 0.2716694192697733], [0.004169265565881826, -0.9997877203041475, 0.02017750117523706, 0.8531071020329748], [0.0, 0.0, 0.0, 1.0]]

    # # RECALIBRATE
    # adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # adjust_pos_mat = np.array([[1, 0, 0, -0.2], [0, 1, 0, -0.075], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust 

    desk2camera = [[0.9995041445583243, 0.015233611744522804, 0.027557251023093778, -0.06908457188626575], [-0.02793638244421772, 0.025262065250222506, 0.9992904415610229, 0.2559835936616738], [0.014526649533291223, -0.9995647878614526, 0.025675110921401928, 0.8168659375022858], [0.0, 0.0, 0.0, 1.0]]
    # RECALIBRATE
    adjust_ori_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    adjust_pos_mat = np.array([[1, 0, 0, -0.2], [0, 1, 0, -0.06], [0, 0, 1, 0], [0, 0, 0, 1]]) # manually adjust 

    base2camera = desk2camera@adjust_ori_mat@adjust_pos_mat
    cam2base = np.linalg.inv(base2camera)
    print('cam2base:', cam2base)

    # coord_real = np.dot(cam2base, coord)[:3] * 1000
    # print(coord_real)
    # height = 160

    # # BOUNDS FOR NEW CAMERA CONFIG
    # # bounds = [-0.25, 0.27, -0.15, 0.16, 0.6, 0.7]
    # # bounds = [-0.3, 0.34, -0.20, 0.12, 0.6, 0.7]
    # #bounds = [-0.3, 0.34, -0.20, 0.12, 0.6, 0.7] # new camera config bounds for shape bowl
    # bounds = [-0.21, 0.29, -0.16, 0.09, 0.6, 0.7]  # new camera config bounds for shape bowl
    # # bounds = [-0.20, 0.30, -0.15, 0.10, 0.6, 0.7]
    # #bounds = [-0.3, 0.32, -0.18, 0.13, 0.0, 0.7]  # new camera config bounds
    # # bounds = [-0.28, 0.30, -0.17, 0.12, 0.6, 0.7]  # new camera config bounds for shape box

    # # BOUNDS FOR OLD CAMERA CONFIG
    # # bounds = [-0.29, 0.27, -0.20, 0.08, 0, 0.6] # most recent for old camera config
    # #bounds = [-0.5, 0.5, -0.5, 0, 0, 1]
    # #bounds = [-0.3, 0.5, -0.5, 0.5, 0, 1]
    # bounds = np.array(bounds).reshape(3, 2)
    # heightmap, colormap, pixel_size = get_heightmap(xyz, color, bounds, width, height)
    # heightmap = np.float32(heightmap)
    # colormap = np.uint8(np.round(colormap))

    # rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('method/color6.jpg', rgb)

    # np.save('/media/HDD/yuying/cliport-local/method/hmap_'+str(6)+'.npz', heightmap)
    # np.save('/media/HDD/yuying/cliport-local/method/cmap_'+str(6)+'.npz', colormap)

    # #heightmap = np.float32(heightmap).transpose(1,0)
    # #colormap = np.uint8(np.round(colormap)).transpose(1,0,2)

    # plt.imshow(colormap)
    # #plt.savefig('real_data_shape/colormap'+str(seq)+'.jpg')
    # plt.show()
    # plt.close()
    # plt.imshow(heightmap)
    # #plt.savefig('real_data_shape/depthmap'+str(seq)+'.jpg')
    # plt.show()
    # plt.close()

    # return colormap, heightmap, bounds, pixel_size
    # plt.imshow(colormap)
    # plt.show()
    # plt.imshow(heightmap)
    # plt.show()

if __name__ == '__main__':
    # use this to get rgb and pointcloud, the default folder is "kitchen"
    demo = 25
    seq = 3
    get_input(demo, seq)