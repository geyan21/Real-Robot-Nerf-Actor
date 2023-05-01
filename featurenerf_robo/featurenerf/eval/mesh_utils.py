import argparse
import os
import numpy as np
import torch
# import models

from importlib import import_module
from pytorch3d.structures import Meshes
from skimage import measure
from nerf_helpers import export_obj, batchify
import mcubes
# from lightning_modules import PathParser


def create_mesh(vertices, faces_idx):
    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    vertices = vertices - vertices.mean(0)
    scale = max(vertices.abs().max(0)[0])
    vertices = vertices / scale

    # We construct a Meshes structure for the target mesh
    target_mesh = Meshes(verts=[vertices], faces=[faces_idx])

    return target_mesh


def extract_radiance(model, args, device, nums):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert (isinstance(nums, tuple) or isinstance(nums, list) or isinstance(nums, int)), \
        "Nums arg should be either iterable or int."

    if isinstance(nums, int):
        nums = (nums,) * 3
    else:
        assert (len(nums) == 3), "Nums arg should be of length 3, number of axes for 3D"

    # Create sample tiles
    """
    x: -0.3103, 3.0554
    y: -2.8981, 0.3607
    z: -1.8048, 1.3243

    """

    # create a grid of points
    x = torch.linspace(-0.3103, 3.0554, nums[0], device=device)
    y = torch.linspace(-2.8981, 0.3607, nums[1], device=device)
    z = torch.linspace(-1.8048, 1.3243, nums[2], device=device)
    tiles = [x, y, z]

    # tiles = [torch.linspace(-args.limit, args.limit, num) for num in nums]

    # Generate 3D samples
    samples = torch.stack(torch.meshgrid(*tiles), -1).view(-1, 3).float()

    radiance_samples = []
    for (samples,) in batchify(samples, batch_size=args.batch_size, device=device):
        # Query radiance batch
        samples = samples.unsqueeze(0)
        viewdirs = torch.zeros_like(samples)
        radiance_batch = model(samples, coarse=True, viewdirs=viewdirs, ret_last_feat=False)
        # get rgb and density
        radiance_batch = radiance_batch[..., :4].squeeze(0)
        # Accumulate radiance
        radiance_samples.append(radiance_batch.cpu())

    # Radiance 3D grid (rgb + density)
    radiance = torch.cat(radiance_samples, 0).view(*nums, 4).contiguous().numpy()

    return radiance


def extract_iso_level(density, args):
    # Density boundaries
    min_a, max_a, std_a = density.min(), density.max(), density.std()

    # Adaptive iso level
    iso_value = min(max(args.iso_level, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    return iso_value


def extract_geometry(model, device, args):
    # Sample points based on the grid
    radiance = extract_radiance(model, args, device, args.res)

    # Density grid
    density = radiance[..., 3]

    # Adaptive iso level
    iso_value = extract_iso_level(density, args)
    
    # threshold = 50.
    # print('fraction occupied', np.mean(density > threshold))

    # Extracting iso-surface triangulated
    results = measure.marching_cubes(density, iso_value)
    # results = measure.marching_cubes(density, threshold)

    # Use contiguous tensors
    vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

    # Use contiguous tensors
    normals = torch.from_numpy(np.ascontiguousarray(normals))
    vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))

    # Normalize vertices, to the (-limit, limit)
    vertices = args.limit * (vertices / (args.res / 2.) - 1.)

    return vertices, triangles, normals, density


def extract_radiance_voxel(model, voxel, args, device, nums):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert (isinstance(nums, tuple) or isinstance(nums, list) or isinstance(nums, int)), \
        "Nums arg should be either iterable or int."

    if isinstance(nums, int):
        nums = (nums,) * 3
    else:
        assert (len(nums) == 3), "Nums arg should be of length 3, number of axes for 3D"

    

    samples = voxel

    radiance_samples = []
    for (samples,) in batchify(samples, batch_size=args.batch_size, device=device):
        # Query radiance batch
        samples = samples.unsqueeze(0)
        viewdirs = torch.zeros_like(samples)
        radiance_batch = model(samples, coarse=True, viewdirs=viewdirs, ret_last_feat=False)
        # get rgb and density
        radiance_batch = radiance_batch[..., :4].squeeze(0)
        # Accumulate radiance
        radiance_samples.append(radiance_batch.cpu())

    # Radiance 3D grid (rgb + density)
    radiance = torch.cat(radiance_samples, 0).view(*nums, 4).contiguous().numpy()

    return radiance


def export_marching_cubes(model, renderer, args):
    # Mesh Extraction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.super_sampling >= 1:
        print("Generating mesh geometry...")

        # Extract model geometry with super sampling across each axis
        extract_geometry_with_super_sampling(model, device, args)
        return

    # Cached mesh path containing data
    mesh_cache_path = os.path.join(args.save_dir, args.cache_name)

    cached_mesh_exists = os.path.exists(mesh_cache_path)
    cache_new_mesh = args.use_cached_mesh and not cached_mesh_exists
    if cache_new_mesh:
        print(f"Cached mesh does not exist - {mesh_cache_path}")

    if args.use_cached_mesh and cached_mesh_exists:
        print("Loading cached mesh geometry...")
        vertices, triangles, normals, density = torch.load(mesh_cache_path)
    else:
        print("Generating mesh geometry...")
        # Extract model geometry
        vertices, triangles, normals, density = extract_geometry(model, device, args)

        if cache_new_mesh or args.override_cache_mesh:
            torch.save((vertices, triangles, normals, density), mesh_cache_path)
            print(f"Cached mesh geometry saved to {mesh_cache_path}")

    # Extracting the mesh appearance

    # Ray targets and directions
    targets, directions = vertices, -normals

    diffuse = []
    if args.no_view_dependence:
        print("Diffuse map query directly  without specific-views...")
        # Query directly without specific-views
        batch_generator = batchify(targets, directions, batch_size=args.batch_size, device=device)
        for (pos_batch, dir_batch) in batch_generator:
            # Diffuse batch queried
            diffuse_batch = model.sample_points(pos_batch, dir_batch)[..., :3]
            # rays = torch.cat([pos_batch, dir_batch], -1)
            # diffuse_batch = renderer(rays)

            # Accumulate diffuse
            diffuse.append(diffuse_batch.cpu())
    else:
        print("Diffuse map query with view dependence...")
        near, far = 0.1, 4.0

        # Query with view dependence
        # Move ray origins slightly towards negative sdf
        ray_origins = targets - args.view_disparity * directions

        print("Started ray-casting")
        batch_generator = batchify(ray_origins, directions, batch_size=args.batch_size, device=device)
        for (ray_origins, ray_directions) in batch_generator:
            # View dependent diffuse batch queried
            ray_bounds = torch.stack([near * torch.ones_like(ray_origins[..., :1]), far * torch.ones_like(ray_origins[..., :1])], -1).squeeze(1)
            rays = torch.cat([ray_origins, ray_directions, ray_bounds], -1).unsqueeze(0)
            output_bundle = renderer(rays)  

            rgb_map = output_bundle['fine']['rgb'].squeeze(0)

            # Accumulate diffuse
            diffuse.append(rgb_map.cpu())

    # Query the whole diffuse map
    diffuse = torch.cat(diffuse, dim=0).numpy()

    # Target mesh path
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    mesh_path = os.path.join(args.save_dir, args.mesh_name)

    # Export model
    export_obj(vertices, triangles, diffuse, normals, mesh_path)

def mesh_parser(parser):
    parser.add_argument(
        "--save-dir", type=str, default=".",
        help="Save mesh to this directory, if specified.",
    )
    parser.add_argument(
        "--mesh-name", type=str, default="mesh.obj",
        help="Mesh name to be generated.",
    )
    parser.add_argument(
        "--iso-level", type=float, default=32,
        help="Iso-level value for triangulation",
    )
    parser.add_argument(
        "--limit", type=float, default=1.2,
        help="Limits in -xyz to xyz for marching cubes 3D grid.",
    )
    parser.add_argument(
        "--res", type=int, default=128,
        help="Sampling resolution for marching cubes, increase it for higher level of detail.",
    )
    parser.add_argument(
        "--super-sampling", type=int, default=0,
        help="Add super sampling along the edges.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096,
        help="Higher batch size results in faster processing but needs more device memory.",
    )
    parser.add_argument(
        "--no-view-dependence", action="store_true", default=False,
        help="Disable view dependent appearance, use sampled diffuse color based on the grid"
    )
    parser.add_argument(
        "--view-disparity", type=float, default=1e-2,
        help="Ray origins offset from target based on the inverse normal for the view dependent appearance.",
    )
    parser.add_argument(
        "--view-disparity-max-bound", type=float, default=4e0,
        help="Far max possible bound, usually set to (cfg.far - cfg.near), lower it for better "
             "appearance estimation when using higher resolution e.g. at least view_disparity * 2.0.",
    )
    parser.add_argument(
        "--use-cached-mesh", action="store_true", default=False,
        help="Use the cached mesh.",
    )
    parser.add_argument(
        "--override-cache-mesh", action="store_true", default=False,
        help="Caches the mesh, useful for rapid configuration appearance tweaking.",
    )
    parser.add_argument(
        "--cache-name", type=str, default="mesh_cache.pt",
        help="Mesh cache name, allows for multiple unique meshes of different resolutions.",
    )
    

    return parser

if __name__ == "__main__":
    
    config_args = mesh_parser().parse_args()

    # Existent log path
    path_parser = PathParser()
    cfg, _ = path_parser.parse(None, config_args.log_checkpoint, None, config_args.checkpoint)

    # Available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model checkpoint
    print(f"Loading model from {path_parser.checkpoint_path}")
    model = getattr(models, cfg.experiment.model).load_from_checkpoint(path_parser.checkpoint_path)
    model = model.eval().to(device)

    with torch.no_grad():
        # Perform marching cubes and export the mesh
        export_marching_cubes(model, config_args, cfg, device)