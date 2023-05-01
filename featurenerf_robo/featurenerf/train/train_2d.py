# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import trainlib
from model import make_model, loss
from render import NeRFEmbedRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch
from dotmap import DotMap


def extra_args(parser):
    parser.add_argument(
        "--batch_size", "-B", type=int, default=4, help="Object batch size ('SB')"
    )
    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        "--freeze_enc",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )

    parser.add_argument(
        "--no_bbox_step",
        type=int,
        default=100000,
        help="Step to stop using bbox sampling",
    )
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        default=None,
        help="Freeze encoder weights and only train MLP",
    )
    parser.add_argument("--not_use_wandb", action="store_true")
    return parser


args, conf = util.args.parse_args(extra_args, training=True, default_ray_batch_size=128)
device = util.get_cuda(args.gpu_id[0])

if args.dataset_format.startswith("feature"):
    extra_dataset_kwargs = {"synset": args.synset}
else:
    extra_dataset_kwargs = {}

dset, val_dset, _ = get_split_dataset(args.dataset_format, args.datadir, task_list=conf['data']['task_list'], **extra_dataset_kwargs)
print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

embed_dim = conf["model"]["d_embed"]
net = make_model(conf["model"]).to(device=device)
student_net = net.encoder
dummy_input = torch.randn(1, 3, 128, 128).to(device=device)
dummy_output = student_net(dummy_input)
net_aux = torch.nn.Conv2d(dummy_output.shape[1], embed_dim, kernel_size=1, padding=0, stride=1).to(device=device)

net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

regress_coord = conf.get_float("loss.lambda_coord", 0.0) > 0
renderer = NeRFEmbedRenderer.from_conf(conf["renderer"], lindisp=dset.lindisp, regress_coord=regress_coord).to(device=device)

# Parallize
render_par = renderer.bind_parallel(net, args.gpu_id).eval()

nviews = list(map(int, args.nviews.split()))


class Student2DTrainer(trainlib.Trainer2DWandb):
    def __init__(self):
        super().__init__(net, net_aux, dset, val_dset, args, conf["train"], device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        self.lambda_embed = conf.get_float("loss.lambda_embed", 1.0)
        self.lambda_coord = conf.get_float("loss.lambda_coord", 0.0)
        print(
            "lambda coarse {}, fine {}, embed {}, coord {}".format(self.lambda_coarse, self.lambda_fine,
                                                                   self.lambda_embed, self.lambda_coord)
        )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        self.embed_crit = torch.nn.MSELoss()
        self.coord_crit = torch.nn.MSELoss()

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                renderer.load_state_dict(
                    torch.load(self.renderer_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far

        self.use_bbox = args.no_bbox_step > 0
        self.mask_feat = conf.get_bool("data.mask_feat", False)
        self.mask_white_bkgd = conf.get_bool("renderer.white_bkgd", True)

    def post_batch(self, epoch, batch):
        renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(renderer.state_dict(), self.renderer_state_path)

    def calc_losses(self, data, is_train=True, global_step=0):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)
        all_feats = data["feats"].to(device=device)  # (SB, NV, D, H // 8, W // 8)

        SB, NV, _, H, W = all_images.shape
        all_images = all_images*0.5 + 0.5 # [-1, 1] -> [0, 1]

        loss = 0.
        # reshape imgs and feats to (SB * NV, 3, H, W)
        all_images = all_images.reshape(SB * NV, 3, H, W)
        all_feats = all_feats.reshape(SB * NV, -1, all_feats.shape[-2], all_feats.shape[-1])

        # pred and compute loss
        feats_pred = student_net(all_images)
        feats_pred = net_aux(feats_pred)
        feats_gt = F.interpolate(all_feats, size=feats_pred.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.embed_crit(feats_pred, feats_gt)

        loss_dict = {}
        if is_train:
            loss.backward()
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data, global_step):
        return self.calc_losses(data, is_train=True, global_step=global_step)

    def eval_step(self, data, global_step):
        renderer.eval()
        losses = self.calc_losses(data, is_train=False, global_step=global_step)
        renderer.train()
        return losses

    def vis_step(self, data, global_step, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        feats = data["feats"][batch_idx].to(device=device)  # (NV, D, H // 8, W // 8)

        
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        NV, _, H, W = images.shape
        gt = images_0to1[0].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        feats_gt = feats[0].permute(1, 2, 0).cpu().numpy().mean(-1)
        with torch.no_grad():
            feats_pred = student_net(images_0to1)
            feats_pred = net_aux(feats_pred)
            feats_tgt = F.interpolate(feats, size=feats_pred.shape[-2:], mode="bilinear", align_corners=False)
            # compute loss
            loss = self.embed_crit(feats_pred, feats_tgt)
            # reshape for aligned visualization
            feats_pred = F.interpolate(feats_pred, size=feats.shape[-2:], mode="bilinear", align_corners=False)
            feats_pred = feats_pred[0].permute(1, 2, 0).cpu().numpy().mean(-1)


        print("c rgb min {} max {}".format(images_0to1.min(), images_0to1.max()))

        feats_gt = util.cmap(feats_gt) / 255
        feats_pred = util.cmap(feats_pred) / 255
        vis_list = [
            gt,
            feats_gt,
            feats_pred,
        ]

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        vals = {'feature_loss': loss.item()}

        # set the renderer network back to train mode
        renderer.train()
        return vis, vals


trainer = Student2DTrainer()
trainer.start()
