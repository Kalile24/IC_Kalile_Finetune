"""Copia local de predict.py (traj_intention/) com argumento de contexto (OS-6).

Nao altera o arquivo original em IC_Kalile_Intention_Prediction_HRC/. Mudanca
em relacao ao original: IntentionPredictor aceita `context_dim` na construcao
(repassado ao Model_FinalIntention desta copia, hrc-finetune/DLinear.py) e
`predict()` ganha um argumento opcional `context` que e repassado ao forward
do modelo como tensor (1, context_dim), conforme Secao 5.3 do roadmap:

    "predict.py: o IntentionPredictor recebe o vetor como argumento opcional
    de predict() e o repassa ao forward do modelo como tensor
    (1, context_dim)."

Com context_dim=0 (padrao) ou context=None, o comportamento e identico ao
predict.py original - retrocompatibilidade exigida pela OS-6.
"""
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
from torch.nn.functional import softmax
from torch.distributions import Categorical

from DLinear import Model_FinalIntention, Model_FinalTraj

INTENTION_LIST = {"no_action": 0, "get_connectors": 1, "get_screws": 2, "get_wheels": 3}

SKELETON_LIST = {
    "left_shoulder": 0,
    "right_shoulder": 1,
    "left_elbow": 2,
    "right_elbow": 3,
    "left_wrist": 4,
    "right_wrist": 5,
    "left_pinky": 6,
    "right_pinky": 7,
    "left_index": 8,
    "right_index": 9,
    "left_thumb": 10,
    "right_thumb": 11,
    "left_hip": 12,
    "right_hip": 13,
    "nose": 14,
}


class Args:
    def __init__(self, **kwargs):
        self.default = {
            "seq_len": 5,
            "pred_len": 5,
            "class_num": 4,
            "individual": False,
            "channels": 15 * 3,
            "half_body": False,
            "epochs": 40,
            "input_type": "pkl",
        }
        for key in (
            "seq_len",
            "pred_len",
            "class_num",
            "individual",
            "channels",
            "half_body",
            "epochs",
            "input_type",
        ):
            if key in kwargs and kwargs[key]:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, self.default[key])


class IntentionPredictor:
    def __init__(
        self,
        ckpt_path=None,
        model_type="final_intention",
        no_mask=False,
        filter_type=None,
        context_dim=0,
        **kwargs,
    ):
        args = Args(**kwargs)
        self.context_dim = context_dim
        if model_type == "final_intention":
            self.model = Model_FinalIntention(args, context_dim=context_dim)
        elif model_type == "final_traj":
            if context_dim:
                raise ValueError("context injection is only implemented for final_intention")
            self.model = Model_FinalTraj(args)
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
        else:  # default
            if filter_type:
                checkpoint = torch.load(
                    f"{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole_{args.input_type}_{model_type}_nomask{no_mask}_filter{filter_type}.pth"
                )
            else:
                checkpoint = torch.load(
                    f"{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole_{args.input_type}_{model_type}_nomask{no_mask}.pth"
                )
        # strict=False: um checkpoint treinado com context_dim=0 nao tem as
        # chaves context_proj.*, que so existem quando context_dim>0.
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    def predict(self, poses, restrict, poses_world=None, context=None):
        assert poses.shape[0] == 1
        if context is not None and not torch.is_tensor(context):
            context = torch.tensor(context, dtype=torch.float32)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)  # (context_dim,) -> (1, context_dim)

        if self.context_dim > 0:
            pred_traj, pred_intention = self.model(poses, context)
        else:
            pred_traj, pred_intention = self.model(poses)
        batch_intention = torch.argmax(pred_intention, 1)

        working_area_flag = False
        ood_flag = False
        if restrict == "working_area":
            working_area_flag = True
        elif restrict == "ood":
            ood_flag = True
        elif restrict == "all":
            working_area_flag = True
            ood_flag = True
        elif restrict == "no":
            pass
        else:
            print("restrict invalid!")
            return

        if ood_flag:
            entropy = Categorical(probs=softmax(pred_intention, dim=1)[0].detach()).entropy()
            if entropy > 0.4 and torch.argmax(pred_intention) != 3:
                batch_intention[0] = INTENTION_LIST["no_action"]
            elif entropy > 0.5 and torch.argmax(pred_intention) == 3:
                batch_intention[0] = INTENTION_LIST["no_action"]

        if working_area_flag:
            intention = batch_intention[0]
            if intention != INTENTION_LIST["no_action"]:
                if intention == INTENTION_LIST["get_connectors"]:
                    index = 16
                    right_wrist_traj = poses_world[:, :, index, 0]
                    if right_wrist_traj[0][-1] > -22:  # TODO
                        print("modifying connectors!")
                        batch_intention[0] = INTENTION_LIST["no_action"]
                elif intention == INTENTION_LIST["get_screws"]:
                    index1 = 16
                    index2 = 15
                    right_wrist_traj = poses_world[:, :, index1, 0]
                    left_wrist_traj = poses_world[:, :, index2, 0]
                    if right_wrist_traj[0][-1] < 70 and left_wrist_traj[0][-1] < 70:  # TODO
                        batch_intention[0] = INTENTION_LIST["no_action"]
                        print("modifying screws!")
                elif intention == INTENTION_LIST["get_wheels"]:
                    index1 = 16
                    index2 = 15
                    right_wrist_traj = poses_world[:, :, index1, 0]
                    left_wrist_traj = poses_world[:, :, index2, 0]
                    if right_wrist_traj[0][-1] < 65 and left_wrist_traj[0][-1] < 65:  # TODO
                        batch_intention[0] = INTENTION_LIST["no_action"]
                        print("modifying wheels!")

        return pred_traj, batch_intention
