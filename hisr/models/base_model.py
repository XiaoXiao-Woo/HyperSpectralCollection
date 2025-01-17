# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2022/10/4 17:34
# @Author  : Xiao Wu
# @reference:
#
import os
from scipy import io as sio
import torch
from udl_vis.Basis.python_sub_class import ModelDispatcher
from hisr.common.evaluate import analysis_accu
import imageio
import numpy as np


def save_results(idx, save_model_output, filename, save_fmt, output):
    if filename is None:
        save_name = os.path.join(
            f"{save_model_output}", "output_mulExm_{}.mat".format(idx)
        )
        sio.savemat(save_name, {"sr": output.cpu().detach().numpy()})
    else:
        filename = os.path.basename(filename).split(".")[0]
        if save_fmt == "mat":
            filename = "/".join([save_model_output, "output_" + filename + ".mat"])
            sio.savemat(filename, {"sr": output.cpu().detach().numpy()})
        else:
            raise NotImplementedError(f"{save_fmt} is not supported")


class HISRModel(ModelDispatcher, name="hisr"):

    _models = {}

    def __init__(self, device, model=None, criterion=None):
        super(HISRModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = device
        if model is not None:
            if hasattr(self.model, "module"):
                self.model.module.forward_task = getattr(
                    self.model.module, f"forward_{self._name}"
                )
            else:
                self.model.forward_task = getattr(self.model, f"forward_{self._name}")

    def __init_subclass__(cls, name="", **kwargs):

        # print(name, cls)
        if name != "":
            cls._models[name] = cls
            cls._name = name

        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

    def train_step(self, *args, **kwargs):
        log_vars = {}
        data = args[0]
        data = {k: v.to(self.device) for k, v in data.items()}
        # print(f"gt.device: {data['gt'].device}")

        # sr = self.model.train_step(data, **kwargs)
        # with torch.no_grad():
        #     log_vars.update(analysis_accu(data['gt'].cuda(), sr, 4, choices=4))
        # loss = self.criterion(sr, data['gt'])
        # log_vars.update(loss=loss["loss"])
        # return {"loss": loss["loss"], "log_vars": log_vars}
        log_vars = self.model(data, **kwargs)

        return {"loss": log_vars["loss"], "log_vars": log_vars}

    def val_step(self, *args, **kwargs):
        data = args[0]
        gt = data.pop("gt")

        for k in list(data.keys()):
            if hasattr(data[k], "cuda"):
                data[k] = data[k].to(self.device)
            else:
                kwargs[k] = data.pop(k)

        sr = self.model(data, **kwargs)
        # print(sr.max())
        sr = torch.clamp(sr, 0, 1)
        with torch.no_grad():
            metrics = analysis_accu(
                gt[0].permute(1, 2, 0).to(self.device),
                sr[0].permute(1, 2, 0),
                4,
                flag_cut_bounds=False,
            )
            metrics.update(metrics)
        if kwargs["save_dir"] is not None:
            if kwargs["save_fmt"] == "mat":
                save_name = kwargs["save_dir"] + f"/output_mulExm_{kwargs['idx']}.mat"
                sio.savemat(save_name, {"output": sr.permute(0, 2, 3, 1).cpu().numpy()})
            else:
                save_name = kwargs["save_dir"] + f"/output_mulExm_{kwargs['idx']}.png"
                imageio.imwrite(
                    save_name,
                    (sr.permute(0, 2, 3, 1)[0, :, :, [30, 19, 9]].cpu().numpy() * 255.0)
                    .round()
                    .astype(np.uint8),
                )
        # 高光谱评价逐维度算的
        # sr[kwargs['idx'] - 1] = sr1.cpu()
        #
        # if kwargs['idx'] == 10:
        #     save_name = kwargs['save_dir'] + "/outputCAVE.mat"
        #     sio.savemat(save_name, {'output': sr.permute(0, 2, 3, 1).numpy()})

        return {"log_vars": metrics}


class HsPansharpeningModel(ModelDispatcher, name="hsp"):

    _models = {}

    def __init__(self, model=None, criterion=None):
        super(HsPansharpeningModel, self).__init__()
        self.model = model
        self.criterion = criterion

    def __init_subclass__(cls, name="", **kwargs):

        # print(name, cls)
        if name != "":
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name_

    def train_step(self, *args, **kwargs):

        return self.model.train_step(args[0], **kwargs)

        # loss_backward = loss.pop('loss')
        # log_vars.update(**loss)
        # return {'loss': loss['loss'], 'log_vars': log_vars, 'num_samples': len(args[0]['pan'])}

    def val_step(self, *args, **kwargs):
        sr, gt = self.model["PAN2MS"].val_step(*args, **kwargs)
        result_our = torch.squeeze(sr).permute(1, 2, 0)
        result_our = result_our * 2047
        metrics = analysis_accu(gt.to(self.device).squeeze(0), result_our, 4)

        if kwargs["idx"] not in [220, 231, 236, 469, 766, 914]:
            if kwargs["save_fmt"] is not None:
                save_results(
                    kwargs["idx"],
                    kwargs["save_dir"],
                    kwargs["filename"],
                    kwargs["save_fmt"],
                    result_our,
                )
        return {"log_vars": metrics}


class MHIFModel(HISRModel, name="mhf"):
    _models = {}

    def __init__(self, device=None, model=None, criterion=None):
        super(MHIFModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = device
        if model is not None:
            if hasattr(self.model, "module"):
                self.model.module.forward_task = getattr(
                    self.model.module, f"forward_{self._name}"
                )
            else:
                try:
                    self.model.forward_task = getattr(
                        self.model, f"forward_{self._name}"
                    )
                except:
                    raise ValueError(
                        f"No forward_{self._name} method in model {self.model}"
                    )

    def __init_subclass__(cls, name="", **kwargs):

        # print(name, cls)
        if name != "":
            cls._models[name] = cls
            cls._name = name
        else:
            cls._models[cls.__name__] = cls
            cls._name = cls.__name__
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')

