# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2022/10/4 17:42
# @Author  : Xiao Wu
# @reference:
#
import udl_vis.Basis.option

import hisr.models

def build_model(cfg=None):
    from udl_vis.Basis.python_sub_class import ModelDispatcher

    return ModelDispatcher.build_model_from_task(cfg)

def getDataSession(cfg):

    from .hisr_dataset import HISRSession as DataSession

    return DataSession(cfg)
