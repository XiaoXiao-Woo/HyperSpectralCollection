from hisr import build_model, getDataSession
from hisr.models.base_model import HISRModel
from hisr import build_model as build_mhif_model

def hydra_run(full_config_path="configs/config", import_path=None, build_model=None):
    from udl_vis.AutoDL.trainer import run_hydra

    run_hydra(
        full_config_path=full_config_path,
        import_path=import_path,
        taskModel=HISRModel,
        build_model=build_model if build_model is not None else build_mhif_model,
        getDataSession=getDataSession,
    )


if __name__ == "__main__":
    hydra_run(full_config_path="configs/config")
