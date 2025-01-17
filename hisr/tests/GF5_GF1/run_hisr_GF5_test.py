def run_demo():
    from hisr.configs.configs import TaskDispatcher
    from udl_vis.AutoDL.trainer import main
    from hisr.common.builder import build_model, getDataSession
    cfg = TaskDispatcher.new(task='hisr', mode='entrypoint', arch='DHIF',
                             data_dir="G:/woo/Datasets/hisr/GF5-GF1",
                             resume_from=r"G:\woo\gitSync\HyperSpectralCollection\results\hisr\cave_x4\DHIF\Test\model_2023-09-20-01-21-01\941.pth.tar".replace('\\', '/'),
                             train_path="train_GF5_GF1_23tap_new.h5",
                             # test_path="test_GF5_GF1_23tap_new.h5",
                             test_path="test_GF5_GF1_OrigScale.h5"
                             # reset_lr=True,
                             )
    print(TaskDispatcher._task.keys())
    cfg.workflow = [("test", 1)]
    cfg.eval = True

    main(cfg, build_model, getDataSession)

if __name__ == '__main__':
    run_demo()
