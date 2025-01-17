def run_demo():
    from hisr.configs.configs import TaskDispatcher
    from udl_vis.AutoDL.trainer import main
    from hisr.common.builder import build_model, HISRSession
    cfg = TaskDispatcher.new(task='hisr', mode='entrypoint', arch='DHIF',
                             data_dir="G:/woo/Datasets/hisr/GF5-GF1",
                             resume_from=r"G:\woo\gitSync\HyperSpectralCollection\results\hisr\cave_x4\DHIF\Test\model_2023-09-20-01-21-01\500.pth.tar".replace('\\', '/'),
                             train_path="train_GF5_GF1_23tap_new.h5",
                             test_path="test_GF5_GF1_23tap_new.h5",
                             # reset_lr=True,
                             )
    print(TaskDispatcher._task.keys())

    main(cfg, build_model, HISRSession)

if __name__ == '__main__':
    run_demo()
