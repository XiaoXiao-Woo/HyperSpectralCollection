def run_demo():
    from hisr.configs.configs import TaskDispatcher
    from udl_vis.AutoDL.trainer import main
    from hisr.common.builder import build_model, getDataSession
    cfg = TaskDispatcher.new(task='hisr', mode='entrypoint', arch='SSRNet',
                             data_dir="G:/woo/Datasets/hisr/GF5-GF1",
                             resume_from=r"".replace('\\', '/'),
                             train_path="train_GF5_GF1_23tap.h5",
                             test_path="test_GF5_GF1_23tap.h5",
                             )
    # cfg.reset_lr = True
    cfg.lr = 5e-5
    cfg.dataset = {'train': 'GF5-GF1', 'valid': 'GF5-GF1', 'test': 'GF5-GF1'}
    print(TaskDispatcher._task.keys())

    main(cfg, build_model, getDataSession)

if __name__ == '__main__':
    run_demo()
