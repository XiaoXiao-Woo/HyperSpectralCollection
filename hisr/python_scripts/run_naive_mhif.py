from hisr.python_scripts.accelerate_mhif import accelerate_run


def torch_run(full_config_path, import_path=None):
    accelerate_run(
        full_config_path=full_config_path,
        import_path=import_path,
        backend="naive",
        launcher="pytorch",
    )


if __name__ == "__main__":
    torch_run()
