from setuptools import setup, find_packages

if __name__ == '__main__':

    setup(
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ],
        name="hisr",
        description="HyperSpectralCollection based on UDL (https://github.com/XiaoXiao-Woo/UDL)",
        author="XiaoXiao-Woo",
        author_email="wxwsx1997@gmail.com",
        url="https://github.com/XiaoXiao-Woo/HyperSpectralCollection",
        version="0.1",
        packages=find_packages(),
        license="GPLv3",
        python_requires=">=3.7",
        entry_points={  # 如果有命令行工具
            "console_scripts": [
                "accelerate_mhif=hisr.python_scripts.accelerate_mhif:hydra_run",
            ],
        },
        install_requires=[
            "psutil",
            "opencv-python",
            "numpy",
            "matplotlib",
            "tensorboard",
            "addict",
            "yapf",
            "imageio",
            "colorlog",
            "scipy",
            "h5py",
            "regex",
            "packaging",
            "colorlog",
            "pyyaml",
        ],
    )
