@echo off

:: 第一个命令行窗口，运行 shape_run.py
start cmd /k "conda activate Hunyuan3D && python shape_run.py && pause"

:: 第二个命令行窗口，运行 t2i_run.py
start cmd /k "conda activate Hunyuan3D && python t2i_run.py && pause"
