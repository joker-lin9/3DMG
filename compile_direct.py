import os
import subprocess
import sys

def compile_with_pybind11():
    # 获取包含路径
    import pybind11
    pybind11_include = pybind11.get_include()
    python_include = sys.prefix + '\\include'
    
    # 编译命令
    cmd = [
        'cl.exe', '/O2', '/LD', '/EHsc',
        f'/I{pybind11_include}',
        f'/I{python_include}',
        'mesh_inpaint_processor.cpp',
        f'/link', f'/LIBPATH:{sys.prefix}\\libs'
    ]
    
    print('Running:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print('Compilation successful!')
        # 重命名为 .pyd 文件
        if os.path.exists('mesh_inpaint_processor.dll'):
            os.rename('mesh_inpaint_processor.dll', 'mesh_inpaint_processor.pyd')
    else:
        print('Compilation failed:')
        print(result.stderr)

if __name__ == '__main__':
    compile_with_pybind11()