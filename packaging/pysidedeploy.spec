[app]
title = QQQSimulator
project_dir = D:/optionsimulation/packaging
input_file = main.py
exec_directory = D:/optionsimulation/dist
project_file = 
icon = D:\optionsimulation\.venv\Lib\site-packages\PySide6\scripts\deploy_lib\pyside_icon.ico

[python]
packages = Nuitka==4.1.1
android_packages = buildozer==1.6.0,cython==0.29.33

[qt]
qml_files = 
excluded_qml_plugins = 
modules = 
plugins = platforms,imageformats

[android]
wheel_pyside = 
wheel_shiboken = 
plugins = 

[ios]
wheel_pyside = 
wheel_shiboken = 
xcframework_path = 
bundle_id = 
team_id = 
version = 1.0
deployment_target = 
entitlements = 
signing_style = Automatic
header_search_paths = 

[nuitka]
macos.permissions = 
mode = standalone
extra_args = --assume-yes-for-downloads --noinclude-qt-translations --include-package=qqq_sim --include-package=duckdb --include-package=scipy._external.array_api_compat --include-module=PySide6.QtOpenGL --include-package-data=tzdata --nofollow-import-to=torch,pytest,tkinter,matplotlib,IPython --jobs=6 --windows-console-mode=attach --output-filename=QQQSimulator.exe --include-distribution-metadata=PySide6 --include-distribution-metadata=pyqtgraph --include-distribution-metadata=numpy --include-distribution-metadata=scipy --include-distribution-metadata=pyarrow --include-data-files=D:/optionsimulation/.venv/Lib/site-packages/duckdb-2.0.0.dist-info/METADATA=duckdb-2.0.0.dist-info/METADATA --include-distribution-metadata=tzdata

[buildozer]
mode = debug
recipe_dir = 
jars_dir = 
ndk_path = 
sdk_path = 
local_libs = 
arch = 

