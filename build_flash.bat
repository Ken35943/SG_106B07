@echo off
set "IDF_TOOLS_PATH=C:\Espressif"
set "PATH=C:\Espressif\tools\idf-python\3.11.2;%PATH%"
call C:\Espressif\frameworks\esp-idf-v5.4.4\export.bat
cd c:\Users\Ken\Documents\ESP-CSI\esp-csi\examples\get-started\csi_send
call idf.py set-target esp32s3
call idf.py build
call idf.py flash -b 921600 -p COM5
cd ..\csi_recv
call idf.py set-target esp32s3
call idf.py build
call idf.py flash -b 921600 -p COM3
