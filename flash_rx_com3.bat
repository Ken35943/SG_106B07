@echo off
echo Flashing csi_recv firmware to COM3 (Receiver 2)...
python -m esptool --chip esp32s3 -p COM3 -b 921600 --before default_reset --after hard_reset write_flash --flash_mode dio --flash_freq 80m 0x0 esp-csi/examples/get-started/csi_recv/build/bootloader/bootloader.bin 0xa000 esp-csi/examples/get-started/csi_recv/build/partition_table/partition-table.bin 0x20000 esp-csi/examples/get-started/csi_recv/build/csi_recv.bin
echo Flash complete!
pause
