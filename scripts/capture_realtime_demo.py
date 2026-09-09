"""Capture high-resolution screenshot of realtime_activity_demo.py during live active walking."""

import sys
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Add scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realtime_activity_demo import ActivityMonitorWindow

def capture_demo():
    app = QApplication.instance() or QApplication(sys.argv)
    
    replay_file = "data/raw/walking/sample_20260715_192540.csv"
    print(f"Loading ActivityMonitorWindow with replay: {replay_file}")
    win = ActivityMonitorWindow(port="COM3", replay=replay_file)
    win.resize(1100, 720)
    win.show()
    
    start = time.time()
    captured = False
    
    # Run loop for 3.5 seconds to populate 200 history frames and establish active walking state
    while time.time() - start < 3.5:
        app.processEvents()
        time.sleep(0.015)
        
    # Grab window pixmap
    pixmap = win.grab()
    output_path = "assets/realtime_demo.png"
    pixmap.save(output_path, "PNG")
    print(f"Successfully saved clean demo screenshot to {output_path} ({pixmap.width()}x{pixmap.height()})")
    
    win.worker.stop()
    win.close()
    app.processEvents()

if __name__ == "__main__":
    capture_demo()
