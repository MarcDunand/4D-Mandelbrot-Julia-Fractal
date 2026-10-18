"""Run the WHCT viewer:  python -m whct  [--mapping NAME]"""

import argparse
import sys

from PySide6.QtGui import QColor, QPalette, QSurfaceFormat
from PySide6.QtWidgets import QApplication

STYLE = """
QWidget#bar { background: #16191E; border-top: 1px solid #2B3039; }
QLabel { color: #A3AAB7; }
QLabel#status { color: #8A919E; font-size: 9pt; }
"""


def dark_palette() -> QPalette:
    pal = QPalette()
    roles = {
        QPalette.Window: "#16191E", QPalette.WindowText: "#E7E9EE", QPalette.Base: "#0E1013",
        QPalette.AlternateBase: "#1E2228", QPalette.Text: "#E7E9EE", QPalette.Button: "#1E2228",
        QPalette.ButtonText: "#E7E9EE", QPalette.Highlight: "#3D5A99", QPalette.HighlightedText: "#FFFFFF",
        QPalette.ToolTipBase: "#1E2228", QPalette.ToolTipText: "#E7E9EE", QPalette.PlaceholderText: "#6B7280",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    return pal


def main(argv=None):
    ap = argparse.ArgumentParser(description="WHCT 4D viewer")
    ap.add_argument("--mapping", default=None, help="Color mapping name, e.g. centered_bgr or 'raw bits'.")
    args = ap.parse_args(argv)

    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)

    from .window import MainWindow  # after QApplication so GL defaults apply

    win = MainWindow(mapping=args.mapping)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
