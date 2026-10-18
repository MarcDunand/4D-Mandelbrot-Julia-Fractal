"""Editable number fields that float over the widgets in values mode (V).

Click a field to type in it. Press and drag left or right over a field to change
its value smoothly. Rotation boxes start minimized and open when clicked; the
clicked box comes to the front.
"""

from __future__ import annotations

import re
from typing import List, Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QLineEdit, QToolButton, QVBoxLayout, QWidget

BOX_STYLE = """
QFrame#valuebox { background: rgba(22,25,30,240); border: 1px solid rgba(255,255,255,90); border-radius: 7px; }
QToolButton { border: 0; background: transparent; font: bold 9pt "Consolas"; padding: 2px 6px; }
QLabel { color: #A3AAB7; font: 8pt "Consolas"; }
QLineEdit { background: #0E1013; color: #E7E9EE; border: 1px solid #2B3039; border-radius: 3px;
            font: 8pt "Consolas"; padding: 0 3px; min-width: 46px; max-width: 54px; min-height: 18px; }
QLineEdit:focus { border-color: #82AEFF; }
"""


def parse_number(text: str) -> Optional[float]:
    cleaned = text.replace("−", "-").replace(",", ".")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
    return float(m.group(0)) if m else None


class ScrubLineEdit(QLineEdit):
    """A line edit you can also drag horizontally to change."""

    scrubStarted = Signal()
    scrubbed = Signal(float)  # pixels moved since last event
    scrubEnded = Signal()
    committed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.SizeHorCursor)
        self._press: Optional[QPoint] = None
        self._last_x = 0
        self._scrubbing = False
        self.editingFinished.connect(self._commit)

    def mousePressEvent(self, e):
        if self.hasFocus() or e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        self._press = e.globalPosition().toPoint()
        self._last_x = self._press.x()
        self._scrubbing = False
        e.accept()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return super().mouseMoveEvent(e)
        x = e.globalPosition().toPoint().x()
        if not self._scrubbing and abs(x - self._press.x()) < 3:
            return
        if not self._scrubbing:
            self._scrubbing = True
            self.scrubStarted.emit()
        self.scrubbed.emit(float(x - self._last_x))
        self._last_x = x

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return super().mouseReleaseEvent(e)
        was_scrubbing = self._scrubbing
        self._press = None
        self._scrubbing = False
        if was_scrubbing:
            self.scrubEnded.emit()
        else:
            self.setFocus(Qt.MouseFocusReason)
            self.selectAll()
            self.setCursor(Qt.IBeamCursor)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self.setCursor(Qt.SizeHorCursor)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.clearFocus()
            return
        super().keyPressEvent(e)

    def _commit(self):
        v = parse_number(self.text())
        if v is not None:
            self.committed.emit(v)
        self.clearFocus()

    def set_display(self, text: str):
        if not self.hasFocus() and self._press is None:
            self.setText(text)


class ValueBox(QFrame):
    """A small floating box of labeled fields. Minimized boxes show only their title."""

    raised = Signal()

    def __init__(self, parent: QWidget, title: str, color: str, labels: List[str], minimized: bool):
        super().__init__(parent)
        self.setObjectName("valuebox")
        self.setStyleSheet(BOX_STYLE)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(3, 2, 3, 3)
        lay.setSpacing(2)
        self.header: Optional[QToolButton] = None
        if minimized:
            self.header = QToolButton(self)
            self.header.setText(title)
            self.header.setStyleSheet(f"color: {color};")
            self.header.clicked.connect(self.toggle)
            lay.addWidget(self.header)
        self.body = QWidget(self)
        grid = QGridLayout(self.body)
        grid.setContentsMargins(2, 0, 2, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        self.fields: List[ScrubLineEdit] = []
        cols = 2 if len(labels) > 2 else 1
        for i, lab in enumerate(labels):
            r, c = divmod(i, cols)
            ql = QLabel(lab, self.body)
            if not minimized:
                ql.setStyleSheet(f"color: {color}; font-weight: bold;")
            f = ScrubLineEdit(self.body)
            f.scrubStarted.connect(self._start_scrub)
            f.scrubEnded.connect(self._end_scrub)
            grid.addWidget(ql, r, 2 * c)
            grid.addWidget(f, r, 2 * c + 1)
            self.fields.append(f)
        lay.addWidget(self.body)
        self.body.setVisible(not minimized)
        self.pinned = False  # held in place while one of its fields is being dragged
        self.adjustSize()

    def _start_scrub(self):
        self.pinned = True
        self._bring_front()

    def _end_scrub(self):
        self.pinned = False

    def mousePressEvent(self, e):
        self._bring_front()
        super().mousePressEvent(e)

    def _bring_front(self):
        self.raise_()
        self.raised.emit()

    def toggle(self):
        self.body.setVisible(not self.body.isVisible())
        self._bring_front()
        center = self.geometry().center()
        self.adjustSize()
        self.move(center - self.rect().center())

    def place_centered(self, x: float, y: float):
        if self.pinned:
            return
        self.adjustSize()
        self.move(int(x - self.width() / 2), int(y - self.height() / 2))
