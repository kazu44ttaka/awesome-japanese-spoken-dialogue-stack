"""デスクトップ版（PySide6）で共通に使う部品。"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox


def fill_grouped(combo: QComboBox, items: list[dict], current: str) -> None:
    """items（name・label・group を持ち、group ごとに並んだもの）を、group の見出しを挟んで combo に入れる。

    見出しは選べない行にする。行のデータは name なので、選んだものは combo.currentData() で取る。
    """
    combo.clear()
    group = None
    for it in items:
        if it["group"] != group:
            group = it["group"]
            combo.addItem(f"―― {group} ――")
            head = combo.model().item(combo.count() - 1)
            head.setEnabled(False)
            font = head.font()
            font.setBold(True)
            head.setFont(font)
        combo.addItem("    " + it["label"], it["name"])
    i = combo.findData(current)
    combo.setCurrentIndex(i if i >= 0 else 1)
