import sys
import os
import glob
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import numpy as np

class Annotation(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.file_names = glob.glob(os.path.dirname(os.getcwd())+'/dataset/depth*.jpg')
        self.file_num = len(self.file_names)

        self.window = QWidget()
        self.window.setWindowTitle('Depth Image View')

        self.px, self.py = None, None
        self.points = []
        self.psets = []

    def loadImage(self, idx):
        # ファイルを読み込み

        cv_depth = Image.open(self.file_names[idx])
        # RGB形式に変換
        cv_depth = np.array(cv_depth)
        h, w = cv_depth.shape[:2]
        qimg = QImage(cv_depth.flatten(), w, h, QImage.Format_RGB888)
    
        imageLabel = QLabel()
    
        # ラベルに読み込んだ画像を反映
        imageLabel.setPixmap(QPixmap.fromImage(qimg))
    
        # スケールは1.0
        imageLabel.scaleFactor = 1.0
        layout = QVBoxLayout()
        layout.addWidget(imageLabel)
        self.window.setLayout(layout)
        self.window.resize(400, 300)
        # self.window.show()
    
    def mousePressEvent(self, event):
        self.points.append(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.points.append(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        self.pressed = False
        self.psets.append(self.points)
        self.points = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.white)
        painter.drawRect(self.rect())

        painter.setPen(Qt.red)

        for points in self.psets:
            painter.drawPolyline(*points)

if __name__ == '__main__':
    app = QApplication([])
    ano = Annotation()
    for i in range(ano.file_num):
        ano.loadImage(i)
        ano.show()
    sys.exit(app.exec_())