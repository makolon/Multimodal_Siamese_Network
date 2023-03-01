import os
import posix
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# text_dir = os.path.dirname(os.getcwd()) + '/dataset/depth.jpg'

class Test(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.images = None
        self.positions = []
        while true:
            pass

    def initUI(self):
        self.setWindowTitle("depth image")
        self.resize(640, 640)
        self.link = ''
        self.view = QGraphicsView()
        self.view.setMouseTracking(True)
        self.scene = QGraphicsScene(self.view)
        self.srect = self.view.rect()
        self.btn = QPushButton("Select", self)
        self.btn.clicked.connect(self.ShowDialog)
        self.view.installEventFilter(self)

        layout = QGridLayout()
        layout.setSpacing(10)
        layout.addWidget(self.view, 2, 0, 1, 1)
        layout.addWidget(self.btn, 1, 0)

        self.setLayout(layout)
        self.move(300, 200)
        self.show()

    def ShowDialog(self):
        if self.link != "":
            text = self.link
        else:
            text = text_dir
        fname = QFileDialog.getOpenFileName(self, 'Open file', text, "画像ファイル(*.jpg)")
        self.link = fname[0]
        
        depth_image = Image.open(fname[0])
        depth_image = np.array(depth_image)
        h, w = depth_image.shape[:2]
        # q_depth = QImage(depth_image.flatten(), w, h, QImage.Format_Mono)

        self.images = QPixmap(fname[0])
        self.pic_item = QGraphicsPixmapItem(self.images)
        self.scene.addItem(self.pic_item)
        self.view.setScene(self.scene)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and source is self.view and self.view.itemAt(event.pos()):
            if event.button() == Qt.LeftButton:
                pos = self.view.mapToScene(event.pos())
                # QMessageBox.warning(self, "Message", str(pos.x())+"," + str(pos.y()), QMessageBox.Ok, QMessageBox.Ok)
                self.positions.append(pos)
                if len(self.positions) == 2:
                    return QWidget.eventFilter(self, source, event)
        return False

    def paintEvent(self, event): 
        if self.images is not None:
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.images) 
            pen = QPen(Qt.red, 3) 
            painter.setPen(pen)
            if len(self.positions) > 1:
                painter.drawLine(int(self.positions[0].x()), int(self.positions[0].y()),
                    int(self.positions[1].x()), int(self.positions[1].y()))
            print(self.positions)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Test()
    sys.exit(app.exec_())
