import sys 
import os
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 

class Example(QWidget): 
    def __init__(self): 
        super().__init__() 
        self.setGeometry(30, 30, 640, 480) 

    def paintEvent(self, event): 
        painter = QPainter(self)
        file_name = os.path.dirname(os.getcwd()) + '/dataset/depth_1.jpg'
        pixmap = QPixmap(file_name) 
        painter.drawPixmap(self.rect(), pixmap) 
        pen = QPen(Qt.red, 3) 
        painter.setPen(pen) 
        painter.drawLine(10, 20, 20, 30) 

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and source is self.view and self.view.itemAt(event.pos()):
            if event.button() == Qt.LeftButton:
                pos = self.view.mapToScene(event.pos())
                QMessageBox.warning(self, "Message", str(pos.x())+"," + str(pos.y()), QMessageBox.Ok, QMessageBox.Ok)
                self.positions.append(pos)
                if len(self.positions) == 2:
                    next
            print('self.positions: ', self.positions)
        return QWidget.eventFilter(self, source, event)

if __name__ == '__main__': 
    app = QApplication(sys.argv) 
    ex = Example() 
    ex.show() 
    sys.exit(app.exec_()) 