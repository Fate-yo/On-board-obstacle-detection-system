# This Python file uses the following encoding: utf-8
from datetime import datetime
import io

import cv2
from PyQt5.QtGui import *
from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QDialog, QApplication, QWidget
from form import Ui_Widget
from ultralytics import YOLO
from PyQt5.QtMultimedia import QSound
from login import Ui_Login
from regiest import Ui_Regiest
import hashlib, sys, mysql.connector
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageQt



class Widget(QWidget, Ui_Widget):
    source_parameter = "0"
    Video_Photo_camera = 0  # 参数Video_Photo_camera  0 指定为照片 1 为视频 2 为摄像头
    directory = []
    msg = ""
    signal_stop = 0

    # 初始化
    def __init__(self):
        super(Widget, self).__init__()
        self.setupUi(self)
        pic = QPixmap("sources/background--.png")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pic))
        self.setPalette(palette)
        self.push_Photo_dir.clicked.connect(self.Photo_choice)
        self.push_video_dir.clicked.connect(self.video_choice)
        self.push_camera_info.clicked.connect(self.camera_choice)
        self.start.clicked.connect(self.start_fun)
        self.stop.clicked.connect(self.close_fun)
        self.mainwondow.setScaledContents(True)
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.play_radar_sound) #

    ##资源输入加载
    def Photo_choice(self):
        self.directory = QFileDialog.getOpenFileName(self, "选取文件", "./", "")
        self.Photo_dir.setText(self.directory[0])
        self.camera_info.setText("请选择摄像头")
        self.Video_dir.setText("请选择视频路径")
        self.source_parameter = self.directory[0]
        self.Video_Photo_camera = 0

    def video_choice(self):
        self.directory = QFileDialog.getOpenFileName(self, "选取文件", "./", "")
        self.Video_dir.setText(self.directory[0])
        self.Photo_dir.setText("请选择图片路径")
        self.camera_info.setText("请选择摄像头")
        self.source_parameter = self.directory[0]
        self.Video_Photo_camera = 1

    def camera_choice(self):
        self.camera_info.setText("0")
        self.Video_dir.setText("请选择图片路径")
        self.Photo_dir.setText("请选择视频路径")
        self.source_parameter = "0"
        self.Video_Photo_camera = 2

    # 启动函数
    def start_fun(self):
        sound_ck = self.sound.isChecked()
        signal_ck = self.signal_check.isChecked()
        people_ck = self.people_check.isChecked()
        car_ck = self.car_check.isChecked()

        self.start_model(self.source_parameter, sound_ck, signal_ck, people_ck, car_ck, self.Video_Photo_camera,
                         self.get_arg())

    # 获取参数
    def get_arg(self):
        if self.Video_Photo_camera == 0:
            return self.directory[0]
        elif self.Video_Photo_camera == 1:
            return self.directory[0]
        else:
            return 0

    # 关闭程序
    def close_fun(self):
        self.signal_stop = 1

    # 启动模型
    def start_model(self, source_parameter, sound_ck, signal_check, people_check, car_check, Video_Photo_camera, args):
        net = YOLO('other.pt')
        self.signal_stop = 0
        if Video_Photo_camera == 1 or Video_Photo_camera == 2:
            cap = cv2.VideoCapture(args)
            num = 1
            while cap.isOpened():
                if self.signal_stop == 1:
                    pix = QPixmap(":/Qt_icon/sources/carmer01.png")
                    self.mainwondow.setPixmap(pix)
                    break
                ret, img = cap.read()

                if ret:  # 若是读取成功
                    out = net.predict(source=img, verbose=False)
                    boxes = out[0].boxes.xyxy.cpu().detach()
                    cls = out[0].boxes.cls.reshape(-1, 1)
                    cnt = 0  # 检测种类数量
                    msgs = ""
                    for label_idx, (lx, ly, rx, ry) in zip(cls, boxes):
                        cnt += 1
                        label = out[0].names[int(label_idx.item())]
                        self.target_kind_display_3.setText(label)
                        lx, ly, rx, ry = list(map(int, (lx, ly, rx, ry)))

                        self.target_Local_xmin_display_2.setText(str(lx))
                        self.target_Local_Ymin_display_3.setText(str(ly))
                        self.target_Local_Xmax_display_4.setText(str(rx))
                        self.target_Local_Ymax_display_5.setText(str(ry))

                        if signal_check and label != "people" and label != "car":
                            cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(0, 255, 0))
                            msgs += "注意前方" + label + '\n'
                        if people_check and label == "people":
                            cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(255, 0, 0))
                        if car_check and label == "car":
                            cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(0, 0, 255))
                        # cv2.imshow("", img)
                        pred = str(out[0].boxes.conf[0].item())
                        self.Confidence_display_2.setText(pred[:4])
                    self.msg += msgs

                    self.msginfo.setText(self.msg)

                    num += 1
                    if sound_ck and (people_check or car_check) and cnt >= 1 and num % 3 == 0:
                        self.play_radar_sound()
                        num = 0
                    self.target_num_display.setText(str(cnt))
                    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, depth = rgb_image.shape
                    rgb_image = QtGui.QImage(rgb_image.data, width, height, width * depth, QtGui.QImage.Format_RGB888)
                    image = QtGui.QPixmap(rgb_image).scaled(rgb_image.width(), rgb_image.height())
                    self.mainwondow.setPixmap(image)
                    key = cv2.waitKey(1)  # 等待一段时间，并且检测键盘输入

                else:
                    pix = QPixmap(":/Qt_icon/sources/carmer01.png")
                    self.mainwondow.setPixmap(pix)
                    break
        else:

            img = cv2.imread(args)

            out = net.predict(source=img, verbose=False)

            boxes = out[0].boxes.xyxy.cpu().detach()
            cls = out[0].boxes.cls.reshape(-1, 1)
            cnt = 0  # 检测种类数量
            msgs = ""
            for label_idx, (lx, ly, rx, ry) in zip(cls, boxes):
                cnt += 1

                label = out[0].names[int(label_idx.item())]
                self.target_kind_display_3.setText(label)
                lx, ly, rx, ry = list(map(int, (lx, ly, rx, ry)))
                self.target_Local_xmin_display_2.setText(str(lx))
                self.target_Local_Ymin_display_3.setText(str(ly))
                self.target_Local_Xmax_display_4.setText(str(rx))
                self.target_Local_Ymax_display_5.setText(str(ry))
                if signal_check and label != "people" and label != "car":
                    cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(255, 0, 0))
                    msgs += "注意前方" + label + '\n'
                if people_check and label == "people":
                    cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(0, 0, 255))
                if car_check and label == "car":
                    cv2.rectangle(img, (lx, ly), (rx, ry), thickness=2, color=(0, 255, 0))

                pred = str(out[0].boxes.conf[0].item())
                self.Confidence_display_2.setText(pred[:4])
            self.msg += msgs

            self.msginfo.setText(self.msg)
            self.target_num_display.setText(str(cnt))
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, depth = rgb_image.shape
            rgb_image = QtGui.QImage(rgb_image.data, width, height, width * depth, QtGui.QImage.Format_RGB888)
            image = QtGui.QPixmap(rgb_image).scaled(rgb_image.width(), rgb_image.height())
            self.mainwondow.setPixmap(image)

    # 播放声音
    def play_radar_sound(self):
        sound_file = ":/Qt_icon/sources/sound.wav"  # 替换为你的音频文件路径
        QSound.play(sound_file)


# Login注册功能初始化
class Login(QWidget, Ui_Login):
    # Login功能初始化
    def __init__(self):
        super(Login, self).__init__()
        self.setupUi(self)
        pic = QPixmap(":/Qt_icon/sources/background.jpg")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pic))
        self.setPalette(palette)
        self.pwd.setEchoMode(QLineEdit.Password)

        self.login.clicked.connect(self.action_login)
        self.regiest.clicked.connect(self.action_regiest)

    # 登录确认功能
    def action_login(self):
        account = self.id.text()
        password = self.pwd.text()
        md = hashlib.md5(password.encode())  # 创建md5对象
        md5pwd = md.hexdigest()
        try:
            db = mysql.connector.connect(host="127.0.0.1", user="account", password="XnzchmFM2SJExEJR",
                                         database="account")

            cursor = db.cursor()
            query = "SELECT PWD FROM ID WHERE account = (%s) "
            cursor.execute(query, (account,))

            result = cursor.fetchone()

            if result is not None and result[0] == md5pwd:
                current_time = datetime.now()
                # 格式化时间为 "yyyy/mm/dd 00:00:00"

                formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")

                query = "UPDATE ID SET time = %s WHERE account = %s"
                cursor.execute(query, (formatted_time, account))
                db.commit()

                self.hide()
                self.widget = Widget()
                self.widget.show()

            else:
                QMessageBox.warning(self, "错误", "账号不存在或密码错误", QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.Yes)
        except mysql.connector.Error as e:
            QMessageBox.warning(self, "错误", e.msg, QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)
    # 转注册页面
    def action_regiest(self):
        self.hide()
        self.widget = Regiest()
        self.widget.show()


# Regiest注册功能
class Regiest(QWidget, Ui_Regiest):
    path_code = ""

    # Regiest注册功能初始化
    def __init__(self):
        super(Regiest, self).__init__()
        self.setupUi(self)
        pic = QPixmap(":/Qt_icon/sources/background.jpg")
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(pic))
        self.setPalette(palette)
        self.repwd.setEchoMode(QLineEdit.Password)
        self.regiest_check.clicked.connect(self.regiest)
        self.return_push.clicked.connect(self.back)
        self.path_code = self.generate_verification_code()
        verification_image = self.generate_verification_image(self.path_code)
        pixmap = self.pil_image_to_pixmap(verification_image)
        self.code_display.setPixmap(pixmap)
        self.reflush.clicked.connect(self.flush)
        self.pwd.setEchoMode(QLineEdit.Password)

    # 注册功能函数
    def regiest(self):
        reload_code = self.code_input.text()
        if self.pwd.text() != self.repwd.text():
            QMessageBox.warning(self, "提示", "两次密码不一致", QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)
        elif reload_code == self.path_code:
            user = self.id.text()
            pwd = self.pwd.text()
            md = hashlib.md5(pwd.encode())  # 创建md5对象
            md5pwd = md.hexdigest()
            current_time = datetime.now()
            # 格式化时间为 "yyyy/mm/dd 00:00:00"
            formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")
            try:
                db = mysql.connector.connect(host="127.0.0.1", user="account", password="XnzchmFM2SJExEJR",
                                             database="account")

                cursor = db.cursor()
                query = "INSERT INTO `ID` (`account`, `pwd`, `time`) VALUES (%s, %s, %s);"
                cursor.execute(query, (user, md5pwd, formatted_time))
                db.commit()
                QMessageBox.about(self, "成功", "账号注册成功")

                self.hide()
                self.widget = Login()
                self.widget.show()
            except mysql.connector.Error as e:
                if e.errno == 1062:

                    QMessageBox.warning(self, "错误", "账号已存在", QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes)
                else:
                    QMessageBox.warning(self, "错误", e.msg, QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes)
            else:
                db.close()
            self.flush()
        else:
            self.flush()
            QMessageBox.warning(self, "错误", "验证码输入错误", QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.Yes)

    # 返回主页面函数
    def back(self):
        self.hide()
        self.widget = Login()
        self.widget.show()

    # 刷新验证码函数
    def flush(self):
        self.path_code = self.generate_verification_code()
        verification_image = self.generate_verification_image(self.path_code)
        pixmap = self.pil_image_to_pixmap(verification_image)
        self.code_display.setPixmap(pixmap)

    # 获取验证码
    def generate_verification_code(self):
        code = ''.join(random.choices('0123456789abcdefghijklmnopqrstuvwxyz', k=4))
        return code

    # 生成验证码图片
    def generate_verification_image(self, code):
        image = Image.new('RGB', (150, 50), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        width, height = 150, 50
        font = ImageFont.truetype('arial.ttf', 30)
        draw.text((10, 10), code, fill=(0, 0, 0), font=font)
        for _ in range(4000):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

        return image

    # 将验证码图片加载到Qlabel
    def pil_image_to_pixmap(self, pil_image):
        image_data = pil_image.convert("RGBA").tobytes("raw", "RGBA")
        q_image = QImage(image_data, pil_image.size[0], pil_image.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
