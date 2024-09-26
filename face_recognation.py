# library
import cv2 as cv

# objek 
# penggunan r / double slash untuk mengatasi masalah perizinan pada path cascade
objek_path = (r"D:\Productivity\belajar koding\python\python-menengah\citra_digital\face_detection\haarcascade_frontalface_alt.xml") 
acuan_gambar = cv.CascadeClassifier(objek_path) # mengambil dataset wajah
camera = cv.VideoCapture(0) # membuka kamera

# function
def deteksi_wajah(frame):
    grayscalling = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) # proses optimalisasi
    faces = acuan_gambar.detectMultiScale(grayscalling ,scaleFactor=1.1, minSize=(250,250) , minNeighbors=3) # deteksi wajah
    return faces

def drawer_box(frame):
    for x, y, w, h in deteksi_wajah(frame): # looping untuk menggambar kotak penanda
        cv.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 4) # BGR bukan RGB

def close_window():
    camera.release() # matiin kamera
    cv.destroyAllWindows() # matiin library opencv nya
    exit() # exit

def main():
    while True :
        _, frame = camera.read() # baca frame dari kamera
        drawer_box(frame) # panggil function drawer box
        cv.imshow("Deteksi Muka", frame) # tampilin window hasil deteksi
        
        # kondisonal untuk matiin program
        if cv.waitKey(1) & 0xFF == ord('q'): 
            close_window() # panggil function close window

# jalanin function
if __name__ == '__main__':
    main()

# deteksi eror
# if acuan_gambar.empty():
#     print(" haar cascade tidak bisa dimuat")
# else:
#     print(" haar cascade berhasil dimuat")