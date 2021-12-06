import tkinter as tk
from PIL import Image, ImageTk
from pglcm import *
from klasifikasi import *

class MyFirstGUI:
    
    def __init__(self, master):
        global file_citra
        global in_glcm
        global hasil_glcm
        global fitur_0, fitur_45, fitur_90, fitur_135
        self.master = master
        master.title("KLASIFIKASI JENIS KAYU METODE GLCM DAN BACKPROPAGATION")
        master.geometry("800x500")
        master.configure(bg='#bca674')

        self.lbNama = tk.Label(master, text='nama file', anchor='center', bg='#e6dcbc', fg='#201314')
        self.lbNama.place(x=30, y=65, width=260, height=30)
        self.lbNama.configure(font=("Helvetica", 12))
        
        self.btPilih = tk.Button(master, text="PILIH\n GAMBAR", bd=1, cursor="hand2", command=self.fungsi_open, bg='#e6dcbc', fg='#201314')
        self.btPilih.place(x=30, y=105, width=110, height=60)
        self.btPilih.configure(font=("Helvetica", 10, 'bold'))
        
        self.btRun = tk.Button(master, text="RUN", bd=1, cursor="hand2", command=self.fungsi_proses, bg='#e6dcbc', fg='#201314')
        self.btRun.place(x=180, y=105, width=110, height=60)
        self.btRun.configure(font=("Helvetica", 10, 'bold'))
        
        self.btReset = tk.Button(master, text="RESET", bd=1, cursor="hand2", command=self.fungsi_reset, bg='#e6dcbc', fg='#201314')
        self.btReset.place(x=60, y=180, width=200, height=40)
        self.btReset.configure(font=("Helvetica", 10, 'bold'))
       
        self.citra_test = tk.Label(master, bg='#e6dcbc')
        self.citra_test.place(x=370, y=80, width=180, height=110)
        
        self.citra_gray = tk.Label(master, bg='#e6dcbc')
        self.citra_gray.place(x=595, y=80, width=180, height=110)
        
        # Button
        #frame 3: hasil glcm
        
        self.frame3 = tk.Frame(master, bg='#947147')
        self.frame3.place(x=20, y=260, width=450, height=220)
        
        
        #------------Fitur GLCM-------------
        self.lbKontras = tk.Label(self.frame3, text='CONTRAST', anchor='w', bg='#947147', fg='white')
        self.lbKontras.place(x=10, y=22, width=130, height=35)
        self.lbKontras.configure(font=("Helvetica", 12))
        
        self.lbhKontras = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhKontras.place(x=150, y=22, width=280, height=30)
        self.lbhKontras.configure(font=("Helvetica", 12))
        
        self.lbHomog = tk.Label(self.frame3, text='ENTROPY', anchor='w', bg='#947147', fg='white')
        self.lbHomog.place(x=10, y=62, width=130, height=20)
        self.lbHomog.configure(font=("Helvetica", 12))
        
        self.lbhHomog = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhHomog.place(x=150, y=62, width=280, height=30)
        self.lbhHomog.configure(font=("Helvetica", 12))
        
        self.lbDissi = tk.Label(self.frame3, text='ENERGY', anchor='w', bg='#947147', fg='white')
        self.lbDissi.place(x=10, y=102, width=130, height=20)
        self.lbDissi.configure(font=("Helvetica", 12))
        
        self.lbhDissi = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhDissi.place(x=150, y=102, width=280, height=30)
        self.lbhDissi.configure(font=("Helvetica", 12))
        
        self.lbEnergy = tk.Label(self.frame3, text='HOMOGENEITY', anchor='w', bg='#947147', fg='white')
        self.lbEnergy.place(x=10, y=142, width=130, height=20)
        self.lbEnergy.configure(font=("Helvetica", 12))
        
        self.lbhEnergy = tk.Label(self.frame3, anchor='e', bg='white')
        self.lbhEnergy.place(x=150, y=142, width=280, height=30)
        self.lbhEnergy.configure(font=("Helvetica", 12))
        
        self.lbGLCM = tk.Label(self.frame3, text='GLCM', anchor='center', bg='#947147', fg='white')
        self.lbGLCM.place(x=0, y=182, width=450, height=30)
        self.lbGLCM.configure(font=("Helvetica", 14, 'bold'))
                
        #frame 3: hasil backpro
        
        self.frame4 = tk.Frame(master, bg='#947147')
        self.frame4.place(x=505, y=260, width=265, height=220)
        
        self.lbBPNN = tk.Label(self.frame4, text='HASIL', anchor='center', bg='#947147', fg='white')
        self.lbBPNN.place(x=0, y=182, width=180, height=30)
        self.lbBPNN.configure(font=("Helvetica", 14, 'bold'))
        
        self.hsBPNN = tk.Label(self.frame4, anchor='w', bg='white')
        self.hsBPNN.place(x=10, y=25, width=160, height=60)
        self.hsBPNN.configure(font=("Helvetica", 14))        


    def greet(self):
        print("Greetings!")
    
    def fungsi_reset(self):
        self.lbNama.configure(text="")
        # self.hsPNN.configure(text="")
        self.citra_test.configure(image='')
        self.citra_gray.configure(image='')
        self.hsBPNN.configure(text="")

        
    def fungsi_open(self):
        global file_citra
        print("-----------------KLASIFIKASI BARU---------------------")
        
        self.nama_file =  tk.filedialog.askopenfilename(initialdir = "/kayu/dataset",title = "Select Image",filetypes = (("all files","*.*"),("png files","*.png")))
        self.citra_kayu = Image.open(self.nama_file)
        self.citra_kayu = self.citra_kayu.resize((180,110), Image.ANTIALIAS)
        self.citra_kayu = ImageTk.PhotoImage(self.citra_kayu)
        self.lbNama.configure(text = self.nama_file)
        
        self.citra_test.configure(image=self.citra_kayu)
        self.citra_test.image=self.citra_kayu
        print (self.nama_file)
        file_citra = self.nama_file
        return file_citra
    
    def fungsi_proses(self):
        
        global file_citra
        global in_glcm
        print("tombol proses di klik")
        # memanggil fungsi ekstraksi yang ada di file pglcm.py
        self.fitur, self.hasil_prepro = ekstraksi(file_citra)
        self.hasil_pre = Image.fromarray(self.hasil_prepro)
        self.hasil_pre = self.hasil_pre.resize((180,110), Image.ANTIALIAS)
        self.hasil_pre = ImageTk.PhotoImage(self.hasil_pre)
        
        self.citra_gray.configure(image=self.hasil_pre)
        self.citra_gray.image=self.citra_gray
        print (self.fitur)
        self.lbhKontras.configure(text=self.fitur[0])
        self.lbhHomog.configure(text=self.fitur[1])
        self.lbhDissi.configure(text=self.fitur[2])
        self.lbhEnergy.configure(text=self.fitur[3])
        in_glcm = self.hasil_prepro
        # memanggil fungsi klasifikasi yang ada di file klasifikasi.py
        self.kelas = klasifikasiBP(self.fitur)
        if self.kelas == 0:
            print ("Hasil : Jati")
            self.hasil = 'Jati'
        elif self.kelas == 1:
            print ("Hasil : Kelapa")
            self.hasil = 'Kelapa'
        elif self.kelas == 2:
            print ("Hasil : Nangka")
            self.hasil = 'Nangka'
        elif self.kelas == 3:
            print ("Hasil : Sengon")
            self.hasil = 'Sengon'
        elif self.kelas == 4:
            print ("Hasil : Suren")
            self.hasil = 'Suren'
        elif self.kelas == 5:
            print ("Hasil : Mindi")
            self.hasil = 'Mindi'
        elif self.kelas == 6:
            print ("Hasil : mahoni")
            self.hasil = 'Mahoni'
        
        self.hsBPNN.configure(text=self.hasil)

        

root = tk.Tk()
my_gui = MyFirstGUI(root)
root.mainloop()