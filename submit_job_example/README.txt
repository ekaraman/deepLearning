HPC Cluster job running example.

HPC uzerinde job run etmek icin login.kuacc.ku.edu.tr sunucusuna ssh ile baglanilir.

Aciklamalar asagidaki sitede bulunabilir.

http://login.kuacc.ku.edu.tr/

HPC uzerinde job submit etmek icin kullanilabilecek ornek job submit scriptleri
login.kuacc.ku.edu.tr sunucusu uzerinde /kuacc/jobscripts dizininde bulunur.

submit.sh benim tarafimdan python scripti run edecek sekilde example1_submit.sh ornek alinarak hazilanmistir.

"sbatch submit.sh" komutu ile test.py scripti cluster uzerinde calistirilir.

test.py scripti icersindeki commentleri message.log dosyasina yazar.

Kullandigim python modullerini yukleyebilmek icin home dizinine anaconda kurdum.

Anaconda kurulumu icin asagidaki sitedeki adimlari kullandim:

https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart

Anaconda kurulumdan sonra python komuutunun home dizinimde bulunan anaconda icinde oldugu gordum:

[emrekaraman14@login03 submit_job_example]$ which python
~/anaconda3/bin/python
[emrekaraman14@login03 submit_job_example]$ python --version
Python 3.6.5 :: Anaconda, Inc.

Paket kurulumu icin kullanilan pip modulune update ettim ve pip3 geldi.

pytorch modulunu asagidaki komut ile kurdum:

https://pytorch.org/

pip3 install torch torchvision

CUDA hellow world hazirlamak icin asagidaki adreste bulunan orengi kullandim:

http://computer-graphics.se/hello-world-for-cuda.html

GPU Cuda ve PyTorch test etmek icin:

sbatch submit_gpu.sh


