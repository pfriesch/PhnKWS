
# Ubuntu packages
RUN apt-get update && apt-get install -y ssh gcc=4:5.3.1-1ubuntu1 g++=4:5.3.1-1ubuntu1 make vim zlib1g-dev libbz2-dev libssl-dev python-dev man libreadline-dev build-essential libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev python3 python3-dev

# Install Python 3.6 from source for optimizations
RUN cd /mnt/data/tmp && wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tgz && tar xzvf Python-3.7.0.tgz && cd Python-3.7.0 && ./configure && make && sudo make install && cd /mnt/data/tmp && wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py


RUN wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.6.tar.gz && 
 tar xzvf openfst-1.6.6.tar.gz && 
 cd openfst-1.6.6/ && 
 Run configure with python2

 ./configure --enable-far --enable-python && 
 vim 
 -c ":%s/PYTHON_CPPFLAGS = -I\/usr\/include\/python2.7/PYTHON_CPPFLAGS = -I\/opt\/anaconda3\/include\/" 
 -c ":%s/PYTHON_LDFLAGS = -L\/usr\/lib\/python2.7 -lpython2.7/PYTHON_LDFLAGS = -L\/opt\/anaconda3\/lib\/python3.7/" 
 -c ":%s/PYTHON_SITE_PKG = \/usr\/lib\/python2.7/PYTHON_SITE_PKG = \/opt\/anaconda3\/lib\/python3.7\/site-packages/" 
 -c ":%s/PYTHON_VERSION = 2.7/PYTHON_VERSION = 3.7/" 
 -c ":%s/lib\/python2.7\/dist-packages/opt\/anaconda3\/lib\/python3.7\/site-packages/" 
 -c ":wq" ./src/extensions/python/Makefile && 
 make && 
 make install && 
 echo 'export LD_LIBRARY_PATH="/usr/local/lib/:$LD_LIBRARY_PATH"' >> ~/.bashrc


Comment out line 229 in src/include/fst/float-weight.h:229 >> "static_assert(!TropicalWeight::NoWeight().Member(), "NoWeight not member");"