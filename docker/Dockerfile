ARG PYTORCH="1.0"
ARG CUDA="10.0"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV DEBIAN_FRONTEND noninteractive

# figure cuda repository error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# install necessary tools
RUN apt-get update && apt-get install -y ffmpeg \
    libsm6 libxext6 git ninja-build libglib2.0-0 \
    libsm6 libxrender-dev libxext6 openssh-server \
    && apt-get install -yq curl wget jq sudo vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set passwd
RUN echo "root:tracyst605c" | chpasswd

# Config SSH
RUN git clone https://github.com/Taited/DockerHub \
    && apt-get install openssh-server -y\
    && mkdir /root/.ssh \
    && cat ./DockerHub/A100_ted.pub >> /root/.ssh/authorized_keys \
    && chmod 777 /etc/ssh/sshd_config \
    && rm /etc/ssh/sshd_config \
    && mv ./DockerHub/sshd_config /etc/ssh/ \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && rm -rf DockerHub \
    && touch /root/startup_run.sh \
    && chmod +x /root/startup_run.sh \
    && echo "#!/bin/bash" >> /root/startup_run.sh \
    && echo "LOGTIME=$(date \"+%Y-%m-%d %H:%M:%S\")" >> /root/startup_run.sh \
    && echo "echo \"[$LOGTIME] startup run...\" >>/root/startup_run.log" >> /root/startup_run.sh\
    && echo "service ssh start >>/root/startup_run.log" >> /root/startup_run.sh \
    && echo "if [ -f /root/startup_run.sh ]; then" >> /root/.bashrc \
    && echo "    /root/startup_run.sh" >> /root/.bashrc \
    && echo "fi" >> /root/.bashrc

# config environment variables
RUN echo "export PATH=/opt/conda/lib/python:$PATH" >> /root/.bashrc \
    && echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> /root/.bashrc \
    && echo "LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> /root/.bashrc \
    && echo "export CUDA_HOME=/usr/local/cuda" >> /root/.bashrc

