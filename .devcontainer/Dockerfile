FROM igibson/igibson:latest

RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    psmisc \
    vim \
    wget \
    curl \
    openssh-client \
    libglib2.0-0 \
    htop \
    less \
    && rm -rf /var/lib/apt/lists/*

# fzf
RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install --all

# RUN pip3 install igibson==2.2.0
RUN pip3 install stable-baselines3==1.4.0
RUN pip3 install wandb

WORKDIR /workspaces

CMD bash
