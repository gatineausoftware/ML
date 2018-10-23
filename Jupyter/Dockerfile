FROM jupyterhub/jupyterhub

RUN pip install --upgrade pip

RUN npm install -g configurable-http-proxy && \
	python3 -m pip install jupyterhub && \
	python3 -m pip install --upgrade notebook;

#RUN python --version
#COPY requirements.txt .
#RUN python3 -m pip install -r requirements.txt

RUN jupyterhub --generate-config
RUN useradd -m admin
RUN echo 'admin:admin' | chpasswd



CMD ["jupyterhub"]
