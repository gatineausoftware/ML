FROM jupyterhub/jupyterhub

RUN pip install --upgrade pip

RUN npm install -g configurable-http-proxy && \
	python3 -m pip install jupyterhub && \
	python3 -m pip install --upgrade notebook;

RUN python --version
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN jupyterhub --generate-config
RUN useradd -m admin
RUN echo 'admin:admin' | chpasswd

# Should move these higher as these layers shouldn't work; not sure if dependent on jupyter install to work correctly?
# COPY . /home/admin/
COPY ML_Workbench.py /home/admin/
COPY ML_Workbench_demo.ipynb /home/admin/
COPY postalcode_info.json /home/admin/
COPY metadata_extract.py /home/admin/

CMD ["jupyterhub"]
