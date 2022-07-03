# Specify the base image
FROM tensorflow/tensorflow:2.8.2

# Update the package manager and install a simple module. The RUN command
# will execute a command on the container and then save a snapshot of the
# results. The last of these snapshots will be the final image
RUN apt-get update -y && apt-get install -y zip graphviz wget

# Make sure the contents of our repo are in /app
COPY . /app

# # Install the SCIP solver
# RUN dpkg -i /app/dependencies/SCIPOptSuite-8.0.0-Linux-ubuntu.deb

# Install additional Python packages
RUN pip install --upgrade pip
# RUN pip install jupyter==1.0.0 pandas==1.4.3 scikit-learn==1.1.1 matplotlib==3.5.2 \
#     ipympl==0.9.1 ortools==9.3.10497 pydot==1.4.2 rise==5.7.1 jupyter_contrib_nbextensions==0.5.1 \
#     tables==3.7.0 tensorflow-lattice==2.0.10
RUN pip install jupyter==1.0.0 pandas==1.4.3 scikit-learn==1.1.1 matplotlib==3.5.2 \
    ipympl==0.9.1 ortools==9.3.10497 pydot==1.4.2 rise==5.7.1 jupyter_contrib_nbextensions==0.5.1 \
    tables==3.7.0 tensorflow-lattice==2.0.10
RUN jupyter contrib nbextension install --system

# Specify working directory
WORKDIR /app/notebooks

# Use CMD to specify the starting command
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", \
     "--ip=0.0.0.0", "--allow-root"]
