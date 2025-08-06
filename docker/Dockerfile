# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
# NOTE: We'll copy just the necessary folders and files to keep the image small
COPY inference/ inference/
COPY models/ models/
COPY data/sample_labels.csv data/
COPY data/images/ data/images/

# NOTE: This line is to copy a sample DICOM, you may remove it if you don't need it for your demo.
COPY data/image-00001.dcm data/

# Set the command to run the script
CMD ["python", "inference/predict.py"]
