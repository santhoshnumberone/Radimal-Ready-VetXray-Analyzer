# Radimal-Ready VetXray Analyzer

**A Hybrid AI Pipeline for Radiology Report Generation**

This project is a proof-of-concept demonstrating an end-to-end AI pipeline capable of analyzing radiographs and patient data, a crucial first step for a solution like the one being built at Radimal. I built this to showcase my ability to handle diverse data types, train multi-modal models, and deploy a complete solution using modern MLOps practices.

---

### **Project Goal**

The objective of this project was to build a robust and transferable AI pipeline that combines computer vision and structured data analysis to make a confident prediction about an X-ray image. This serves as a foundational blueprint for a real-world application that could generate radiology reports for veterinary clinics.

### **Key Features**

* **DICOM Proficiency:** The pipeline begins with a demonstration of handling real DICOM (Digital Imaging and Communications in Medicine) files, extracting both rich metadata and image data, which is a fundamental requirement for medical imaging projects.
* **Hybrid AI Architecture:** I built a multi-modal system that uses a pre-trained CNN (MobileNetV2) to analyze the X-ray image and an XGBoost classifier to integrate structured patient data (age and gender). This ensemble approach enhances predictive accuracy and resilience.
* **Blueprint for Veterinary Data:** To ensure a reliable ground truth, the model was trained on a publicly available, expert-labeled human dataset (NIH Chest X-ray Dataset, 2017). This project explicitly demonstrates a pipeline that is directly applicable and ready to be used with veterinary-specific data once it becomes available.
* **Containerized Inference:** The entire application is packaged into a Docker container. This ensures that the pipeline is portable, reproducible, and can be deployed consistently across different environments, from local machines to production servers.

### **Technical Stack**

* **Languages:** Python 3.10
* **Computer Vision:** PyTorch, torchvision, Pillow
* **Structured Data:** XGBoost, scikit-learn, pandas
* **Data Handling:** pydicom, NumPy
* **MLOps:** Docker
* **Other:** Jupyter Notebooks, Git

### **How to Run**

The simplest way to run this project is with Docker. Ensure you have Docker installed and running on your machine.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/santhoshnumberone/Radimal-Ready-VetXray-Analyzer.git
    cd Radimal-Ready-VetXray-Analyzer
    ```

2.  **Build the Docker Image:**
    This command will build the Docker image, pulling all the necessary dependencies and code.
    ```bash
    docker build -t radimal-analyzer .
    ```

3.  **Run the Container:**
    This command will run the container and execute the main inference script (`predict.py`).
    ```bash
    docker run radimal-analyzer
    ```
    You will see the final prediction results printed directly to your terminal.

### Model Training and Performance
The graphs below show the model's performance during the training process. The loss steadily decreases, and accuracy increases, indicating that the model is effectively learning from the data without significant overfitting.
<img width="597" height="651" alt="Screenshot 2025-08-06 at 4 40 15 PM" src="https://github.com/user-attachments/assets/b5458ee4-37f3-4a74-89e2-1f921fed0684" />

### Docker Run result
<img width="381" height="233" alt="Screenshot 2025-08-06 at 4 43 44 PM" src="https://github.com/user-attachments/assets/cb46b67a-d334-43a8-87b7-e74748d76ab9" />


### **Data Source**

Due to the size of the image data (~4.2 GB), the image files are not included in this repository.

To run the project, you need to download the `sample.zip` file, which contains the X-ray images, from the Kaggle page of the [NIH Chest X-ray Dataset Sample](https://www.kaggle.com/datasets/nih-chest-xrays/sample?resource=download).

1.  Download the `sample.zip` file.
2.  Unzip the contents and place the `images` folder directly into the `data` directory of this project.
