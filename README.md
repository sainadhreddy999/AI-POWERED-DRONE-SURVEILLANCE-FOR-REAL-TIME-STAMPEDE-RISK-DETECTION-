

# 🚀 Stampede Predictor


> A web application for real-time crowd density analysis and stampede risk prediction.


---


## 📌 Problem Statement


PS 3: Real-Time Stampede Risk Detection


---


## 🎯 Objective


The primary objective of the Stampede Predictor project is to develop and deploy an intelligent, real-time system capable of analyzing crowd dynamics from video feeds (live streams or uploaded media) to detect, predict, and prevent potential stampede situations. The system not only identifies individuals using deep learning but also analyzes crowd density distribution using grid-based methods and visualizes risk using heatmaps.

In addition to real-time monitoring, the system integrates advanced predictive capabilities such as panic probability estimation using LSTM-based models and future risk prediction based on temporal crowd patterns. It also incorporates multi-agent path planning to suggest safe evacuation routes dynamically, improving emergency response strategies.

By combining detection, prediction, visualization, and decision-support mechanisms, the system transforms traditional reactive crowd monitoring into a proactive, AI-driven safety solution. This enables authorities, event organizers, and public safety agencies to take timely preventive measures, significantly reducing the chances of stampede incidents and enhancing overall crowd safety.


---


## 🧠 Team & Approach

### Team Members:

- Katam Jnana Sumayi 
- C. Sainadha Reddy 
- Y. Shashidhar


### Your Approach:

- **Why you chose this problem:** We were motivated by the critical need for improved safety measures in crowded public spaces. Witnessing incidents in such environments underscored the importance of developing a tool that could proactively identify and alert to potential dangers before they escalate into stampedes. Leveraging AI for this purpose felt like a direct way to contribute to public safety.
- **Key challenges you addressed:** Significant challenges included achieving efficient real-time object detection (specifically, detecting people) in varying conditions, accurately calculating and interpreting crowd density across a defined grid, managing the streaming of video data and analysis results, ensuring the application could handle both live camera feeds and static file uploads, and integrating a messaging queue like Fluvio for decoupled data processing and monitoring. Building a responsive and intuitive user interface that clearly communicates risk levels was also a key focus.
- **Any pivots, brainstorms, or breakthroughs during hacking:** An initial consideration was using simpler image processing techniques, but brainstorming led us to adopt a deep learning approach with YOLO for more accurate and robust person detection. Integrating Fluvio was a crucial pivot that allowed us to separate the core analysis logic from the web presentation layer, making the system more scalable and enabling external consumption of the crowd data. Developing the Server-Sent Events (SSE) implementation for the live feed was a breakthrough that provided seamless, real-time status updates to the user interface without constant polling.


---


## 🛠️ Tech Stack


### Core Technologies Used:

- Frontend: HTML, CSS, JavaScript (for the web interface and live status updates via SSE)
- Backend: Flask (Python framework for the web server, handling requests and running the analysis)
- Database: None (The application focuses on real-time processing and data streaming rather than persistent storage of analysis results, though Fluvio could potentially feed into a database).
- APIs:
    - OpenCV (`cv2`): For video/image capture, processing, drawing overlays, and encoding frames.
    - Ultralytics (`YOLO`): Provides the implementation for the YOLOv11 Nano object detection model.
    - Fluvio: Used as a real-time messaging queue for streaming analysis data.
    Behavior Analysis & Prediction Modules: Custom Python modules (behavior.py, risk_prediction.py, panic_predictor.py, path_planning.py) for advanced analytics including panic prediction, future risk estimation, and evacuation planning.
- Hosting: Local Development Environment (designed for potential deployment on cloud platforms or edge devices).


### Sponsor Technologies Used:

- ✅ **Fluvio:** Fluvio is a mandatory component of this project, used for real-time data streaming. The Flask application (`app.py`) acts as a Fluvio producer, publishing structured data (including frame number, overall status, person count, and the detailed density grid) to the `crowd-data` topic for each processed frame or image. A separate Python script (`predict_stampede.py`) acts as a Fluvio consumer, subscribing to the `crowd-data` topic to receive and process this stream of real-time crowd metrics. This architecture demonstrates a decoupled approach where the analysis results are available for immediate consumption by other services or monitoring tools, enhancing the system's flexibility and scalability.



---


## ✨ Key Features



✅ Real-time Live Analysis: Performs continuous crowd monitoring using live video feeds with instant processing and visualization.
✅ AI-Powered Person Detection: Utilizes YOLOv11 Nano for fast and accurate real-time detection of individuals in crowded environments.
✅ Heatmap-Based Crowd Visualization: Generates smooth thermal heatmaps to highlight crowd density distribution instead of traditional bounding boxes.
✅ Grid-Based Density Analysis: Divides frames into grids to compute localized crowd density and detect high-risk regions precisely.
✅ Dynamic Risk Classification: Classifies crowd conditions into Normal, Warning, and Critical levels based on density thresholds.
✅ Panic Prediction (LSTM-Based): Predicts the probability of panic situations using temporal crowd behavior features such as movement patterns and density variations.
✅ Future Risk Prediction: Analyzes trends in crowd data to forecast potential risk escalation before it becomes critical.
✅ Multi-Agent Path Planning: Implements time-aware A* algorithm to generate safe evacuation paths while avoiding collisions in high-density areas.
✅ Behavior Analysis Module: Detects abnormal crowd movement patterns using velocity and directional analysis.
✅ Fluvio Data Streaming: Streams real-time crowd analytics data for scalable monitoring and integration with external systems.
✅ Interactive Web Dashboard: Provides real-time visualization, alerts, and monitoring through a Flask-based interface.
✅ Audio Alert System: Triggers alerts when critical crowd conditions are detected.
✅ Multi-Input Support: Supports both live camera feeds and uploaded image/video files for analysis.
✅ Session Insights & Reporting: Stores recent analysis results and generates downloadable reports for further evaluation.



## 🧪 How to Run the Project


### Requirements:

- Operating System: Linux or macOS (Recommended)
- Python 3.7+
- pip (Python package installer)
- Fluvio Cluster (Running locally or accessible)
- OpenCV library and its dependencies (usually installed with `opencv-python`)
- YOLO model file (`yolo11n.pt`) - The `ultralytics` library will attempt to download this automatically on the first run if it's not found locally.


### Local Setup:


1.  **Clone the repository:**

    ```bash
    git clone https://github.com/sainadhreddy999/AI-POWERED-DRONE-SURVEILLANCE-FOR-REAL-TIME-STAMPEDE-RISK-DETECTION-.git
    cd Stampede-Detection
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    This isolates project dependencies from your system Python environment.

    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all necessary Python packages listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```
    This command will install `Flask`, `opencv-python`, `ultralytics`, `numpy`, `fluvio`, `werkzeug`, `mimetypes`, and `shutil`.

4.  **Ensure Fluvio Cluster is Running:**
    The application requires a running Fluvio cluster to stream data. If you don't have Fluvio installed, follow the official Fluvio installation guide. Once installed, start a local cluster:

    ```bash
    fluvio cluster start
    ```
    Verify the `crowd-data` topic exists. If not, create it:

    ```bash
    fluvio topic create crowd-data
    ```

5.  **Run the Fluvio Consumer:**
    Open a **new terminal window** separate from where you will run the Flask app. Activate the virtual environment (steps from step 2). Then run the consumer script:

    ```bash
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    python predict_stampede.py
    ```
    This script will connect to your Fluvio cluster and subscribe to the `crowd-data` topic. It is configured to start reading from the end of the topic (`Offset.from_end(0)`), meaning it will only display messages produced *after* the consumer starts running. You will see analysis data printed to this terminal as the Flask app processes media.

6.  **Run the Flask Web Application:**
    Open a **third terminal window** (or reuse one if you don't need the consumer output visible simultaneously). Activate the virtual environment (steps from step 2). Then run the Flask application:

    ```bash
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    python app.py
    ```
    The application will start the Flask development server, typically hosting on `http://127.0.0.1:5000/`. It will attempt to load the YOLO model and connect to Fluvio during its startup process. Check the console output for confirmation of successful loading and connection.

7.  **Access the Web Application:**
    Open your preferred web browser and navigate to the application's address:

    * **Main Page (Upload):** `http://127.0.0.1:5000/`
    * **Live Feed Page:** `http://127.0.0.1:5000/live` (You can specify a camera index as a query parameter, e.g., `http://127.0.0.1:5000/live?camera=1`, if you have multiple webcams connected).

    Use the web interface to upload media files or start the live webcam analysis. Observe the output in your browser and, if running, the data stream in the `predict_stampede.py` terminal.


#### Optional: Adjusting Density Thresholds

If you need to fine-tune the sensitivity of the crowd density detection, you can adjust the threshold values directly in the `app.py` file:

1.  Open the `app.py` file in a text editor.
2.  Locate the "Density Analysis Settings" section.
3.  Modify the values for the following constants as needed:

    ```python
    HIGH_DENSITY_THRESHOLD = 5
    CRITICAL_DENSITY_THRESHOLD = 8
    HIGH_DENSITY_CELL_COUNT_THRESHOLD = 3
    CRITICAL_DENSITY_CELL_COUNT_THRESHOLD = 2

    DETECTION_THRESHOLD = 0.02125 # Confidence threshold for YOLO detections
    ```

    * `HIGH_DENSITY_THRESHOLD`: The number of persons in a grid cell to be considered high density.
    * `CRITICAL_DENSITY_THRESHOLD`: The number of persons in a grid cell to be considered critical density.
    * `HIGH_DENSITY_CELL_COUNT_THRESHOLD`: The number of high-density cells required to trigger a "High Density Warning" status.
    * `CRITICAL_DENSITY_CELL_COUNT_THRESHOLD`: The number of critical-density cells required to trigger a "CRITICAL RISK" status.

4.  Save the `app.py` file.
5.  Restart the Flask web application (`python app.py`) for the changes to take effect.

---


🧬 Future Scope (Updated)



📈 Advanced Predictive Analytics: Enhance the panic prediction and future risk modules using larger datasets and more sophisticated deep learning models to improve accuracy and early warning capabilities.
🧠 Improved LSTM Panic Modeling: Refine the LSTM-based panic prediction by incorporating additional behavioral features such as crowd flow rate, acceleration, and interaction patterns.
🚁 Autonomous Drone Integration: Extend the system to work with real drones, enabling automated navigation toward high-risk areas and dynamic surveillance coverage.
🧭 Advanced Multi-Agent Path Planning: Improve evacuation path generation by incorporating real-world constraints such as obstacles, exit capacities, and dynamic crowd behavior.
🌐 Cloud and Distributed Deployment: Deploy the system on cloud platforms to support large-scale real-time monitoring across multiple locations simultaneously.
📊 Smart Monitoring Dashboard: Develop a more advanced dashboard with live analytics, historical trends, and interactive visualizations for better decision-making.
🔊 Multi-Channel Alert System: Implement alert mechanisms through SMS, email, and mobile notifications to ensure faster response from authorities.
📱 Mobile Application Support: Build a mobile application for real-time monitoring and alerts, improving accessibility for field personnel.
⚙️ Edge AI Optimization: Optimize the system for deployment on edge devices to reduce latency and enable faster on-site processing.
🚧 Advanced Crowd Behavior Analysis: Extend analysis beyond density by incorporating movement patterns, anomaly detection, and behavioral intelligence for deeper insights.


## 📎 Resources / Credits


- Flask: The Python micro web framework used for the application backend.
- OpenCV: Essential library for computer vision tasks, including video processing and image manipulation.
- Ultralytics: Provides the easy-to-use implementation for the YOLO object detection models.
- NumPy: Fundamental package for scientific computing with Python, used for numerical operations on image data.
- Fluvio: The real-time data streaming platform used for decoupling analysis data from the web application.
- YOLOv11 Nano: The specific object detection model used for person detection due to its balance of speed and performance.


---
