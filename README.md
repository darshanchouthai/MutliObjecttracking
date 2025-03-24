# 🚀 Multi-Object Tracking using Reinforcement Learning  

An AI-powered **real-time object tracking system** integrating **YOLOv5** and **Reinforcement Learning (PPO)** to track multiple objects dynamically in live video streams.  

![Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYTgwZWIzNGYyM2U0NzM0NDdlYTVmZmViY2QyMGMxMWI0MTYxMGQ1NyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/Y4Uv6WCNWl36X4s4kF/giphy.gif)  

## 📌 Features  
✔ **Real-time object detection** using **YOLOv5**  
✔ **Multi-object tracking** with **PPO-based Reinforcement Learning**  
✔ **Flask-based web application** for real-time video processing  
✔ **Path visualization** for detected objects  
✔ **Customizable tracking classes**  
✔ **Supports live camera feed & video input**  

## ⚙️ Tech Stack  
- **Machine Learning**: YOLOv5, PPO (Proximal Policy Optimization)  
- **Deep Learning**: TensorFlow, OpenCV  
- **Backend**: Flask  
- **Web Framework**: HTML, CSS, JavaScript  
- **Data Handling**: NumPy, Pandas  
- **Deployment**: PythonAnywhere / Local  

## 🚀 Installation & Usage  
```bash
# 1️⃣ Clone the Repository  
git clone https://github.com/darshanchouthai/MultiObjectTracking.git  
cd MultiObjectTracking  

# 2️⃣ Install Dependencies  
pip install -r requirements.txt  

# 3️⃣ Run the Flask App  
python app.py  

# The web app will be available at http://127.0.0.1:5000/  
```

## 📜 Project Structure  
```bash
📂 MultiObjectTracking  
├── 📁 models/               # Pretrained YOLOv5 & PPO models  
├── 📁 static/               # Frontend assets (CSS, JS, images)  
├── 📁 templates/            # HTML files for web interface  
├── 📄 app.py                # Main Flask application  
├── 📄 train_ppo.py          # RL model training script  
├── 📄 requirements.txt      # Required dependencies  
└── 📄 README.md             # Project documentation  
```

## 🎯 Future Improvements  
🔹 Implement **object re-identification (ReID)** for long-term tracking.  
🔹 Optimize **model inference speed** using TensorRT.  
🔹 Add **support for edge devices** (Raspberry Pi, Jetson Nano).  

## 🤝 Contributing  
Contributions are welcome! Feel free to **fork, open issues, or submit PRs**.  

## 📬 Contact  
💡 **Darshan Pradeepkumar Chouthayi**  
📧 [Email](mailto:darshanchouthai@gmail.com) | 🐙 [GitHub](https://github.com/darshanchouthai) | 🔗 [LinkedIn](https://www.linkedin.com/in/darshan-chouthayi-8697b225a)  

🌟 **Star this repo if you found it useful!** ⭐  
