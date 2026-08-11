# Smart City Issue Detection System - 100 Interview Questions

Here are 100 technical and behavioral interview questions based on the "Smart City Issue Detection System" project, tailored for Python, AI/ML, and Data Science roles. They cover everything from basic Python to advanced System Design.

---

## Part 1: Project-Specific Architecture & Overview (1-10)

**1. How did you structure your Flask application, and how did you manage object-oriented principles within it?**
*Hint: Discuss how YOLO models and the Flask app were instantiated, and how you encapsulated database initialization and analytics functions away from the API routing layer.*

**2. Explain your use of Python generators or context managers in this project. Why were they necessary?**
*Hint: Mention iterating over YOLO bounding boxes (`result.boxes`) and using `io.BytesIO()` as a context manager for base64 encoding during image processing.*

**3. In a production environment, SQLite might not be sufficient. How would you handle database migration, and what strategies would you use to process heavy ML inferences without blocking the main Flask API threads?**
*Hint: Discuss migrating to PostgreSQL (and PostGIS for spatial data), and using message queues like RabbitMQ or Celery with Redis for asynchronous video/image processing.*

**4. How did you handle class imbalance in your dataset (e.g., if there were significantly more potholes than garbage images)?**
*Hint: Talk about data augmentation techniques (flip, rotate, brightness) applied via Roboflow and threshold adjustments.*

**5. What evaluation metrics did you use to evaluate your YOLOv8 model, and how did you balance precision and recall for this specific use case?**
*Hint: Discuss mAP (Mean Average Precision). Explain precision in the context of avoiding false positives (like shadows detected as potholes) and recall for avoiding false negatives (missing actual garbage).*

**6. If you notice model drift occurring after deploying the system in the real world (e.g., due to different weather conditions), how would you detect and address it?**
*Hint: Mention implementing MLOps tools like MLflow or Weights & Biases to track performance, and setting up automated retraining pipelines as new data is collected.*

**7. Why did you choose YOLOv8 over other object detection architectures (like Faster R-CNN or SSD)?**
*Hint: Explain the "You Only Look Once" single-stage architecture, how it divides the image into a grid, and its balance between accuracy and real-time inference speed (using YOLOv8m).*

**8. Explain the concept of Non-Maximum Suppression (NMS) and why it's critical in your object detection pipeline.**
*Hint: Describe how NMS removes redundant or overlapping bounding boxes that predict the same object, ensuring clean output.*

**9. When deploying this system to edge devices (like garbage trucks or CCTV cameras), what optimization techniques would you apply to the model to reduce latency and memory consumption?**
*Hint: Discuss using lighter models (YOLOv8n) and converting the weights (`.pt`) to ONNX or NVIDIA TensorRT formats.*

**10. Walk me through the complete architecture of your application from the moment a user uploads an image to when the dashboard analytics are updated.**
*Hint: Trace the flow from the Client Browser -> Flask Backend -> YOLO Inference -> Severity/Department Calculation -> SQLite Storage -> Dynamic Dashboard Rendering.*

---

## Part 2: Python Fundamentals (11-20)

**11. What are the key differences between Python lists and tuples, and when would you use each in a data processing pipeline?**

**12. Explain the concept of list comprehensions in Python and provide an example of how you might use one to filter bounding box predictions.**

**13. How does Python handle memory management and garbage collection?**

**14. What are Python decorators, and how can they be used in a Flask application (e.g., for routing or authentication)?**

**15. Explain the difference between deep copy and shallow copy in Python.**

**16. How do you handle exceptions in Python? Why is catching bare `Exception` generally frowned upon?**

**17. What is the Global Interpreter Lock (GIL) in Python, and how does it affect multithreading in your Flask application?**

**18. Differentiate between `__str__` and `__repr__` methods in Python classes.**

**19. What is a lambda function, and where might you use it when processing Pandas DataFrames?**

**20. Explain the use of the `yield` keyword. How does it differ from `return`?**

---

## Part 3: Advanced Python & Object-Oriented Programming (21-30)

**21. What are the four pillars of Object-Oriented Programming, and how are they implemented in Python?**

**22. Explain the concept of Multiple Inheritance in Python and the Method Resolution Order (MRO).**

**23. What are Python "magic methods" (dunder methods)? Provide examples of how you might use them in a custom dataset class.**

**24. How can you implement a Singleton design pattern in Python, and in what scenario (like model loading) might it be useful?**

**25. Describe the purpose of `@staticmethod` and `@classmethod`. When would you choose one over the other?**

**26. What is the difference between an iterator and an iterable in Python?**

**27. How does Python's `asyncio` work, and how does it compare to threading for IO-bound tasks like fetching images from a camera stream?**

**28. Explain duck typing in Python with an example.**

**29. What are abstract base classes (abc module) in Python, and why would you use them?**

**30. How would you optimize a Python script that is processing a massive CSV file that doesn't fit in memory?**

---

## Part 4: Web Development & APIs (Flask) (31-40)

**31. How does Flask's application context differ from the request context?**

**32. What is a RESTful API? What principles did you follow when designing your `/predict` endpoint?**

**33. How do you handle cross-origin resource sharing (CORS) in a Flask application?**

**34. Explain the difference between HTTP GET and POST requests. When should each be used?**

**35. How would you implement rate limiting on your API to prevent abuse?**

**36. Discuss strategies for securing a Flask application in a production environment.**

**37. How can you perform background tasks in Flask without blocking the main request thread (e.g., using Celery)?**

**38. What are Flask Blueprints, and how do they help in organizing large applications?**

**39. How would you handle file uploads in Flask securely, considering malicious users might upload executable scripts?**

**40. What is WSGI, and why do you need a WSGI server like Gunicorn instead of Flask's built-in server for production?**

---

## Part 5: Database Management (SQLite/PostgreSQL) (41-50)

**41. What are the ACID properties in database management, and why are they important?**

**42. Explain the difference between inner, left, right, and full joins in SQL.**

**43. What is an index in a database, and how does it speed up query performance? Are there any downsides?**

**44. How does SQLite differ from PostgreSQL, and why is PostgreSQL recommended for your scalable architecture?**

**45. What is SQL injection, and how do you prevent it when using raw SQL queries or an ORM like SQLAlchemy?**

**46. Describe database normalization and denormalization. When might denormalization be useful?**

**47. How would you model a many-to-many relationship in a relational database (e.g., Reports and Tags)?**

**48. Explain the concept of database migrations and why tools like Alembic are used.**

**49. What is a transaction in SQL, and how do you ensure data integrity using `COMMIT` and `ROLLBACK`?**

**50. How would you optimize a slow-running SQL query that aggregates data for your web dashboard?**

---

## Part 6: Data Science, EDA & Preprocessing (51-60)

**51. What steps do you take during Exploratory Data Analysis (EDA) on a new image dataset?**

**52. How do you handle missing or corrupted images in your training dataset?**

**53. Explain the bias-variance tradeoff in machine learning.**

**54. What are some common data augmentation techniques for object detection, and why are they effective?**

**55. How do you ensure your training, validation, and test splits are representative of the real-world data distribution?**

**56. Explain the concept of data leakage and how it can occur during the preprocessing phase.**

**57. What is feature scaling (e.g., normalization vs. standardization), and is it necessary for Convolutional Neural Networks (CNNs)?**

**58. How do you handle outliers in a tabular dataset?**

**59. What is cross-validation, and why might it be difficult to implement for large image datasets?**

**60. Explain the difference between generative and discriminative models.**

---

## Part 7: Machine Learning Fundamentals (61-70)

**61. Explain the difference between supervised, unsupervised, and reinforcement learning.**

**62. How does a Random Forest algorithm work, and what are its advantages over a single Decision Tree?**

**63. What is the Curse of Dimensionality, and how do techniques like PCA (Principal Component Analysis) address it?**

**64. Explain how Support Vector Machines (SVMs) classify data. What is the kernel trick?**

**65. What is the difference between L1 (Lasso) and L2 (Ridge) regularization?**

**66. How does Gradient Descent work? Explain the difference between Batch, Mini-batch, and Stochastic Gradient Descent.**

**67. What are the common reasons a machine learning model might overfit, and how can you prevent it?**

**68. Explain the ROC curve and the AUC metric. When would you use them instead of accuracy?**

**69. What is a confusion matrix, and how do you calculate precision, recall, and F1-score from it?**

**70. In a highly imbalanced dataset, why is accuracy a misleading metric? What alternative metrics or sampling strategies would you use?**

---

## Part 8: Computer Vision & Deep Learning Basics (71-80)

**71. What is a Convolutional Neural Network (CNN)? Explain the role of convolutional and pooling layers.**

**72. Why are non-linear activation functions like ReLU important in deep learning?**

**73. What is the vanishing gradient problem, and how do architectures like ResNet (Residual Networks) solve it?**

**74. Explain the concept of transfer learning and fine-tuning. Why did you use it for your YOLO model?**

**75. What is the difference between image classification, object detection, and image segmentation?**

**76. How do color spaces (e.g., RGB, HSV, Grayscale) impact computer vision algorithms?**

**77. Explain how the OpenCV library is used in your project for image preprocessing and visualization.**

**78. What is the role of an optimizer (like Adam or SGD) in training a neural network?**

**79. What is batch normalization, and why is it used in modern CNN architectures?**

**80. Explain the concept of Intersection over Union (IoU) and its significance in object detection evaluation.**

---

## Part 9: YOLO Architecture & Advanced Object Detection (81-90)

**81. Explain the evolution of the YOLO architecture. What major improvements does YOLOv8 introduce over its predecessors?**

**82. YOLO is a "single-stage" detector. How does it differ from "two-stage" detectors like Faster R-CNN in terms of architecture and performance?**

**83. What are anchor boxes, and how are they used in object detection models?**

**84. How does YOLO handle detecting small objects, which is often a challenge in computer vision?**

**85. Explain the loss function used in object detection (bounding box regression loss vs. classification loss).**

**86. What is Mean Average Precision (mAP), and how is mAP@0.5 different from mAP@0.5:0.95?**

**87. How would you go about training a custom YOLO model on a dataset with very limited examples (few-shot learning)?**

**88. What is the impact of image resolution on YOLO's inference speed and accuracy?**

**89. How do you implement object tracking (e.g., DeepSORT) on top of YOLO detections for video streams?**

**90. Explain how converting a PyTorch model to TensorRT or ONNX improves inference speed on edge devices.**

---

## Part 10: System Design, MLOps, & Behavioral (91-100)

**91. Design a system architecture for a real-time smart city traffic monitoring application that processes 10,000 CCTV feeds simultaneously.**

**92. What is model drift (concept drift and data drift), and how would you monitor it in a production environment?**

**93. Explain the principles of CI/CD for machine learning (CT - Continuous Training).**

**94. How would you containerize your ML application using Docker? What considerations are needed for GPU support?**

**95. Discuss the trade-offs between deploying ML models on the cloud (cloud inference) versus edge devices (edge inference).**

**96. Describe a time you faced a difficult bug in a machine learning pipeline. How did you debug and resolve it?**

**97. How do you stay updated with the rapidly evolving field of AI and Deep Learning?**

**98. If a stakeholder asks you to achieve 100% accuracy on this Smart City project, how would you manage their expectations?**

**99. Explain a situation where you had to choose between a simple, interpretable model and a complex, highly accurate one. What was your decision process?**

**100. What do you consider the biggest ethical challenges in deploying AI-powered surveillance systems in a smart city, and how would you mitigate them?**
