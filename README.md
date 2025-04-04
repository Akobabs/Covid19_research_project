project_root/
│
├── data/
│   ├── raw/
│   │   ├── xray_images/
│   │   ├── cough_sounds/
│   │   └── temperature_readings/
│   ├── processed/
│   │   ├── xray_images/
│   │   ├── cough_sounds/
│   │   └── temperature_readings/
│   └── augmentation/
│       ├── xray_images/
│       ├── cough_sounds/
│       └── temperature_readings/
│
├── models/
│   ├── cnn/
│   ├── rnn/
│   ├── mlp/
│   └── multimodal/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training_cnn.ipynb
│   ├── model_training_rnn.ipynb
│   ├── model_training_mlp.ipynb
│   ├── multimodal_fusion.ipynb
│   └── real_time_pipeline.ipynb
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── train_cnn.py
│   ├── train_rnn.py
│   ├── train_mlp.py
│   ├── multimodal_fusion.py
│   └── real_time_pipeline.py
│
├── results/
│   ├── logs/
│   ├── model_checkpoints/
│   └── evaluation/
│
├── docs/
│   ├── project_overview.md
│   ├── data_description.md
│   ├── model_architecture.md
│   ├── real_time_pipeline.md
│   └── gui_development.md
│
├── gui/
│   ├── main.py
│   ├── interface/
│   └── assets/
│
├── requirements.txt
└── README.md
