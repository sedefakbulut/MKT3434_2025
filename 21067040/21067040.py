import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QPushButton, QLabel,
                             QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                             QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                             QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, mean_absolute_error
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.problem_type = None

        # Neural network configuration
        self.layer_config = []

        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()

        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)

        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)

        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])

        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)

        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    def load_dataset(self):
        """Load selected dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()

            if dataset_name == "Load Custom Dataset":
                return

            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.problem_type = "classification"
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                self.update_classical_loss_options()
                self.update_deep_learning_loss_options()
                return

            # Split data
            test_size = self.split_spin.value()
            self.X_train, self.X_test, self.y_train, self.y_test = \
                model_selection.train_test_split(data.data, data.target,
                                                 test_size=test_size,
                                                 random_state=42)

            # Problem tipini belirle
            if len(np.unique(self.y_train)) > 10:
                self.problem_type = "regression"
            else:
                self.problem_type = "classification"

            # Apply scaling if selected
            self.apply_scaling()

            self.status_bar.showMessage(f"Loaded {dataset_name}")
            self.update_classical_loss_options()
            self.update_deep_learning_loss_options()

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )

            if file_name:
                # Load data
                data = pd.read_csv(file_name)

                # Ask user to select target column
                target_col = self.select_target_column(data.columns)

                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]

                    # Split data
                    test_size = self.split_spin.value()
                    self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(X, y,
                                                         test_size=test_size,
                                                         random_state=42)

                    # Problem tipini belirle
                    if len(np.unique(self.y_train)) > 10:
                        self.problem_type = "regression"
                    else:
                        self.problem_type = "classification"

                    # Apply scaling if selected
                    self.apply_scaling()

                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    self.update_classical_loss_options()
                    self.update_deep_learning_loss_options()

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)

        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)

        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()

        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()

                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")

    def create_tabs(self):
        """Create the main tabs for different ML categories"""
        self.tabs = QTabWidget()
        self.tab_classical_ml = self.create_classical_ml_tab()
        self.tab_deep_learning = self.create_deep_learning_tab()
        self.tab_reinforcement_learning = QWidget() # Placeholder
        self.tabs.addTab(self.tab_classical_ml, "Classical ML Algorithms")
        self.tabs.addTab(self.tab_deep_learning, "Deep Learning")
        self.tabs.addTab(self.tab_reinforcement_learning, "Reinforcement Learning (Coming Soon)")
        self.layout.addWidget(self.tabs)

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Kayıp Fonksiyonu Seçimi
        loss_group = QGroupBox("Loss Function")
        loss_layout = QHBoxLayout()
        self.loss_combo_classical = QComboBox()
        self.update_classical_loss_options() # Başlangıçta seçenekleri ayarla
        loss_layout.addWidget(QLabel("Loss:"))
        loss_layout.addWidget(self.loss_combo_classical)
        loss_group.setLayout(loss_layout)
        layout.addWidget(loss_group, 0, 0, 1, 2) # İlk satırın tamamını kaplasın

        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        regression_layout.addWidget(lr_group)

        # SVM Regression (SVR)
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {"kernel": ["linear", "rbf", "poly"],
             "C": "double",
             "epsilon": "double"}
        )
        regression_layout.addWidget(svr_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 1, 0)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        classification_layout.addWidget(logistic_group)

        # Naive Bayes
        nb_params = {"var_smoothing": "double",
                     "priors": ["uniform", "user_defined"],
                     "user_defined_priors": "user_defined_priors"}
        self.nb_group = self.create_algorithm_group(
            "Naive Bayes",
            nb_params
        )
        classification_layout.addWidget(self.nb_group)

        # SVM Classification (SVC)
        svc_group = self.create_algorithm_group(
            "Support Vector Classification",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int",
             "probability": "checkbox"} # Olasılık tahmini için
        )
        classification_layout.addWidget(svc_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 1, 1)

        return widget

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # Kayıp Fonksiyonu Seçimi (Derin Öğrenme)
        loss_group_dl = QGroupBox("Loss Function")
        loss_layout_dl = QHBoxLayout()
        self.loss_combo_dl = QComboBox()
        self.update_deep_learning_loss_options()
        loss_layout_dl.addWidget(QLabel("Loss:"))
        loss_layout_dl.addWidget(self.loss_combo_dl)
        loss_group_dl.setLayout(loss_layout_dl)
        layout.addWidget(loss_group_dl, 0, 0, 1, 2)

        # Veri Ön İşleme Bölümü
        preprocess_group_dl = QGroupBox("Data Preprocessing")
        preprocess_layout_dl = QVBoxLayout()

        missing_value_label = QLabel("Missing Values:")
        self.missing_value_combo = QComboBox()
        self.missing_value_combo.addItems(["None", "Mean Imputation", "Interpolation", "Forward Fill", "Backward Fill"])
        preprocess_layout_dl.addWidget(missing_value_label)
        preprocess_layout_dl.addWidget(self.missing_value_combo)

        preprocess_group_dl.setLayout(preprocess_layout_dl)
        layout.addWidget(preprocess_group_dl, 1, 0, 1, 2)

        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()

        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)

        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)

        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)

        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 2, 0)

        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()

        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)

        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 2, 1)

        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()

        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)

        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 3, 0)

        return widget

    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()

        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))

            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1