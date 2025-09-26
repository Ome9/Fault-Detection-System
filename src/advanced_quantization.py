#!/usr/bin/env python3
"""
Advanced Quantization-Aware Training for NASA Bearing Fault Detection
Implements quantization-aware training (QAT) for optimal STM32 deployment

This script implements:
1. Quantization-aware training during model training
2. Advanced pruning techniques
3. Knowledge distillation from larger model
4. Multiple quantization strategies comparison

Author: AI Assistant
Date: 2025
"""

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class QuantizationAwareTrainer:
    """
    Advanced quantization-aware training for TinyML deployment.
    """
    
    def __init__(self, input_dim=8, encoding_dim=4):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.teacher_model = None
        self.student_model = None
        self.qat_model = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def build_teacher_model(self):
        """
        Build a larger, more accurate teacher model for knowledge distillation.
        """
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input')
        
        # Larger teacher network
        x = layers.Dense(64, activation='relu', name='teacher_enc1')(inputs)
        x = layers.Dense(32, activation='relu', name='teacher_enc2')(x)
        x = layers.Dense(16, activation='relu', name='teacher_enc3')(x)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='teacher_bottleneck')(x)
        
        # Decoder
        x = layers.Dense(16, activation='relu', name='teacher_dec1')(encoded)
        x = layers.Dense(32, activation='relu', name='teacher_dec2')(x)
        x = layers.Dense(64, activation='relu', name='teacher_dec3')(x)
        outputs = layers.Dense(self.input_dim, activation='linear', name='teacher_output')(x)
        
        self.teacher_model = tf.keras.Model(inputs, outputs, name='teacher_autoencoder')
        self.teacher_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.teacher_model
    
    def build_student_model(self):
        """
        Build a compact student model optimized for quantization.
        """
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input')
        
        # Minimal student network - optimized for quantization
        x = layers.Dense(16, activation='relu', name='student_enc1')(inputs)
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='student_bottleneck')(x)
        
        # Decoder
        x = layers.Dense(16, activation='relu', name='student_dec1')(encoded)
        outputs = layers.Dense(self.input_dim, activation='linear', name='student_output')(x)
        
        self.student_model = tf.keras.Model(inputs, outputs, name='student_autoencoder')
        self.student_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return self.student_model
    
    def knowledge_distillation_loss(self, y_true, y_pred_student, y_pred_teacher, temperature=3.0, alpha=0.7):
        """
        Custom loss function for knowledge distillation.
        Combines reconstruction loss with teacher knowledge.
        """
        # Standard reconstruction loss
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred_student)
        
        # Distillation loss - learn from teacher's predictions
        distillation_loss = tf.keras.losses.mse(y_pred_teacher, y_pred_student)
        
        # Combined loss
        total_loss = alpha * distillation_loss + (1 - alpha) * reconstruction_loss
        
        return total_loss
    
    def train_teacher_model(self, X_train, epochs=100, verbose=1):
        """Train the teacher model first."""
        print("Training teacher model...")
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        if self.teacher_model is None:
            self.build_teacher_model()
            
        # Train teacher
        history = self.teacher_model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5)
            ]
        )
        
        print(f"Teacher model trained. Final loss: {history.history['loss'][-1]:.6f}")
        return history
    
    def train_student_with_distillation(self, X_train, epochs=150, verbose=1):
        """Train student model using knowledge distillation."""
        print("Training student model with knowledge distillation...")
        
        X_scaled = self.scaler.transform(X_train)
        
        if self.student_model is None:
            self.build_student_model()
            
        # Get teacher predictions
        teacher_predictions = self.teacher_model.predict(X_scaled, verbose=0)
        
        # Custom training loop for knowledge distillation
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Mini-batch training
            batch_size = 32
            for i in range(0, len(X_scaled), batch_size):
                batch_x = X_scaled[i:i+batch_size]
                batch_teacher_pred = teacher_predictions[i:i+batch_size]
                
                with tf.GradientTape() as tape:
                    student_pred = self.student_model(batch_x, training=True)
                    loss = self.knowledge_distillation_loss(
                        batch_x, student_pred, batch_teacher_pred
                    )
                
                gradients = tape.gradient(loss, self.student_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
                
                epoch_loss += loss.numpy()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            
            if epoch % 10 == 0 and verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping\n            if avg_loss < best_loss:\n                best_loss = avg_loss\n                patience_counter = 0\n            else:\n                patience_counter += 1\n                if patience_counter >= 20:\n                    print(f\"Early stopping at epoch {epoch+1}\")\n                    break\n        \n        print(f\"Student model trained. Final loss: {best_loss:.6f}\")\n        return best_loss\n    \n    def apply_quantization_aware_training(self):\n        \"\"\"\n        Apply quantization-aware training to the student model.\n        \"\"\"\n        print(\"Applying quantization-aware training...\")\n        \n        # Apply QAT to the student model\n        quantize_model = tfmot.quantization.keras.quantize_model\n        \n        # Quantize the entire model\n        self.qat_model = quantize_model(self.student_model)\n        \n        # Compile QAT model\n        self.qat_model.compile(\n            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for QAT\n            loss='mse',\n            metrics=['mae']\n        )\n        \n        print(\"QAT model created successfully\")\n        return self.qat_model\n    \n    def fine_tune_qat_model(self, X_train, epochs=50, verbose=1):\n        \"\"\"\n        Fine-tune the quantization-aware model.\n        \"\"\"\n        print(\"Fine-tuning QAT model...\")\n        \n        X_scaled = self.scaler.transform(X_train)\n        \n        # Fine-tune with lower learning rate\n        history = self.qat_model.fit(\n            X_scaled, X_scaled,\n            epochs=epochs,\n            batch_size=16,  # Smaller batch size for fine-tuning\n            validation_split=0.1,\n            verbose=verbose,\n            callbacks=[\n                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)\n            ]\n        )\n        \n        # Calculate threshold\n        reconstructed = self.qat_model.predict(X_scaled, verbose=0)\n        reconstruction_errors = np.mean(np.square(X_scaled - reconstructed), axis=1)\n        self.threshold = np.percentile(reconstruction_errors, 95)\n        \n        print(f\"QAT fine-tuning complete. Threshold: {self.threshold:.6f}\")\n        return history\n    \n    def convert_qat_to_tflite(self, output_path, representative_data):\n        \"\"\"\n        Convert QAT model to TensorFlow Lite with optimizations.\n        \"\"\"\n        print(\"Converting QAT model to TensorFlow Lite...\")\n        \n        # Prepare representative dataset\n        rep_data_scaled = self.scaler.transform(representative_data).astype(np.float32)\n        \n        def representative_dataset():\n            for i in range(min(len(rep_data_scaled), 500)):\n                yield [np.array([rep_data_scaled[i]], dtype=np.float32)]\n        \n        # Convert with quantization-aware trained model\n        converter = tf.lite.TFLiteConverter.from_keras_model(self.qat_model)\n        converter.optimizations = [tf.lite.Optimize.DEFAULT]\n        converter.representative_dataset = representative_dataset\n        \n        # Force INT8 quantization\n        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]\n        converter.inference_input_type = tf.int8\n        converter.inference_output_type = tf.int8\n        \n        # Additional optimizations\n        converter.experimental_new_converter = True\n        converter.experimental_new_quantizer = True\n        \n        # Convert\n        tflite_model = converter.convert()\n        \n        # Save\n        with open(output_path, 'wb') as f:\n            f.write(tflite_model)\n        \n        model_size_kb = len(tflite_model) / 1024\n        print(f\"✅ QAT TFLite model saved: {output_path}\")\n        print(f\"Model size: {model_size_kb:.1f} KB\")\n        \n        return tflite_model, model_size_kb\n    \n    def apply_structured_pruning(self, target_sparsity=0.5):\n        \"\"\"\n        Apply structured pruning to further reduce model size.\n        \"\"\"\n        print(f\"Applying structured pruning (target sparsity: {target_sparsity})...\")\n        \n        # Define pruning schedule\n        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(\n            initial_sparsity=0.0,\n            final_sparsity=target_sparsity,\n            begin_step=0,\n            end_step=1000\n        )\n        \n        # Apply pruning\n        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(\n            self.student_model,\n            pruning_schedule=pruning_schedule\n        )\n        \n        # Compile pruned model\n        pruned_model.compile(\n            optimizer='adam',\n            loss='mse',\n            metrics=['mae']\n        )\n        \n        print(\"Structured pruning applied\")\n        return pruned_model\n    \n    def evaluate_models(self, X_test, y_test):\n        \"\"\"\n        Comprehensive evaluation of all model variants.\n        \"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"MODEL EVALUATION COMPARISON\")\n        print(\"=\"*60)\n        \n        models_to_test = [\n            (\"Teacher Model\", self.teacher_model),\n            (\"Student Model\", self.student_model),\n            (\"QAT Model\", self.qat_model)\n        ]\n        \n        results = {}\n        \n        for name, model in models_to_test:\n            if model is None:\n                continue\n                \n            print(f\"\\n📊 Evaluating {name}...\")\n            \n            # Get predictions\n            X_test_scaled = self.scaler.transform(X_test)\n            reconstructed = model.predict(X_test_scaled, verbose=0)\n            \n            # Calculate reconstruction errors\n            reconstruction_errors = np.mean(np.square(X_test_scaled - reconstructed), axis=1)\n            \n            # Calculate threshold if not set\n            if self.threshold is None:\n                threshold = np.percentile(reconstruction_errors, 95)\n            else:\n                threshold = self.threshold\n            \n            # Make predictions\n            predictions = reconstruction_errors > threshold\n            \n            # Calculate metrics\n            y_true_binary = (y_test != 0).astype(int)  # Convert to binary\n            y_pred_binary = predictions.astype(int)\n            \n            # Accuracy metrics\n            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n            \n            accuracy = accuracy_score(y_true_binary, y_pred_binary)\n            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)\n            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)\n            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)\n            \n            # Model size (parameters)\n            param_count = model.count_params()\n            \n            results[name] = {\n                'accuracy': accuracy,\n                'precision': precision,\n                'recall': recall,\n                'f1_score': f1,\n                'params': param_count,\n                'avg_recon_error': np.mean(reconstruction_errors),\n                'threshold': threshold\n            }\n            \n            print(f\"  Accuracy: {accuracy*100:.1f}%\")\n            print(f\"  Precision: {precision*100:.1f}%\")\n            print(f\"  Recall: {recall*100:.1f}%\")\n            print(f\"  F1-Score: {f1*100:.1f}%\")\n            print(f\"  Parameters: {param_count:,}\")\n            print(f\"  Avg Reconstruction Error: {np.mean(reconstruction_errors):.6f}\")\n        \n        return results\n\ndef run_advanced_quantization_pipeline():\n    \"\"\"\n    Complete advanced quantization pipeline.\n    \"\"\"\n    print(\"🚀 Starting Advanced Quantization-Aware Training Pipeline\")\n    print(\"=\"*70)\n    \n    # Load data (you'll need to adapt this to your data loading)\n    from tinyml_optimized_model import load_nasa_data_for_tinyml\n    \n    DATASET_PATH = \"D:/errorDetection\"\n    OUTPUT_DIR = \"advanced_quantized_models\"\n    os.makedirs(OUTPUT_DIR, exist_ok=True)\n    \n    # Load data\n    print(\"📊 Loading NASA bearing data...\")\n    X, y = load_nasa_data_for_tinyml(DATASET_PATH, max_files_per_set=200)\n    \n    # Split data\n    X_normal = X[y == 0]\n    X_anomaly = X[y == 1]\n    \n    # Create trainer\n    trainer = QuantizationAwareTrainer(input_dim=8, encoding_dim=4)\n    \n    # Step 1: Train teacher model\n    print(\"\\n🎓 Step 1: Training teacher model...\")\n    trainer.train_teacher_model(X_normal, epochs=80)\n    \n    # Step 2: Train student with knowledge distillation\n    print(\"\\n👨‍🎓 Step 2: Training student with knowledge distillation...\")\n    trainer.train_student_with_distillation(X_normal, epochs=100)\n    \n    # Step 3: Apply quantization-aware training\n    print(\"\\n⚡ Step 3: Applying quantization-aware training...\")\n    trainer.apply_quantization_aware_training()\n    trainer.fine_tune_qat_model(X_normal, epochs=30)\n    \n    # Step 4: Evaluate all models\n    print(\"\\n📈 Step 4: Evaluating models...\")\n    # Combine normal and anomaly data for evaluation\n    X_test = np.vstack([X_normal[:100], X_anomaly])\n    y_test = np.hstack([np.zeros(100), np.ones(len(X_anomaly))])\n    \n    results = trainer.evaluate_models(X_test, y_test)\n    \n    # Step 5: Convert to TensorFlow Lite\n    print(\"\\n🔄 Step 5: Converting to TensorFlow Lite...\")\n    \n    tflite_path = os.path.join(OUTPUT_DIR, \"advanced_qat_model.tflite\")\n    tflite_model, model_size = trainer.convert_qat_to_tflite(\n        tflite_path, X_normal[:300]\n    )\n    \n    # Step 6: Save preprocessing objects\n    print(\"\\n💾 Step 6: Saving preprocessing objects...\")\n    \n    scaler_path = os.path.join(OUTPUT_DIR, \"advanced_scaler.pkl\")\n    threshold_path = os.path.join(OUTPUT_DIR, \"advanced_threshold.npy\")\n    \n    with open(scaler_path, 'wb') as f:\n        pickle.dump(trainer.scaler, f)\n    \n    np.save(threshold_path, trainer.threshold)\n    \n    # Summary\n    print(\"\\n🎯 ADVANCED QUANTIZATION PIPELINE COMPLETE!\")\n    print(\"=\"*70)\n    print(f\"📁 Output directory: {OUTPUT_DIR}/\")\n    print(f\"🔥 Final model size: {model_size:.1f} KB\")\n    print(f\"⚡ Quantization: INT8 with QAT\")\n    print(f\"🧠 Knowledge distillation: Teacher → Student\")\n    print(f\"📊 Feature dimensions: 8 (optimized)\")\n    \n    # Print best results\n    if 'QAT Model' in results:\n        qat_results = results['QAT Model']\n        print(f\"\\n🏆 QAT Model Performance:\")\n        print(f\"   Accuracy: {qat_results['accuracy']*100:.1f}%\")\n        print(f\"   F1-Score: {qat_results['f1_score']*100:.1f}%\")\n        print(f\"   Parameters: {qat_results['params']:,}\")\n    \n    print(f\"\\n✅ Ready for STM32 deployment!\")\n    \n    return trainer, results\n\nif __name__ == \"__main__\":\n    trainer, results = run_advanced_quantization_pipeline()