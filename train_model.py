import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

class GestureDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.processor = processor
        self.image_paths = []
        self.labels = []
        
        # Get all gesture classes
        self.gesture_classes = sorted([d for d in os.listdir(data_dir) 
                                    if os.path.isdir(os.path.join(data_dir, d))])
        self.label2id = {label: i for i, label in enumerate(self.gesture_classes)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        # Collect all images and labels
        for gesture in self.gesture_classes:
            gesture_dir = os.path.join(data_dir, gesture)
            for img_name in os.listdir(gesture_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(gesture_dir, img_name))
                    self.labels.append(self.label2id[gesture])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = self.labels[idx]
        return inputs

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

def train_model():
    print("Starting model training...")
    
    # Load the pre-trained model and processor
    model_name = "dima806/smart_tv_hand_gestures_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Create dataset
    dataset = GestureDataset("training_data", processor)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        no_cuda=not torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Training model...")
    trainer.train()
    
    # Save the fine-tuned model
    print("Saving model...")
    model.save_pretrained("./fine_tuned_model")
    processor.save_pretrained("./fine_tuned_model")
    
    print("Training complete! Model saved in ./fine_tuned_model")

if __name__ == "__main__":
    train_model() 