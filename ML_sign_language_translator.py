import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from datetime import datetime
import logging

async def load_classifier():
    dataset_file = "D:\dataset\hand_landmarks.csv"
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found at {dataset_file}")
    
    df = pd.read_csv(dataset_file, header=None)
    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Predictions for metrics
    y_pred = knn.predict(X_train_scaled)

    # Calculate and display classification metrics
    acc = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_train, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred, average='weighted', zero_division=0)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nDetailed Report:\n", classification_report(y_train, y_pred, zero_division=0))

    # Nearest neighbor distances
    distances, _ = knn.kneighbors(X_train_scaled)
    max_distance = np.max(distances)
    avg_distance = np.mean(distances)
    print(f"Max distance: {max_distance:.4f}, Avg distance: {avg_distance:.4f}")
    
    return scaler, knn, X_train

async def main():
    try:
        scaler, knn, X_train = await load_classifier()
        print("Classifier loaded successfully!")
        
        async def handler(websocket, path):
            print(f"Client connected: {websocket.remote_address}")
            async for message in websocket:
                try:
                    data = json.loads(message)
                    landmarks = data.get("landmarks", "")
                    if not landmarks:
                        await websocket.send(json.dumps({"error": "No landmarks"}))
                        continue

                    flat_landmarks = np.array([float(x) for x in landmarks.split(",")])
                    scaled = scaler.transform([flat_landmarks])

                    distances, _ = knn.kneighbors(scaled, n_neighbors=1)
                    if distances[0][0] > 8.5:  # Adjustable threshold
                        await websocket.send("Not recognized")
                    else:
                        prediction = knn.predict(scaled)[0]
                        await websocket.send(str(prediction))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"error": "Invalid JSON"}))
                except Exception as e:
                    await websocket.send(json.dumps({"error": f"Server error: {str(e)}"}))
        
        server = await websockets.serve(handler, "0.0.0.0", 5000)
        print("Server running on ws://0.0.0.0:5000")
        await server.wait_closed()
    
    except Exception as e:
        print(f"Failed to start server: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
